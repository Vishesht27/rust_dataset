import pandas as pd
import requests
import json
import argparse
from tqdm import tqdm
import random
from ast import literal_eval
from task_category import TaskCategory
import asyncio
import aiohttp
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Lock

# Logging will be configured after argument parsing
logger = logging.getLogger(__name__)

# Thread-safe counter for progress tracking
progress_lock = Lock()

# You'll need to replace this with your actual API key
OPENROUTER_KEY = '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'

LLM_MODEL_MAPPING = {
    "gemini": "google/gemini-2.5-pro",
    "claude": "anthropic/claude-sonnet-4", 
    "gpt4": "openai/gpt-4o",
}

parser = argparse.ArgumentParser(description="LLM validation for non-code Rust dataset tasks.")
parser.add_argument("--input_filepath", type=str, help="Input file path", required=True)
parser.add_argument("--output_filepath", type=str, help="Output file path", required=True)
parser.add_argument("--nsamples_per_task", type=int, default=3, help="Number of samples per non-code task")
parser.add_argument("--llm_model", type=str, choices=LLM_MODEL_MAPPING.keys(), default="claude", help="LLM model to use")
parser.add_argument("--max_workers", type=int, default=5, help="Number of parallel API calls")
parser.add_argument("--resume_from", type=int, default=0, help="Resume from specific row index")
parser.add_argument("--checkpoint_every", type=int, default=10, help="Save checkpoint every N completed validations")
parser.add_argument("--retry_attempts", type=int, default=3, help="Number of retry attempts for failed API calls")
parser.add_argument("--validate_all", action='store_true', help="Validate all samples instead of sampling")
parser.add_argument("--max_samples_per_task", type=int, default=None, help="Maximum samples per task when using --validate_all")
parser.add_argument("--log_filename", type=str, default="llm_validation.log", help="Custom log filename for this run")

args = parser.parse_args()

# Setup logging with custom filename
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(args.log_filename),
        logging.StreamHandler()
    ],
    force=True  # Override any existing configuration
)

# Define validation prompts for each non-code task
VALIDATION_PROMPTS = {
    TaskCategory.DOCSTRING_GENERATION: """
You are a Rust documentation expert. Your task is to evaluate if the generated docstring accurately describes the given Rust code.

Evaluation Criteria:
1. **Accuracy**: Does the docstring correctly describe what the code does?
2. **Completeness**: Does it cover the main functionality, parameters, and return values?
3. **Clarity**: Is the explanation clear and understandable?
4. **Rust conventions**: Does it follow Rust documentation conventions?

You must return your response in the following JSON format:
{
  "Is_Answer_Correct": true/false,
  "Corrected": "if Is_Answer_Correct is true return empty string, else return the correct docstring",
  "Reasoning": "Explain why the docstring is correct/incorrect and what improvements were made if any"
}
""",

    TaskCategory.CODE_SUMMARIZATION: """
You are a Rust code analysis expert. Your task is to evaluate if the generated summary accurately captures what the given Rust code does.

Evaluation Criteria:
1. **Accuracy**: Does the summary correctly describe the code's behavior?
2. **Conciseness**: Is it appropriately brief while covering key points?
3. **Technical correctness**: Are technical terms and concepts used correctly?
4. **Completeness**: Does it capture the essential functionality without being verbose?

You must return your response in the following JSON format:
{
  "Is_Answer_Correct": true/false,
  "Corrected": "if Is_Answer_Correct is true return empty string, else return the correct summary",
  "Reasoning": "Explain why the summary is correct/incorrect and what improvements were made if any"
}
""",

    TaskCategory.CODE_EXPLANATION: """
You are a Rust programming instructor. Your task is to evaluate if the generated explanation helps someone understand the given Rust code.

Evaluation Criteria:
1. **Educational value**: Would this help someone learn/understand the code?
2. **Technical accuracy**: Are all technical details explained correctly?
3. **Clarity**: Is the explanation easy to follow and well-structured?
4. **Depth**: Does it explain both what the code does and how/why it works?

You must return your response in the following JSON format:
{
  "Is_Answer_Correct": true/false,
  "Corrected": "if Is_Answer_Correct is true return empty string, else return the correct explanation",
  "Reasoning": "Explain why the explanation is correct/incorrect and what improvements were made if any"
}
""",

    TaskCategory.FUNCTION_NAMING: """
You are a Rust API design expert. Your task is to evaluate if the suggested function name is appropriate for the given function implementation.

Evaluation Criteria:
1. **Semantic accuracy**: Does the name clearly indicate what the function does?
2. **Rust conventions**: Does it follow Rust naming conventions (snake_case, descriptive)?
3. **Clarity**: Is the name self-explanatory and unambiguous?
4. **Appropriateness**: Is it neither too generic nor too specific?

You must return your response in the following JSON format:
{
  "Is_Answer_Correct": true/false,
  "Corrected": "if Is_Answer_Correct is true return empty string, else return the better function name",
  "Reasoning": "Explain why the function name is correct/incorrect and what makes the corrected name better if any"
}
""",

    TaskCategory.VARIABLE_NAMING: """
You are a Rust code quality expert. Your task is to evaluate if the suggested variable name is appropriate for the given code context.

Evaluation Criteria:
1. **Semantic accuracy**: Does the name clearly indicate what the variable represents?
2. **Rust conventions**: Does it follow Rust naming conventions (snake_case, descriptive)?
3. **Context appropriateness**: Does the name make sense in the given code context?
4. **Clarity**: Is the name self-explanatory and helps code readability?

You must return your response in the following JSON format:
{
  "Is_Answer_Correct": true/false,
  "Corrected": "if Is_Answer_Correct is true return empty string, else return the better variable name",
  "Reasoning": "Explain why the variable name is correct/incorrect and what makes the corrected name better if any"
}
"""
}

def parse_data_safely(data):
    """Safely parse input/output data that might be JSON, literal_eval, or already parsed."""
    if not isinstance(data, str):
        return data  # Already parsed
    
    # Try JSON first (most common)
    try:
        return json.loads(data)
    except:
        pass
    
    # Try literal_eval (for Python dict/list strings)
    try:
        return literal_eval(data)
    except:
        pass
    
    # If both fail, return as string wrapped in dict
    return {"raw_data": data}

def create_validation_prompt(task_category, input_data, output_data):
    """Create a validation prompt for a specific task."""
    
    system_prompt = VALIDATION_PROMPTS[task_category]
    
    if task_category == TaskCategory.DOCSTRING_GENERATION:
        input_dict = parse_data_safely(input_data)
        output_dict = parse_data_safely(output_data)
        
        user_prompt = f"""
**Rust Code:**
```rust
{input_dict.get('code', input_dict.get('raw_data', 'N/A'))}
```

**Code Context (if any):**
```rust
{input_dict.get('code_context', 'None provided')}
```

**Generated Docstring:**
{output_dict.get('docstring', output_dict.get('raw_data', 'N/A'))}

**Task:** Evaluate if this docstring is correct for the given Rust code.
"""

    elif task_category == TaskCategory.CODE_SUMMARIZATION:
        input_dict = parse_data_safely(input_data)
        output_dict = parse_data_safely(output_data)
        
        user_prompt = f"""
**Rust Code:**
```rust
{input_dict.get('code', input_dict.get('raw_data', 'N/A'))}
```

**Code Context (if any):**
```rust
{input_dict.get('code_context', 'None provided')}
```

**Generated Summary:**
{output_dict.get('summary', output_dict.get('raw_data', 'N/A'))}

**Task:** Evaluate if this summary is correct for the given Rust code.
"""

    elif task_category == TaskCategory.CODE_EXPLANATION:
        input_dict = parse_data_safely(input_data)
        output_dict = parse_data_safely(output_data)
        
        user_prompt = f"""
**Rust Code:**
```rust
{input_dict.get('code', input_dict.get('raw_data', 'N/A'))}
```

**Code Context (if any):**
```rust
{input_dict.get('code_context', 'None provided')}
```

**Generated Explanation:**
{output_dict.get('explanation', output_dict.get('raw_data', 'N/A'))}

**Task:** Evaluate if this explanation is correct for the given Rust code.
"""

    elif task_category == TaskCategory.FUNCTION_NAMING:
        input_dict = parse_data_safely(input_data)
        output_dict = parse_data_safely(output_data)
        
        user_prompt = f"""
**Function Implementation:**
```rust
{input_dict.get('code', input_dict.get('raw_data', 'N/A'))}
```

**Code Context (if any):**
```rust
{input_dict.get('code_context', 'None provided')}
```

**Suggested Function Name:**
{output_dict.get('function_name', output_dict.get('raw_data', 'N/A'))}

**Task:** Evaluate if this function name is appropriate for the given implementation.
"""

    elif task_category == TaskCategory.VARIABLE_NAMING:
        input_dict = parse_data_safely(input_data)
        output_dict = parse_data_safely(output_data)
        
        user_prompt = f"""
**Code with Placeholder:**
```rust
{input_dict.get('code', input_dict.get('raw_data', 'N/A'))}
```

**Code Context (if any):**
```rust
{input_dict.get('code_context', 'None provided')}
```

**Suggested Variable Name:**
{output_dict.get('variable_name', output_dict.get('raw_data', 'N/A'))}

**Task:** Evaluate if this variable name is appropriate for the given code context.
"""
    
    return system_prompt, user_prompt

def call_llm_api_with_retry(system_prompt, user_prompt, row_idx, max_retries=3):
    """Make API call to LLM for validation with retry logic."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL_MAPPING[args.llm_model],
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Row {row_idx}: API call attempt {attempt + 1}/{max_retries}")
            
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
            response.raise_for_status()
            
            result = response.json()['choices'][0]['message']['content']
            logger.debug(f"Row {row_idx}: API call successful after {attempt + 1} attempts")
            return result
            
        except requests.exceptions.Timeout as e:
            last_error = f"TIMEOUT_ERROR: Request timed out after 120s"
            logger.warning(f"Row {row_idx}: Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)  # Cap at 30s
                logger.info(f"Row {row_idx}: Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limiting
                last_error = f"RATE_LIMIT_ERROR: Too many requests (HTTP 429)"
                wait_time = min(2 ** attempt * 5, 60)  # Longer wait for rate limits, cap at 60s
                logger.warning(f"Row {row_idx}: Rate limited on attempt {attempt + 1}/{max_retries}, waiting {wait_time}s")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
            elif e.response.status_code == 400:
                last_error = f"BAD_REQUEST_ERROR: Invalid request (HTTP 400) - {e.response.text[:200]}"
                logger.error(f"Row {row_idx}: Bad request (likely won't succeed on retry): {last_error}")
                break  # Don't retry bad requests
            elif e.response.status_code == 401:
                last_error = f"AUTH_ERROR: Invalid API key (HTTP 401)"
                logger.error(f"Row {row_idx}: Authentication failed: {last_error}")
                break  # Don't retry auth errors
            elif e.response.status_code >= 500:
                last_error = f"SERVER_ERROR: Server error (HTTP {e.response.status_code})"
                logger.warning(f"Row {row_idx}: Server error on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            else:
                last_error = f"HTTP_ERROR: HTTP {e.response.status_code} - {e.response.text[:200]}"
                logger.error(f"Row {row_idx}: HTTP error: {last_error}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    
        except requests.exceptions.ConnectionError as e:
            last_error = f"CONNECTION_ERROR: Network connection failed"
            logger.warning(f"Row {row_idx}: Connection error on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                
        except json.JSONDecodeError as e:
            last_error = f"JSON_DECODE_ERROR: Invalid JSON response from API"
            logger.warning(f"Row {row_idx}: JSON decode error on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(1)
                
        except KeyError as e:
            last_error = f"RESPONSE_FORMAT_ERROR: Unexpected API response format - missing key: {e}"
            logger.warning(f"Row {row_idx}: Response format error on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(1)
                
        except Exception as e:
            last_error = f"UNKNOWN_ERROR: {str(e)}"
            logger.error(f"Row {row_idx}: Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    # All attempts failed
    final_error = f"FAILED_AFTER_{max_retries}_ATTEMPTS: {last_error}"
    logger.error(f"Row {row_idx}: {final_error}")
    return final_error

def validate_single_sample(args_tuple):
    """Validate a single sample - designed for parallel execution."""
    row_idx, row, total_rows = args_tuple
    
    try:
        task_category = TaskCategory(row['task_category'])
        
        # Skip if not a non-code task
        if task_category not in VALIDATION_PROMPTS:
            result = "SKIPPED: Not a non-code task"
            logger.debug(f"Row {row_idx}: {result}")
            return row_idx, result
        
        logger.info(f"Processing row {row_idx}/{total_rows}: {task_category}")
        
        system_prompt, user_prompt = create_validation_prompt(
            task_category, 
            row['input_data'], 
            row['output_data']
        )
        
        validation_result = call_llm_api_with_retry(
            system_prompt, 
            user_prompt, 
            row_idx, 
            max_retries=args.retry_attempts
        )
        
        logger.info(f"Row {row_idx}: Validation completed")
        return row_idx, validation_result
        
    except Exception as e:
        error_msg = f"PROCESSING_ERROR: {str(e)}"
        logger.error(f"Row {row_idx}: {error_msg}")
        return row_idx, error_msg

def save_checkpoint(df, checkpoint_path, completed_indices):
    """Save current progress to checkpoint file."""
    try:
        checkpoint_data = {
            'completed_indices': list(completed_indices),
            'timestamp': time.time(),
            'total_rows': len(df)
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
        logger.info(f"Checkpoint saved: {len(completed_indices)} completed")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_path):
    """Load progress from checkpoint file."""
    try:
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            completed_indices = set(checkpoint_data.get('completed_indices', []))
            logger.info(f"Loaded checkpoint: {len(completed_indices)} previously completed")
            return completed_indices
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
    return set()

def main():
    # Load the dataset
    logger.info(f"Loading dataset from {args.input_filepath}...")
    df = pd.read_csv(args.input_filepath)
    
    # Filter to only non-code tasks
    non_code_tasks = [
        TaskCategory.DOCSTRING_GENERATION,
        TaskCategory.CODE_SUMMARIZATION, 
        TaskCategory.CODE_EXPLANATION,
        TaskCategory.FUNCTION_NAMING,
        TaskCategory.VARIABLE_NAMING
    ]
    
    df_non_code = df[df['task_category'].isin([task.value for task in non_code_tasks])]
    logger.info(f"Found {len(df_non_code)} samples across non-code tasks")
    
    # Determine sampling strategy
    if args.validate_all:
        logger.info("VALIDATE_ALL mode: Processing all available samples")
        sampled_dfs = []
        for task in non_code_tasks:
            task_df = df_non_code[df_non_code['task_category'] == task.value]
            if len(task_df) > 0:
                # Apply max_samples_per_task limit if specified
                if args.max_samples_per_task and len(task_df) > args.max_samples_per_task:
                    task_df = task_df.sample(n=args.max_samples_per_task, random_state=42)
                    logger.info(f"Limited {task.value} to {args.max_samples_per_task} samples (from {len(df_non_code[df_non_code['task_category'] == task.value])})")
                else:
                    logger.info(f"Selected ALL {len(task_df)} samples for {task.value}")
                sampled_dfs.append(task_df)
    elif args.nsamples_per_task == -1:
        logger.info("ALL SAMPLES mode (nsamples_per_task=-1): Processing all available samples")
        sampled_dfs = []
        for task in non_code_tasks:
            task_df = df_non_code[df_non_code['task_category'] == task.value]
            if len(task_df) > 0:
                sampled_dfs.append(task_df)
                logger.info(f"Selected ALL {len(task_df)} samples for {task.value}")
    else:
        logger.info(f"SAMPLING mode: Taking {args.nsamples_per_task} samples per task")
        sampled_dfs = []
        for task in non_code_tasks:
            task_df = df_non_code[df_non_code['task_category'] == task.value]
            if len(task_df) > 0:
                sample_size = min(args.nsamples_per_task, len(task_df))
                sampled_task_df = task_df.sample(n=sample_size, random_state=42)
                sampled_dfs.append(sampled_task_df)
                logger.info(f"Selected {sample_size} samples for {task.value}")
    
    # Combine sampled data
    df_sampled = pd.concat(sampled_dfs, ignore_index=True) if sampled_dfs else pd.DataFrame()
    
    if df_sampled.empty:
        logger.error("No samples found for validation!")
        return
    
    # Log dataset statistics
    task_counts = df_sampled['task_category'].value_counts()
    logger.info(f"Final dataset composition:")
    for task, count in task_counts.items():
        original_count = len(df_non_code[df_non_code['task_category'] == task])
        percentage = (count / original_count) * 100
        logger.info(f"  {task}: {count}/{original_count} ({percentage:.1f}%)")
    
    total_samples = len(df_sampled)
    total_original = len(df_non_code)
    logger.info(f"Total samples to validate: {total_samples}/{total_original} ({total_samples/total_original*100:.1f}%)")
    
    # Estimate processing time and cost
    estimated_time_minutes = (total_samples / args.max_workers) * 0.5  # Rough estimate: 30s per sample
    # Estimate processing time and cost
    estimated_time_minutes = (total_samples / args.max_workers) * 0.5  # Rough estimate: 30s per sample
    logger.info(f"Estimated processing time: {estimated_time_minutes:.1f} minutes with {args.max_workers} workers")
    logger.info(f"Using LLM model: {args.llm_model} with {args.max_workers} parallel workers")
    
    # Setup checkpoint and resume functionality
    checkpoint_path = f"{args.output_filepath}.checkpoint.json"
    completed_indices = load_checkpoint(checkpoint_path)
    
    # Initialize results column if resuming or starting fresh
    if 'llm_validation_result' not in df_sampled.columns:
        df_sampled['llm_validation_result'] = None
    
    # Determine which rows need processing
    if args.resume_from > 0:
        logger.info(f"Resuming from row {args.resume_from}")
        remaining_indices = list(range(args.resume_from, len(df_sampled)))
    else:
        remaining_indices = [i for i in range(len(df_sampled)) if i not in completed_indices]
    
    logger.info(f"Rows remaining to process: {len(remaining_indices)}")
    
    if not remaining_indices:
        logger.info("All samples already validated!")
        df_sampled.to_csv(args.output_filepath, index=False)
        return
    
    # Prepare arguments for parallel processing
    validation_tasks = [
        (idx, df_sampled.iloc[idx], len(df_sampled)) 
        for idx in remaining_indices
    ]
    
    # Process validations in parallel
    completed_count = len(completed_indices)
    failed_count = 0
    
    logger.info("Starting parallel validation...")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(validate_single_sample, task): task[0] 
            for task in validation_tasks
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(validation_tasks), desc="Validating") as pbar:
            for future in as_completed(future_to_idx):
                try:
                    row_idx, result = future.result()
                    df_sampled.loc[row_idx, 'llm_validation_result'] = result
                    
                    completed_indices.add(row_idx)
                    completed_count += 1
                    
                    if "ERROR" in str(result) or "FAILED" in str(result):
                        failed_count += 1
                        logger.warning(f"Row {row_idx} failed: {result}")
                    
                    # Update progress
                    pbar.update(1)
                    pbar.set_postfix({
                        'completed': completed_count,
                        'failed': failed_count,
                        'success_rate': f"{((completed_count - failed_count) / completed_count * 100):.1f}%"
                    })
                    
                    # Save checkpoint periodically
                    if completed_count % args.checkpoint_every == 0:
                        save_checkpoint(df_sampled, checkpoint_path, completed_indices)
                        # Also save intermediate results
                        df_sampled.to_csv(f"{args.output_filepath}.temp", index=False)
                        
                except Exception as e:
                    row_idx = future_to_idx[future]
                    logger.error(f"Future failed for row {row_idx}: {e}")
                    failed_count += 1
    
    # Final save
    logger.info("Saving final results...")
    df_sampled.to_csv(args.output_filepath, index=False)
    
    # Final checkpoint
    save_checkpoint(df_sampled, checkpoint_path, completed_indices)
    
    # Cleanup temp files
    temp_file = f"{args.output_filepath}.temp"
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    # Summary
    success_count = completed_count - failed_count
    logger.info("=== VALIDATION COMPLETE ===")
    logger.info(f"Total processed: {completed_count}")
    logger.info(f"Successful: {success_count} ({success_count/completed_count*100:.1f}%)")
    logger.info(f"Failed: {failed_count} ({failed_count/completed_count*100:.1f}%)")
    logger.info(f"Results saved to: {args.output_filepath}")
    
    # Clean up checkpoint on successful completion
    if len(remaining_indices) == completed_count - len(load_checkpoint(checkpoint_path)):
        try:
            os.remove(checkpoint_path)
            logger.info("Checkpoint file cleaned up")
        except:
            pass

if __name__ == "__main__":
    main()