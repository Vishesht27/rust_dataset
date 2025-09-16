import pandas as pd
import requests
import json
import argparse
from tqdm import tqdm
import random
from ast import literal_eval
from task_category import TaskCategory

tqdm.pandas()

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
args = parser.parse_args()

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

def create_validation_prompt(task_category, input_data, output_data):
    """Create a validation prompt for a specific task."""
    
    system_prompt = VALIDATION_PROMPTS[task_category]
    
    if task_category == TaskCategory.DOCSTRING_GENERATION:
        input_dict = literal_eval(input_data) if isinstance(input_data, str) else input_data
        output_dict = literal_eval(output_data) if isinstance(output_data, str) else output_data
        
        user_prompt = f"""
**Rust Code:**
```rust
{input_dict.get('code', '')}
```

**Code Context (if any):**
```rust
{input_dict.get('code_context', 'None provided')}
```

**Generated Docstring:**
{output_dict.get('docstring', '')}

**Task:** Evaluate if this docstring is correct for the given Rust code.
"""

    elif task_category == TaskCategory.CODE_SUMMARIZATION:
        input_dict = literal_eval(input_data) if isinstance(input_data, str) else input_data
        output_dict = literal_eval(output_data) if isinstance(output_data, str) else output_data
        
        user_prompt = f"""
**Rust Code:**
```rust
{input_dict.get('code', '')}
```

**Code Context (if any):**
```rust
{input_dict.get('code_context', 'None provided')}
```

**Generated Summary:**
{output_dict.get('summary', '')}

**Task:** Evaluate if this summary is correct for the given Rust code.
"""

    elif task_category == TaskCategory.CODE_EXPLANATION:
        input_dict = literal_eval(input_data) if isinstance(input_data, str) else input_data
        output_dict = literal_eval(output_data) if isinstance(output_data, str) else output_data
        
        user_prompt = f"""
**Rust Code:**
```rust
{input_dict.get('code', '')}
```

**Code Context (if any):**
```rust
{input_dict.get('code_context', 'None provided')}
```

**Generated Explanation:**
{output_dict.get('explanation', '')}

**Task:** Evaluate if this explanation is correct for the given Rust code.
"""

    elif task_category == TaskCategory.FUNCTION_NAMING:
        input_dict = literal_eval(input_data) if isinstance(input_data, str) else input_data
        output_dict = literal_eval(output_data) if isinstance(output_data, str) else output_data
        
        user_prompt = f"""
**Function Implementation:**
```rust
{input_dict.get('code', '')}
```

**Code Context (if any):**
```rust
{input_dict.get('code_context', 'None provided')}
```

**Suggested Function Name:**
{output_dict.get('function_name', '')}

**Task:** Evaluate if this function name is appropriate for the given implementation.
"""

    elif task_category == TaskCategory.VARIABLE_NAMING:
        input_dict = literal_eval(input_data) if isinstance(input_data, str) else input_data
        output_dict = literal_eval(output_data) if isinstance(output_data, str) else output_data
        
        user_prompt = f"""
**Code with Placeholder:**
```rust
{input_dict.get('code', '')}
```

**Code Context (if any):**
```rust
{input_dict.get('code_context', 'None provided')}
```

**Suggested Variable Name:**
{output_dict.get('variable_name', '')}

**Task:** Evaluate if this variable name is appropriate for the given code context.
"""
    
    return system_prompt, user_prompt

def call_llm_api(system_prompt, user_prompt):
    """Make API call to LLM for validation."""
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
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"API call failed: {e}")
        return f"ERROR: {str(e)}"

def validate_sample(row):
    """Validate a single sample using LLM."""
    task_category = TaskCategory(row['task_category'])
    
    # Skip if not a non-code task
    if task_category not in VALIDATION_PROMPTS:
        return "SKIPPED: Not a non-code task"
    
    try:
        system_prompt, user_prompt = create_validation_prompt(
            task_category, 
            row['input_data'], 
            row['output_data']
        )
        
        validation_result = call_llm_api(system_prompt, user_prompt)
        return validation_result
        
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    # Load the dataset
    print(f"Loading dataset from {args.input_filepath}...")
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
    print(f"Found {len(df_non_code)} samples across non-code tasks")
    
    # Sample N examples per task
    sampled_dfs = []
    for task in non_code_tasks:
        task_df = df_non_code[df_non_code['task_category'] == task.value]
        if len(task_df) > 0:
            sample_size = min(args.nsamples_per_task, len(task_df))
            sampled_task_df = task_df.sample(n=sample_size, random_state=42)
            sampled_dfs.append(sampled_task_df)
            print(f"Selected {sample_size} samples for {task.value}")
    
    # Combine sampled data
    df_sampled = pd.concat(sampled_dfs, ignore_index=True) if sampled_dfs else pd.DataFrame()
    
    if df_sampled.empty:
        print("No samples found for validation!")
        return
    
    print(f"Total samples to validate: {len(df_sampled)}")
    print(f"Using LLM model: {args.llm_model}")
    
    # Perform validation
    print("Running LLM validation...")
    df_sampled['llm_validation_result'] = df_sampled.progress_apply(validate_sample, axis=1)
    
    # Save results
    df_sampled.to_csv(args.output_filepath, index=False)
    print(f"Validation results saved to {args.output_filepath}")
    
    # Print summary
    validation_counts = df_sampled.groupby('task_category')['llm_validation_result'].apply(
        lambda x: x.str.contains('ERROR').sum()
    )
    print("\nValidation Summary:")
    print(f"Total samples validated: {len(df_sampled)}")
    print("Errors by task category:")
    print(validation_counts)

if __name__ == "__main__":
    main()