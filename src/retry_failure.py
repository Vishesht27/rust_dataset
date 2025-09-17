import pandas as pd
import argparse
import re
import logging
from llm_validation import (
    validate_single_sample, LLM_MODEL_MAPPING, OPENROUTER_KEY, 
    save_checkpoint, load_checkpoint
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import time

# Logging will be configured after argument parsing
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="[DEPRECATED] Retry failed LLM validations - Use 'llm_validation_non_code.py --retry_failed' instead")
parser.add_argument("--input_filepath", type=str, required=True, help="CSV file with validation results")
parser.add_argument("--output_filepath", type=str, required=True, help="Output CSV file") 
parser.add_argument("--llm_model", type=str, choices=LLM_MODEL_MAPPING.keys(), default="claude", help="LLM model to use")
parser.add_argument("--max_workers", type=int, default=3, help="Number of parallel API calls (lower for retries)")
parser.add_argument("--retry_attempts", type=int, default=5, help="Number of retry attempts for each failed sample")
parser.add_argument("--failure_types", type=str, nargs='+', 
                   default=["FAILED_AFTER", "TIMEOUT_ERROR", "RATE_LIMIT_ERROR", "CONNECTION_ERROR", "PROCESSING_ERROR"],
                   help="Types of failures to retry")
parser.add_argument("--dry_run", action='store_true', help="Show what would be retried without actually doing it")
parser.add_argument("--log_filename", type=str, default="retry_failures.log", help="Custom log filename for this run")

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

def identify_failure_type(result_text):
    """Identify the type of failure from the result text."""
    if pd.isna(result_text):
        return "MISSING_RESULT"
    
    result_str = str(result_text)
    
    # Check for specific error patterns
    error_patterns = {
        "FAILED_AFTER": r"FAILED_AFTER_\d+_ATTEMPTS",
        "TIMEOUT_ERROR": r"TIMEOUT_ERROR",
        "RATE_LIMIT_ERROR": r"RATE_LIMIT_ERROR", 
        "CONNECTION_ERROR": r"CONNECTION_ERROR",
        "SERVER_ERROR": r"SERVER_ERROR",
        "AUTH_ERROR": r"AUTH_ERROR",
        "BAD_REQUEST_ERROR": r"BAD_REQUEST_ERROR",
        "JSON_DECODE_ERROR": r"JSON_DECODE_ERROR",
        "PROCESSING_ERROR": r"PROCESSING_ERROR",
        "UNKNOWN_ERROR": r"UNKNOWN_ERROR"
    }
    
    for error_type, pattern in error_patterns.items():
        if re.search(pattern, result_str):
            return error_type
    
    # Check for partial failures or malformed responses
    if "ERROR" in result_str.upper():
        return "GENERIC_ERROR"
        
    return "SUCCESS"

def analyze_failures(df):
    """Analyze the types and patterns of failures in the dataset."""
    logger.info("=== FAILURE ANALYSIS ===")
    
    # Identify failure types
    df['failure_type'] = df['llm_validation_result'].apply(identify_failure_type)
    
    # Overall statistics
    total_samples = len(df)
    failed_samples = df[df['failure_type'] != 'SUCCESS']
    success_count = total_samples - len(failed_samples)
    
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Successful: {success_count} ({success_count/total_samples*100:.1f}%)")
    logger.info(f"Failed: {len(failed_samples)} ({len(failed_samples)/total_samples*100:.1f}%)")
    
    # Failure breakdown by type
    failure_counts = df['failure_type'].value_counts()
    logger.info(f"\nFailure breakdown:")
    for failure_type, count in failure_counts.items():
        if failure_type != 'SUCCESS':
            logger.info(f"  {failure_type}: {count} ({count/total_samples*100:.1f}%)")
    
    # Failure breakdown by task category
    logger.info(f"\nFailures by task category:")
    task_failure_analysis = df[df['failure_type'] != 'SUCCESS'].groupby('task_category')['failure_type'].value_counts()
    for (task, failure_type), count in task_failure_analysis.items():
        logger.info(f"  {task} - {failure_type}: {count}")
    
    return failed_samples

def should_retry_failure(failure_type, retry_types):
    """Determine if a failure type should be retried."""
    return any(retry_pattern in failure_type for retry_pattern in retry_types)

def retry_failed_validations(failed_df, retry_types, max_workers=3):
    """Retry failed validations in parallel."""
    
    # Filter to only retryable failures
    retryable_failures = failed_df[failed_df['failure_type'].apply(
        lambda x: should_retry_failure(x, retry_types)
    )]
    
    if len(retryable_failures) == 0:
        logger.warning("No retryable failures found!")
        return failed_df
    
    logger.info(f"Retrying {len(retryable_failures)} failed validations...")
    
    # Prepare tasks for retry
    retry_tasks = [
        (idx, row, len(retryable_failures)) 
        for idx, row in retryable_failures.iterrows()
    ]
    
    # Track results
    retry_results = {}
    
    # Process retries in parallel (with fewer workers to be gentler)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(validate_single_sample, task): task[0] 
            for task in retry_tasks
        }
        
        with tqdm(total=len(retry_tasks), desc="Retrying failed validations") as pbar:
            for future in as_completed(future_to_idx):
                try:
                    row_idx, result = future.result()
                    retry_results[row_idx] = result
                    
                    # Check if retry was successful
                    new_failure_type = identify_failure_type(result)
                    if new_failure_type == 'SUCCESS':
                        logger.info(f"Row {row_idx}: Retry successful!")
                    else:
                        logger.warning(f"Row {row_idx}: Retry failed again - {new_failure_type}")
                    
                    pbar.update(1)
                    
                except Exception as e:
                    row_idx = future_to_idx[future]
                    logger.error(f"Retry future failed for row {row_idx}: {e}")
                    retry_results[row_idx] = f"RETRY_FUTURE_ERROR: {str(e)}"
    
    return retry_results

def main():
    # DEPRECATION WARNING
    logger.warning("=" * 80)
    logger.warning("DEPRECATION WARNING: retry_failure.py is deprecated!")
    logger.warning("Use 'llm_validation_non_code.py --retry_failed' instead for integrated retry functionality.")
    logger.warning("This script will be removed in a future version.")
    logger.warning("=" * 80)
    
    # Load the validation results
    logger.info(f"Loading validation results from {args.input_filepath}")
    df = pd.read_csv(args.input_filepath)
    
    if 'llm_validation_result' not in df.columns:
        logger.error("Input file must contain 'llm_validation_result' column!")
        return
    
    # Analyze failures
    failed_samples = analyze_failures(df)
    
    if len(failed_samples) == 0:
        logger.info("No failures found! All validations were successful.")
        # Just copy the input to output
        df.to_csv(args.output_filepath, index=False)
        return
    
    # Show what would be retried
    retryable_count = sum(
        should_retry_failure(failure_type, args.failure_types) 
        for failure_type in failed_samples['failure_type']
    )
    
    logger.info(f"\nRetry configuration:")
    logger.info(f"  Failure types to retry: {args.failure_types}")
    logger.info(f"  Retryable failures: {retryable_count}/{len(failed_samples)}")
    logger.info(f"  Max workers: {args.max_workers}")
    logger.info(f"  Retry attempts per sample: {args.retry_attempts}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - showing what would be retried:")
        retryable = failed_samples[failed_samples['failure_type'].apply(
            lambda x: should_retry_failure(x, args.failure_types)
        )]
        for idx, row in retryable.iterrows():
            logger.info(f"  Row {idx}: {row['task_category']} - {row['failure_type']}")
        return
    
    if retryable_count == 0:
        logger.warning("No retryable failures found with current failure types!")
        df.to_csv(args.output_filepath, index=False)
        return
    
    # Perform retries
    logger.info(f"\nStarting retry process...")
    retry_results = retry_failed_validations(failed_samples, args.failure_types, args.max_workers)
    
    # Update the dataframe with retry results
    df_updated = df.copy()
    for idx, new_result in retry_results.items():
        df_updated.loc[idx, 'llm_validation_result'] = new_result
    
    # Save updated results
    df_updated.to_csv(args.output_filepath, index=False)
    logger.info(f"Updated results saved to {args.output_filepath}")
    
    # Final analysis
    logger.info("\n=== RETRY RESULTS ===")
    original_failures = len(failed_samples)
    new_failures = sum(identify_failure_type(result) != 'SUCCESS' for result in retry_results.values())
    successful_retries = len(retry_results) - new_failures
    
    logger.info(f"Original failures: {original_failures}")
    logger.info(f"Samples retried: {len(retry_results)}")
    logger.info(f"Successful retries: {successful_retries}")
    logger.info(f"Still failing: {new_failures}")
    logger.info(f"Overall improvement: {successful_retries}/{original_failures} ({successful_retries/original_failures*100:.1f}%)")

if __name__ == "__main__":
    main()