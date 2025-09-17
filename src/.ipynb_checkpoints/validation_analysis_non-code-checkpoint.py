import pandas as pd
import json
import re
import argparse
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="Comprehensive analysis of LLM validation results")
parser.add_argument("--filepath", type=str, required=True, help="Path to validation results CSV")
parser.add_argument("--save_plots", action='store_true', help="Save analysis plots")
parser.add_argument("--output_summary", type=str, help="Save detailed summary to file")
args = parser.parse_args()

def identify_failure_type(result_text):
    """Identify the type of failure from the result text."""
    if pd.isna(result_text):
        return "MISSING_RESULT"
    
    result_str = str(result_text)
    
    # Check for specific error patterns  
    error_patterns = {
        "FAILED_AFTER_ATTEMPTS": r"FAILED_AFTER_\d+_ATTEMPTS",
        "TIMEOUT_ERROR": r"TIMEOUT_ERROR",
        "RATE_LIMIT_ERROR": r"RATE_LIMIT_ERROR", 
        "CONNECTION_ERROR": r"CONNECTION_ERROR",
        "SERVER_ERROR": r"SERVER_ERROR",
        "AUTH_ERROR": r"AUTH_ERROR",
        "BAD_REQUEST_ERROR": r"BAD_REQUEST_ERROR",
        "JSON_DECODE_ERROR": r"JSON_DECODE_ERROR",
        "PROCESSING_ERROR": r"PROCESSING_ERROR",
        "PARSE_ERROR": r"PARSE_ERROR",
        "SKIPPED": r"SKIPPED"
    }
    
    for error_type, pattern in error_patterns.items():
        if re.search(pattern, result_str):
            return error_type
    
    if "ERROR" in result_str.upper():
        return "GENERIC_ERROR"
        
    return "SUCCESS"

def parse_json_validation_simple(validation_text):
    """Simple JSON parsing to check if validation worked."""
    if pd.isna(validation_text) or "ERROR" in str(validation_text):
        return None
    
    try:
        # Try to extract JSON
        start = validation_text.find('{')
        end = validation_text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = validation_text[start:end]
            result = json.loads(json_str)
            return result.get("Is_Answer_Correct")
    except:
        pass
    
    # Regex fallback for correctness
    text_lower = str(validation_text).lower()
    if "is_answer_correct: true" in text_lower or "\"is_answer_correct\": true" in text_lower:
        return True
    elif "is_answer_correct: false" in text_lower or "\"is_answer_correct\": false" in text_lower:
        return False
    
    return None

def analyze_validation_quality(df):
    """Analyze the quality of successful validations."""
    successful_df = df[df['failure_type'] == 'SUCCESS'].copy()
    
    if len(successful_df) == 0:
        return {}
    
    # Parse correctness from successful validations
    successful_df['parsed_correctness'] = successful_df['llm_validation_result'].apply(parse_json_validation_simple)
    
    quality_stats = {}
    
    # Overall correctness distribution
    correctness_counts = successful_df['parsed_correctness'].value_counts(dropna=False)
    quality_stats['overall_correctness'] = {
        'correct': correctness_counts.get(True, 0),
        'incorrect': correctness_counts.get(False, 0), 
        'unparseable': correctness_counts.get(None, 0) if pd.isna(None) in correctness_counts else len(successful_df[successful_df['parsed_correctness'].isna()])
    }
    
    # Correctness by task category
    quality_stats['correctness_by_task'] = {}
    for task in successful_df['task_category'].unique():
        task_df = successful_df[successful_df['task_category'] == task]
        task_correctness = task_df['parsed_correctness'].value_counts(dropna=False)
        quality_stats['correctness_by_task'][task] = {
            'total': len(task_df),
            'correct': task_correctness.get(True, 0),
            'incorrect': task_correctness.get(False, 0),
            'unparseable': len(task_df[task_df['parsed_correctness'].isna()])
        }
    
    return quality_stats

def generate_summary_report(df, quality_stats):
    """Generate a comprehensive summary report."""
    
    report = []
    report.append("=" * 60)
    report.append("COMPREHENSIVE LLM VALIDATION ANALYSIS")
    report.append("=" * 60)
    
    # Dataset overview
    total_samples = len(df)
    task_counts = df['task_category'].value_counts()
    
    report.append(f"\nDATASET OVERVIEW:")
    report.append(f"Total samples: {total_samples}")
    report.append(f"Task distribution:")
    for task, count in task_counts.items():
        report.append(f"  {task}: {count} ({count/total_samples*100:.1f}%)")
    
    # Failure analysis
    failure_counts = df['failure_type'].value_counts()
    successful_count = failure_counts.get('SUCCESS', 0)
    
    report.append(f"\nPROCESSING RESULTS:")
    report.append(f"Successfully processed: {successful_count}/{total_samples} ({successful_count/total_samples*100:.1f}%)")
    report.append(f"Failed processing: {total_samples - successful_count}/{total_samples} ({(total_samples - successful_count)/total_samples*100:.1f}%)")
    
    report.append(f"\nFAILURE BREAKDOWN:")
    for failure_type, count in failure_counts.items():
        if failure_type != 'SUCCESS':
            report.append(f"  {failure_type}: {count} ({count/total_samples*100:.1f}%)")
    
    # Failure patterns by task
    report.append(f"\nFAILURE PATTERNS BY TASK:")
    for task in df['task_category'].unique():
        task_df = df[df['task_category'] == task]
        task_failures = task_df[task_df['failure_type'] != 'SUCCESS']
        if len(task_failures) > 0:
            report.append(f"  {task}:")
            failure_breakdown = task_failures['failure_type'].value_counts()
            for failure_type, count in failure_breakdown.items():
                report.append(f"    {failure_type}: {count} ({count/len(task_df)*100:.1f}%)")
    
    # Quality analysis for successful validations
    if quality_stats:
        report.append(f"\nVALIDATION QUALITY ANALYSIS:")
        overall = quality_stats['overall_correctness']
        total_success = sum(overall.values())
        
        if total_success > 0:
            report.append(f"Among successfully processed samples:")
            report.append(f"  Correct outputs: {overall['correct']} ({overall['correct']/total_success*100:.1f}%)")
            report.append(f"  Incorrect outputs: {overall['incorrect']} ({overall['incorrect']/total_success*100:.1f}%)")
            report.append(f"  Unparseable responses: {overall['unparseable']} ({overall['unparseable']/total_success*100:.1f}%)")
        
        report.append(f"\nQUALITY BY TASK CATEGORY:")
        for task, stats in quality_stats['correctness_by_task'].items():
            if stats['total'] > 0:
                correct_pct = stats['correct'] / stats['total'] * 100
                report.append(f"  {task}:")
                report.append(f"    Accuracy: {stats['correct']}/{stats['total']} ({correct_pct:.1f}%)")
                report.append(f"    Incorrect: {stats['incorrect']} ({stats['incorrect']/stats['total']*100:.1f}%)")
                if stats['unparseable'] > 0:
                    report.append(f"    Unparseable: {stats['unparseable']} ({stats['unparseable']/stats['total']*100:.1f}%)")
    
    # Recommendations
    report.append(f"\nRECOMMENDATIONS:")
    
    # Check for retryable failures
    retryable_failures = ['TIMEOUT_ERROR', 'RATE_LIMIT_ERROR', 'CONNECTION_ERROR', 'FAILED_AFTER_ATTEMPTS']
    retryable_count = sum(failure_counts.get(failure_type, 0) for failure_type in retryable_failures)
    
    if retryable_count > 0:
        report.append(f"1. RETRY FAILURES: {retryable_count} samples can likely be retried successfully")
        report.append(f"   Use: python retry_failures.py --input_filepath your_file.csv")
    
    # Check for auth/config issues
    critical_failures = ['AUTH_ERROR', 'BAD_REQUEST_ERROR']
    critical_count = sum(failure_counts.get(failure_type, 0) for failure_type in critical_failures)
    
    if critical_count > 0:
        report.append(f"2. FIX CONFIGURATION: {critical_count} samples failed due to configuration issues")
        report.append(f"   Check API key, model availability, and request format")
    
    # Check parsing issues
    parse_issues = failure_counts.get('JSON_DECODE_ERROR', 0) + failure_counts.get('PARSE_ERROR', 0)
    if parse_issues > 0:
        report.append(f"3. IMPROVE PARSING: {parse_issues} samples had parsing issues")
        report.append(f"   Consider updating the parsing logic or prompt engineering")
    
    # Quality improvements
    if quality_stats and quality_stats['overall_correctness'].get('incorrect', 0) > 0:
        incorrect_pct = quality_stats['overall_correctness']['incorrect'] / sum(quality_stats['overall_correctness'].values()) * 100
        if incorrect_pct > 20:
            report.append(f"4. IMPROVE ACCURACY: {incorrect_pct:.1f}% of validations found incorrect outputs")
            report.append(f"   Consider reviewing prompt engineering or model selection")
    
    return "\n".join(report)

def create_visualizations(df, quality_stats):
    """Create analysis visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Processing success/failure pie chart
    failure_counts = df['failure_type'].value_counts()
    success_count = failure_counts.get('SUCCESS', 0)
    failure_count = len(df) - success_count
    
    axes[0, 0].pie([success_count, failure_count], 
                   labels=['Successful', 'Failed'], 
                   autopct='%1.1f%%',
                   colors=['green', 'red'])
    axes[0, 0].set_title('Overall Processing Results')
    
    # 2. Failure types breakdown
    failure_only = failure_counts[failure_counts.index != 'SUCCESS']
    if len(failure_only) > 0:
        axes[0, 1].bar(range(len(failure_only)), failure_only.values)
        axes[0, 1].set_xticks(range(len(failure_only)))
        axes[0, 1].set_xticklabels(failure_only.index, rotation=45, ha='right')
        axes[0, 1].set_title('Types of Failures')
        axes[0, 1].set_ylabel('Count')
    
    # 3. Success rate by task category
    task_success_rates = []
    tasks = []
    for task in df['task_category'].unique():
        task_df = df[df['task_category'] == task]
        success_rate = len(task_df[task_df['failure_type'] == 'SUCCESS']) / len(task_df)
        task_success_rates.append(success_rate * 100)
        tasks.append(task.replace('_', '\n'))  # Break long task names
    
    axes[1, 0].bar(range(len(tasks)), task_success_rates)
    axes[1, 0].set_xticks(range(len(tasks)))
    axes[1, 0].set_xticklabels(tasks, rotation=45, ha='right')
    axes[1, 0].set_title('Success Rate by Task Category')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].set_ylim(0, 100)
    
    # 4. Quality analysis for successful validations
    if quality_stats and quality_stats['overall_correctness']:
        overall = quality_stats['overall_correctness']
        labels = []
        sizes = []
        colors = ['green', 'orange', 'gray']
        
        if overall['correct'] > 0:
            labels.append(f"Correct ({overall['correct']})")
            sizes.append(overall['correct'])
        if overall['incorrect'] > 0:
            labels.append(f"Incorrect ({overall['incorrect']})")
            sizes.append(overall['incorrect'])
        if overall['unparseable'] > 0:
            labels.append(f"Unparseable ({overall['unparseable']})")
            sizes.append(overall['unparseable'])
        
        if sizes:
            axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(sizes)])
            axes[1, 1].set_title('Validation Quality\n(Among Successful Processes)')
        else:
            axes[1, 1].text(0.5, 0.5, 'No quality data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Validation Quality')
    
    plt.tight_layout()
    
    if args.save_plots:
        plt.savefig('validation_analysis.png', dpi=300, bbox_inches='tight')
        print("Plots saved to validation_analysis.png")
    
    plt.show()

def main():
    # Load validation results
    df = pd.read_csv(args.filepath)
    
    print("Loading and analyzing validation results...")
    
    if 'llm_validation_result' not in df.columns:
        print("ERROR: Input file must contain 'llm_validation_result' column!")
        return
    
    # Add failure type analysis
    df['failure_type'] = df['llm_validation_result'].apply(identify_failure_type)
    
    # Analyze validation quality for successful samples
    quality_stats = analyze_validation_quality(df)
    
    # Generate comprehensive report
    summary_report = generate_summary_report(df, quality_stats)
    
    # Print report
    print(summary_report)
    
    # Save report if requested
    if args.output_summary:
        with open(args.output_summary, 'w') as f:
            f.write(summary_report)
        print(f"\nDetailed summary saved to: {args.output_summary}")
    
    # Create visualizations
    try:
        create_visualizations(df, quality_stats)
    except Exception as e:
        print(f"Could not create visualizations: {e}")
    
    # Generate actionable insights
    print("\n" + "="*60)
    print("ACTIONABLE NEXT STEPS:")
    print("="*60)
    
    failure_counts = df['failure_type'].value_counts()
    
    # 1. Immediate actions for retryable failures
    retryable_failures = ['TIMEOUT_ERROR', 'RATE_LIMIT_ERROR', 'CONNECTION_ERROR', 'FAILED_AFTER_ATTEMPTS']
    retryable_count = sum(failure_counts.get(failure_type, 0) for failure_type in retryable_failures)
    
    if retryable_count > 0:
        print(f"\n1. RETRY COMMAND (for {retryable_count} retryable failures):")
        print(f"   python retry_failures.py \\")
        print(f"       --input_filepath {args.filepath} \\")
        print(f"       --output_filepath {args.filepath.replace('.csv', '_retried.csv')} \\")
        print(f"       --max_workers 3 \\")
        print(f"       --retry_attempts 5")
    
    # 2. Configuration fixes
    config_issues = failure_counts.get('AUTH_ERROR', 0) + failure_counts.get('BAD_REQUEST_ERROR', 0)
    if config_issues > 0:
        print(f"\n2. CONFIGURATION ISSUES ({config_issues} samples):")
        print(f"   - Check your OpenRouter API key")
        print(f"   - Verify model availability: {args.filepath}")
        print(f"   - Review request format and limits")
    
    # 3. Quality improvements
    if quality_stats:
        total_success = sum(quality_stats['overall_correctness'].values())
        incorrect_count = quality_stats['overall_correctness'].get('incorrect', 0)
        
        if total_success > 0 and incorrect_count / total_success > 0.3:
            print(f"\n3. HIGH INCORRECT RATE ({incorrect_count}/{total_success} = {incorrect_count/total_success*100:.1f}%):")
            print(f"   - Review prompt engineering")
            print(f"   - Consider different model or temperature")
            print(f"   - Check dataset quality")
    
    # 4. Parsing improvements
    parse_issues = failure_counts.get('JSON_DECODE_ERROR', 0) + quality_stats['overall_correctness'].get('unparseable', 0)
    if parse_issues > 0:
        print(f"\n4. PARSING ISSUES ({parse_issues} samples):")
        print(f"   - Improve JSON parsing robustness")
        print(f"   - Add more regex fallback patterns")
        print(f"   - Consider prompt engineering for better JSON compliance")
    
    print(f"\n5. RESUME CAPABILITY:")
    print(f"   If validation was interrupted, resume with:")
    print(f"   python llm_validation.py --resume_from <row_number> [other original args]")

if __name__ == "__main__":
    main()