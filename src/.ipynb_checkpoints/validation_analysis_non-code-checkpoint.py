import pandas as pd
import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description="Analyze LLM validation results")
parser.add_argument("--filepath", type=str, required=True, help="Path to validation results CSV")
args = parser.parse_args()

def parse_json_validation(validation_text):
    """Parse JSON validation result."""
    if pd.isna(validation_text) or "ERROR" in str(validation_text):
        return {"Is_Answer_Correct": None, "Corrected": "", "Reasoning": "ERROR"}
    
    try:
        # Try to extract JSON from the text (in case there's extra text)
        start = validation_text.find('{')
        end = validation_text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = validation_text[start:end]
            result = json.loads(json_str)
            return {
                "Is_Answer_Correct": result.get("Is_Answer_Correct"),
                "Corrected": result.get("Corrected", ""),
                "Reasoning": result.get("Reasoning", "")
            }
    except:
        pass
    
    return {"Is_Answer_Correct": None, "Corrected": "", "Reasoning": "PARSE_ERROR"}

def main():
    # Load validation results
    df = pd.read_csv(args.filepath)
    
    print("=== LLM Validation Results Analysis ===\n")
    print(f"Total samples analyzed: {len(df)}")
    
    # Parse JSON validations
    df['parsed_validation'] = df['llm_validation_result'].apply(parse_json_validation)
    df['is_correct'] = df['parsed_validation'].apply(lambda x: x['Is_Answer_Correct'])
    df['has_correction'] = df['parsed_validation'].apply(lambda x: len(x['Corrected'].strip()) > 0)
    df['reasoning'] = df['parsed_validation'].apply(lambda x: x['Reasoning'])
    
    # Summary by task
    print("\n--- Results by Task Category ---")
    
    for task in df['task_category'].unique():
        task_data = df[df['task_category'] == task]
        
        correct_count = sum(task_data['is_correct'] == True)
        incorrect_count = sum(task_data['is_correct'] == False)
        error_count = sum(task_data['is_correct'].isna())
        corrected_count = sum(task_data['has_correction'])
        
        print(f"\n{task.upper()}:")
        print(f"  Total Samples: {len(task_data)}")
        print(f"  Correct: {correct_count} ({correct_count/len(task_data)*100:.1f}%)")
        print(f"  Incorrect: {incorrect_count} ({incorrect_count/len(task_data)*100:.1f}%)")
        print(f"  Errors/Unparseable: {error_count}")
        print(f"  Had Corrections: {corrected_count}")
    
    # Overall statistics
    print(f"\n--- Overall Statistics ---")
    total_correct = sum(df['is_correct'] == True)
    total_incorrect = sum(df['is_correct'] == False)
    total_errors = sum(df['is_correct'].isna())
    
    print(f"Overall Accuracy: {total_correct}/{len(df)} ({total_correct/len(df)*100:.1f}%)")
    print(f"Incorrect Outputs: {total_incorrect} ({total_incorrect/len(df)*100:.1f}%)")
    print(f"Processing Errors: {total_errors} ({total_errors/len(df)*100:.1f}%)")
    
    # Show examples
    print(f"\n--- Sample Results ---")
    
    # Show some correct examples
    correct_samples = df[df['is_correct'] == True]
    if len(correct_samples) > 0:
        print(f"\nCORRECT Example:")
        example = correct_samples.iloc[0]
        print(f"Task: {example['task_category']}")
        print(f"Reasoning: {example['reasoning'][:200]}...")
        print("-" * 50)
    
    # Show some incorrect examples with corrections
    incorrect_samples = df[df['is_correct'] == False]
    if len(incorrect_samples) > 0:
        print(f"\nINCORRECT Example (with correction):")
        example = incorrect_samples.iloc[0]
        parsed = example['parsed_validation']
        print(f"Task: {example['task_category']}")
        print(f"Original Output: {str(example['output_data'])[:100]}...")
        print(f"Correction: {parsed['Corrected'][:100]}...")
        print(f"Reasoning: {parsed['Reasoning'][:200]}...")
        print("-" * 50)
    
    # Identify patterns in corrections
    print(f"\n--- Correction Analysis ---")
    corrections_by_task = df[df['has_correction']].groupby('task_category').size()
    if len(corrections_by_task) > 0:
        print("Tasks needing most corrections:")
        for task, count in corrections_by_task.sort_values(ascending=False).items():
            print(f"  {task}: {count} corrections")
    
    # Common reasoning patterns
    print(f"\n--- Common Issues Found ---")
    all_reasoning = df['reasoning'].dropna().str.lower()
    
    common_issues = {
        "accuracy": sum("accurat" in reason or "incorrect" in reason or "wrong" in reason for reason in all_reasoning),
        "clarity": sum("clear" in reason or "confus" in reason or "unclear" in reason for reason in all_reasoning), 
        "completeness": sum("complet" in reason or "missing" in reason or "lack" in reason for reason in all_reasoning),
        "conventions": sum("convention" in reason or "naming" in reason for reason in all_reasoning)
    }
    
    for issue, count in sorted(common_issues.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {issue.title()}: mentioned in {count} validations")

if __name__ == "__main__":
    main()