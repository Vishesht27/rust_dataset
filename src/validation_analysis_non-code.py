import pandas as pd
import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description="Analyze LLM validation results")
parser.add_argument("--filepath", type=str, required=True, help="Path to validation results CSV")
args = parser.parse_args()

def parse_json_validation(validation_text):
    """Parse JSON validation result with regex fallback."""
    if pd.isna(validation_text) or "ERROR" in str(validation_text):
        return {"Is_Answer_Correct": None, "Corrected": "", "Reasoning": "ERROR"}
    
    validation_text = str(validation_text)
    
    # Method 1: Try to extract and parse JSON
    try:
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
    
    # Method 2: Regex fallback patterns
    try:
        # Extract Is_Answer_Correct
        is_correct = None
        correct_patterns = [
            r'"Is_Answer_Correct"\s*:\s*(true|false)',
            r'"Is_Answer_Correct":\s*(true|false)', 
            r'Is_Answer_Correct["\s]*:\s*(true|false)',
            r'answer.*correct["\s]*:\s*(true|false)',
            r'correct["\s]*:\s*(true|false)'
        ]
        
        for pattern in correct_patterns:
            match = re.search(pattern, validation_text, re.IGNORECASE)
            if match:
                is_correct = match.group(1).lower() == 'true'
                break
        
        # Extract Corrected field
        corrected = ""
        corrected_patterns = [
            r'"Corrected"\s*:\s*"([^"]*)"',
            r'"Corrected":\s*"([^"]*)"',
            r'Corrected["\s]*:\s*"([^"]*)"',
            r'corrected["\s]*:\s*"([^"]*)"'
        ]
        
        for pattern in corrected_patterns:
            match = re.search(pattern, validation_text, re.IGNORECASE | re.DOTALL)
            if match:
                corrected = match.group(1)
                break
        
        # Extract Reasoning
        reasoning = ""
        reasoning_patterns = [
            r'"Reasoning"\s*:\s*"([^"]*)"',
            r'"Reasoning":\s*"([^"]*)"',
            r'Reasoning["\s]*:\s*"([^"]*)"',
            r'reasoning["\s]*:\s*"([^"]*)"'
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, validation_text, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1)
                break
        
        # If we found at least the correctness, return the parsed result
        if is_correct is not None:
            return {
                "Is_Answer_Correct": is_correct,
                "Corrected": corrected,
                "Reasoning": reasoning if reasoning else "Regex parsing - incomplete reasoning"
            }
            
    except Exception as e:
        pass
    
    # Method 3: Last resort - pattern matching for common phrases
    try:
        text_lower = validation_text.lower()
        
        # Determine correctness from common phrases
        is_correct = None
        if any(phrase in text_lower for phrase in ["is correct", "answer is true", "appropriate", "accurate"]):
            is_correct = True
        elif any(phrase in text_lower for phrase in ["is incorrect", "is wrong", "answer is false", "inappropriate", "inaccurate"]):
            is_correct = False
        
        # Look for correction indicators
        has_correction = any(phrase in text_lower for phrase in ["should be", "correct version", "better would be", "instead"])
        
        if is_correct is not None:
            return {
                "Is_Answer_Correct": is_correct,
                "Corrected": "See reasoning for details" if has_correction and not is_correct else "",
                "Reasoning": validation_text[:500] + "..." if len(validation_text) > 500 else validation_text
            }
    except:
        pass
    
    return {"Is_Answer_Correct": None, "Corrected": "", "Reasoning": "PARSE_ERROR"}

def extract_correctness_stats(validation_text):
    """Extract basic correctness even when JSON parsing fails completely."""
    if pd.isna(validation_text):
        return None
    
    text_lower = str(validation_text).lower()
    
    # Strong indicators of correctness/incorrectness
    correct_indicators = ["is correct", "answer is true", "is_answer_correct: true", "appropriate", "accurate"]
    incorrect_indicators = ["is incorrect", "is wrong", "answer is false", "is_answer_correct: false", "inappropriate", "inaccurate"]
    
    correct_score = sum(1 for phrase in correct_indicators if phrase in text_lower)
    incorrect_score = sum(1 for phrase in incorrect_indicators if phrase in text_lower)
    
    if correct_score > incorrect_score:
        return True
    elif incorrect_score > correct_score:
        return False
    else:
        return None

def main():
    # Load validation results
    df = pd.read_csv(args.filepath)
    
    print("=== LLM Validation Results Analysis ===\n")
    print(f"Total samples analyzed: {len(df)}")
    
    # Parse JSON validations with fallback
    df['parsed_validation'] = df['llm_validation_result'].apply(parse_json_validation)
    df['is_correct'] = df['parsed_validation'].apply(lambda x: x['Is_Answer_Correct'])
    df['has_correction'] = df['parsed_validation'].apply(lambda x: len(str(x['Corrected']).strip()) > 0)
    df['reasoning'] = df['parsed_validation'].apply(lambda x: x['Reasoning'])
    
    # Add fallback correctness extraction for failed parses
    df['fallback_correct'] = df['llm_validation_result'].apply(extract_correctness_stats)
    
    # Use fallback when main parsing failed
    df['final_correctness'] = df.apply(lambda row: 
        row['is_correct'] if row['is_correct'] is not None 
        else row['fallback_correct'], axis=1)
    
    # Summary by task
    print("\n--- Results by Task Category ---")
    
    for task in df['task_category'].unique():
        task_data = df[df['task_category'] == task]
        
        # Use final_correctness which includes fallback
        correct_count = sum(task_data['final_correctness'] == True)
        incorrect_count = sum(task_data['final_correctness'] == False) 
        error_count = sum(task_data['final_correctness'].isna())
        
        # JSON parsing stats
        json_parsed = sum(task_data['is_correct'].notna())
        regex_fallback = sum((task_data['is_correct'].isna()) & (task_data['fallback_correct'].notna()))
        
        corrected_count = sum(task_data['has_correction'])
        
        print(f"\n{task.upper()}:")
        print(f"  Total Samples: {len(task_data)}")
        print(f"  Correct: {correct_count} ({correct_count/len(task_data)*100:.1f}%)")
        print(f"  Incorrect: {incorrect_count} ({incorrect_count/len(task_data)*100:.1f}%)")
        print(f"  Unparseable: {error_count}")
        print(f"  JSON Parsed: {json_parsed}, Regex Fallback: {regex_fallback}")
        print(f"  Had Corrections: {corrected_count}")
    
    # Overall statistics
    print(f"\n--- Overall Statistics ---")
    total_correct = sum(df['final_correctness'] == True)
    total_incorrect = sum(df['final_correctness'] == False)
    total_errors = sum(df['final_correctness'].isna())
    
    total_json_parsed = sum(df['is_correct'].notna())
    total_regex_fallback = sum((df['is_correct'].isna()) & (df['fallback_correct'].notna()))
    
    print(f"Overall Accuracy: {total_correct}/{len(df)} ({total_correct/len(df)*100:.1f}%)")
    print(f"Incorrect Outputs: {total_incorrect} ({total_incorrect/len(df)*100:.1f}%)")
    print(f"Completely Unparseable: {total_errors} ({total_errors/len(df)*100:.1f}%)")
    print(f"Parsing Success: JSON={total_json_parsed}, Regex Fallback={total_regex_fallback}")
    
    # Show examples
    print(f"\n--- Sample Results ---")
    
    # Show some correct examples
    correct_samples = df[df['final_correctness'] == True]
    if len(correct_samples) > 0:
        print(f"\nCORRECT Example:")
        example = correct_samples.iloc[0]
        print(f"Task: {example['task_category']}")
        print(f"Reasoning: {str(example['reasoning'])[:200]}...")
        print("-" * 50)
    
    # Show some incorrect examples with corrections
    incorrect_samples = df[df['final_correctness'] == False]
    if len(incorrect_samples) > 0:
        print(f"\nINCORRECT Example (with correction):")
        example = incorrect_samples.iloc[0]
        parsed = example['parsed_validation']
        print(f"Task: {example['task_category']}")
        print(f"Original Output: {str(example['output_data'])[:100]}...")
        print(f"Correction: {str(parsed['Corrected'])[:100]}...")
        print(f"Reasoning: {str(parsed['Reasoning'])[:200]}...")
        print("-" * 50)
    
    # Show regex fallback examples
    regex_samples = df[(df['is_correct'].isna()) & (df['fallback_correct'].notna())]
    if len(regex_samples) > 0:
        print(f"\nREGEX FALLBACK Example:")
        example = regex_samples.iloc[0]
        print(f"Task: {example['task_category']}")
        print(f"Final Correctness: {example['final_correctness']}")
        print(f"Raw Response: {str(example['llm_validation_result'])[:200]}...")
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