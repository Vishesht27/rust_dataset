import pandas as pd
import requests
import json
import argparse
from tqdm import tqdm
tqdm.pandas()

openrouter_key = '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'

LLM_MODEL_MAPPING = {
    "gemini": "google/gemini-2.5-pro",
    "claude": "anthropic/claude-sonnet-4",
    "gpt5": "openai/gpt-5",
}

parser = argparse.ArgumentParser(description="Error analysis on the validated Rust dataset.")
parser.add_argument("--input_filepath", type=str, help="Input file path", required=True)
parser.add_argument("--output_filepath", type=str, help="Output file path", required=True)
parser.add_argument("--nsamples_per_cat", type=int, default=15, help="Number of samples per category")
parser.add_argument("--llm_model", type=str, choices=LLM_MODEL_MAPPING.keys(), default="gemini", help="LLM model to use")
args = parser.parse_args()

SYSTEM_PROMPT = """
You are a Rust language expert. Your task is to validate and potentially modify Rust code output.

Input Format: You will receive four components:
- crate_name: The name of the Rust crate
- input_data: The original input/requirements 
- task_category: The category of the task specified in the input_data
- output_data: The generated code or response to validate

Your Responsibilities:

1. Check the output_data - Check if it meets the requirements based on the instruction/information given in the input_data and task_category, and check if the output is correct.

2. For tasks that require generating code as the output, ensure:
    -  Use appropriate Rust patterns (Result for errors, Option for nullable values)
    -  Include proper error handling where relevant
    -  Use efficient algorithms and data structures
    -  Ensure memory safety and avoid unnecessary allocations
    -  Do not include imports that is already defined in the code_context or test_context

3. Output Instructions:
    - If output is correct as-is, return the original output with the addition of the “dependencies” field.
    - If output is modified, return the modified output with the addition of the “dependencies” field.
    - The “dependencies” field is a List of only external crates (not the std library) needed in Cargo.toml. Please refer to the final output to deduct what are the additional required dependencies besides the ones already included in code_context or test_context. Format: Array of strings in Cargo.toml format. Examples: ["serde", "tokio", "tungstenite"]. Use an empty array [] if no external dependencies are needed.
    - Ensure the output is valid JSON format without any preamble or explanation.
"""

def user_prompt_creation(row):
    crate_name = row['crate_name']
    input_data = row['input_data']
    output_data = row['output_data']
    task_category = row['task_category']

    
    user_prompt = f"""
Crate Name: {crate_name}
Input Data: {input_data}
task_category: {task_category}
Output Data: {output_data}"""

    return user_prompt


def llm_calling(system_prompt,user_prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
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
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()['choices'][0]['message']['content']


def run_script(row):
    system_prompt = SYSTEM_PROMPT
    user_prompt = user_prompt_creation(row)
    answer = llm_calling(system_prompt,user_prompt)
    return answer


if __name__ == "__main__":
    df = pd.read_csv(args.input_filepath)
    print("Initial DataFrame:")
    print(df.info())
    df = df.groupby("task_category").apply(lambda x: x.sample(n=args.nsamples_per_cat, 
                                                              random_state=42))\
                                    .reset_index(drop=True)
    print("Sampled DataFrame:")
    print(df.info())
    print(f"Running scripts with {args.llm_model}...")
    df["output"] = df.progress_apply(run_script, axis=1)
    print("Finished running scripts.")
    df.to_csv(args.output_filepath, index=False)
