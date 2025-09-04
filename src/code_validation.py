import pandas as pd
import subprocess
import tempfile
import os
import re
import shutil
from ast import literal_eval
from task_category import TaskCategory
import argparse

parser = argparse.ArgumentParser(description="Validate some Rust code.")
parser.add_argument("--filepath", type=str, default="rust_dataset.csv", help="Input file path")
parser.add_argument("--task_column", type=str, default="task_category", help="Task column")
parser.add_argument("--input_column", type=str, default="input_data", help="Input column")
parser.add_argument("--output_column", type=str, default="output_data", help="Output column")
parser.add_argument("--output_file", type=str, help="Output file path",required=True)
args = parser.parse_args()


DOCKER_IMAGE = "rustlang/rust:nightly"

CODE_KEY_CATEGORY_MAP = {
    TaskCategory.CODE_GENERATION: "code", 
    TaskCategory.CODE_COMPLETION: "completion",
    TaskCategory.TEST_GENERATION: "test_cases",
    TaskCategory.DOCSTRING_GENERATION: "",
    TaskCategory.CODE_SUMMARIZATION: "",
    TaskCategory.BUG_DETECTION: "fixed_code",
    TaskCategory.CODE_REFACTORING: "code_after", 
    TaskCategory.FUNCTION_NAMING: "",
    TaskCategory.VARIABLE_NAMING: "",
    TaskCategory.API_USAGE_PREDICTION: "next_api_call",
    TaskCategory.CODE_SEARCH: "code_snippet",
    TaskCategory.CODE_EXPLANATION: "",
    TaskCategory.CODE_REVIEW: "code_after",
    TaskCategory.CODE_OPTIMIZATION: "code_after",
    TaskCategory.COMMENT_GENERATION: "commented_code",
}


def extract_crates(code: str) -> dict:
    # Very naive crate extraction (expand later if needed)
    crates = {}
    for line in code.splitlines():
        m = re.match(r"use\s+([a-zA-Z0-9_]+)", line.strip())
        if m:
            crate = m.group(1)
            if crate not in crates:
                crates[crate] = "*"  # default version, can be refined later
    return crates


def generate_cargo_toml(proj_dir: str, crates: dict, force_replace_to_hyphen: bool = True) -> None:
    cargo_toml = os.path.join(proj_dir, "Cargo.toml")
    with open(cargo_toml, "a") as f:
        # f.write("\n[dependencies]\n")
        for crate in crates:
            if force_replace_to_hyphen:
                crate = crate.replace("_", "-") # Crate names use hyphens
            if crate in ["std", "core", "alloc", "crate"]:
                continue
            elif crate == "tokio":
                f.write('tokio = { version = "1", features = ["full"] }\n')
            else:
                # Let Cargo resolve latest version
                f.write(f'{crate} = "*"\n')


def run_cargo_fetch(docker_proj:str, cargo_registry: str, cargo_git: str):
    fetch_proc = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{docker_proj}:/proj",
                    "-v", f"{cargo_registry}:/usr/local/cargo/registry",
                    "-v", f"{cargo_git}:/usr/local/cargo/git",
                    "-w", "/proj",
                    DOCKER_IMAGE,
                    "cargo", "fetch"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120
            )
    return fetch_proc


def run_cargo_check(docker_proj:str, cargo_registry: str, cargo_git: str):
    check_proc = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{docker_proj}:/proj",
                    "-v", f"{cargo_registry}:/usr/local/cargo/registry",
                    "-v", f"{cargo_git}:/usr/local/cargo/git",
                    "-w", "/proj",
                    "--network", "none",
                    DOCKER_IMAGE,
                    "cargo", "check", "--quiet"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120
            )
    return check_proc


def run_cargo_run(docker_proj:str, cargo_registry: str, cargo_git: str):
    run_proc = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{docker_proj}:/proj",
                    "-v", f"{cargo_registry}:/usr/local/cargo/registry",
                    "-v", f"{cargo_git}:/usr/local/cargo/git",
                    "-w", "/proj",
                    "--network", "none",
                    "--memory=256m", "--cpus=1",
                    DOCKER_IMAGE,
                    "cargo", "run", "--quiet"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120
            )
    return run_proc

def check_with_docker(code: str, crates: dict | None = None, 
                      force_replace_crates_to_hyphen: bool = True) -> tuple[bool, bool]:
    """
    Check if Rust code compiles and executes inside a Docker sandbox.
    Returns (compiled, executable, stdout/stderr).
    """
    if not crates:
        crates = extract_crates(code)

    print(f"Extracted crates: {crates}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = os.path.join(tmpdir, "testproj")

            # Create a new Cargo project
            subprocess.run(
                ["cargo", "new", "testproj", "--bin", "--quiet"],
                cwd=tmpdir,
                check=True
            )

            # Overwrite main.rs
            src_file = os.path.join(proj_dir, "src", "main.rs")
            with open(src_file, "w") as f:
                f.write(code)

            # Append dependencies to Cargo.toml
            generate_cargo_toml(proj_dir,crates,
                                force_replace_to_hyphen=force_replace_crates_to_hyphen)

            # Copy to docker context
            docker_proj = os.path.join(tmpdir, "docker_proj")
            shutil.copytree(proj_dir, docker_proj)
            cargo_registry = os.path.expanduser("~/.cargo/registry")
            cargo_git = os.path.expanduser("~/.cargo/git")

            # STEP 1: Resolve dependencies (needs network!)
            fetch_proc = run_cargo_fetch(docker_proj,cargo_registry,cargo_git)
            if fetch_proc.returncode != 0:
                print(f"Dependency fetch failed: {fetch_proc.stderr}")
                return False, False, fetch_proc.stderr
            else:
                print(f"Dependency fetch succeeded: {fetch_proc.stdout}")

            # STEP 2: Check without network
            check_proc = run_cargo_check(docker_proj,cargo_registry,cargo_git)
            if check_proc.returncode != 0:
                print(f"Check failed: {check_proc.stderr}")
                return False, False, check_proc.stderr
            else:
                print(f"Check succeeded: {check_proc.stdout}")

            # STEP 3: Run without network, with limits
            run_proc = run_cargo_run(docker_proj,cargo_registry,cargo_git)
            if run_proc.returncode != 0:
                print(f"Run failed: {run_proc.stderr}")
                return True, False, run_proc.stderr
            else:
                print(f"Run succeeded: {run_proc.stdout}")
            return True, True, run_proc.stdout

    except Exception as e:
        print(f"Error during Docker execution: {e}")
        return False, False, str(e)


def build_rust_main(code: str, code_context: str) -> str:
    if "fn main" in code:
        return f"{code_context}\n{code}"
    else:
        return f"""{code_context}
fn main() {{
    {code}
}}"""


def main(csv_file, input_column, output_column, task_column, output_file="result.csv"):
    df = pd.read_csv(csv_file)

    compiled_results = []
    executable_results = []
    stdout_results = []

    num_rows = len(df)
    print(f"Processing {num_rows} rows from {csv_file}...")

    for idx in range(num_rows):
        try:
            print(f"Checking row {idx} inside Docker sandbox...")
            input_data = df.loc[idx, input_column]
            code_context = literal_eval(input_data).get("code_context", "")
            output = df.loc[idx, output_column]
            task = df.loc[idx, task_column]

            print(f"Task: {task}")
            if CODE_KEY_CATEGORY_MAP[task] == "":
                print(f"Skipping row {idx} for task {task} as it has no code output to validate.")
                compiled_results.append(None)
                executable_results.append(None)
                stdout_results.append(f"Skipping row {idx} for task {task} as it has no code output to validate.")
                continue
            elif task == TaskCategory.TEST_GENERATION:
                # For test generation, output might be a list of test cases
                output_code = "\n".join(literal_eval(output).get(CODE_KEY_CATEGORY_MAP[task], []))
            else:
                output_code = literal_eval(output).get(CODE_KEY_CATEGORY_MAP[task], "")

            if not output_code.strip():
                compiled_results.append(False)
                executable_results.append(False)
                stdout_results.append("Empty code output.")
                continue

            if task == TaskCategory.CODE_COMPLETION:
                prefix = literal_eval(input_data).get("prefix", "")
                suffix = literal_eval(input_data).get("suffix", "")
                code = f"{prefix}\n{output_code}\n{suffix}"
            elif task == TaskCategory.TEST_GENERATION:
                code_to_test = literal_eval(input_data).get("code_to_test", "")
                test_context = literal_eval(input_data).get("test_context", "")
                code = build_rust_main(output_code,f"{code_context}\n{code_to_test}\n\n{test_context}")
            elif task == TaskCategory.API_USAGE_PREDICTION:
                current_code = literal_eval(input_data).get("code", "")
                code = build_rust_main(output_code,f"{code_context}\n{current_code}")
            else:
                code = build_rust_main(output_code,code_context)

            compiled, executable, stdout = check_with_docker(code)

            # print(f"Code:\n{code}\n{'-'*40}")
            print(f"Row {idx}: Compiled={compiled}, Executable={executable}")
            print('-'*80)

            compiled_results.append(compiled)
            executable_results.append(executable)
            stdout_results.append(stdout)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            compiled_results.append(False)
            executable_results.append(False)
            stdout_results.append(str(e))

    df["compiled"] = compiled_results
    df["executable"] = executable_results
    df["stdout"] = stdout_results
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def check_alexey_data(csv_file, code_column, crates_column, output_file="result_alexey.csv"):
    df = pd.read_csv(csv_file)

    compiled_results = []
    executable_results = []
    stdout_results = []

    num_rows = len(df)
    print(f"Processing {num_rows} rows from {csv_file}...")

    for idx in range(num_rows):
        try:
            print(f"Checking row {idx} inside Docker sandbox...")
            code = df.loc[idx, code_column]
            crates_list = literal_eval(df.loc[idx, crates_column])
            crates = {crate: "*" for crate in crates_list}
     
            compiled, executable, stdout = check_with_docker(code, crates,
                                                             force_replace_crates_to_hyphen=False)

            # print(f"Code:\n{code}\n{'-'*40}")
            print(f"Row {idx}: Compiled={compiled}, Executable={executable}")
            print('-'*80)

            compiled_results.append(compiled)
            executable_results.append(executable)
            stdout_results.append(stdout)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            compiled_results.append(False)
            executable_results.append(False)
            stdout_results.append(str(e))

    df["compiled"] = compiled_results
    df["executable"] = executable_results
    df["stdout"] = stdout_results
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # main(args.filepath,
    #      task_column=args.task_column,
    #      input_column=args.input_column,
    #      output_column=args.output_column,
    #      output_file=args.output_file
    #      )

    check_alexey_data(args.filepath, 
                      code_column="checked_code", 
                      crates_column="dependencies_to_add",
                      output_file=args.output_file,
                      )
