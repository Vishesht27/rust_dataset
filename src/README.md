# Rust Dataset SFT Pipeline - Scripts Usage Guide

This guide covers all the scripts in the pipeline and their various usage patterns for processing and validating the Rust dataset.

## Table of Contents
- [Overview](#overview)
- [Scripts Summary](#scripts-summary)
- [LLM Validation Scripts](#llm-validation-scripts)
- [Analysis Scripts](#analysis-scripts)
- [Code Validation Scripts](#code-validation-scripts)
- [Common Usage Patterns](#common-usage-patterns)
- [Logging and Debugging](#logging-and-debugging)

## Overview

The pipeline consists of several scripts for different stages of data processing:
1. **LLM Validation**: Validate non-code tasks using LLM models
2. **Code Validation**: Compile and test Rust code samples
3. **Analysis**: Analyze results and generate reports
4. **Error Analysis**: Deep dive into specific errors and failures

## Scripts Summary

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `llm_validation_non_code.py` | LLM validation for non-code tasks with built-in retry | CSV dataset | Validated CSV with results |
| `code_validation.py` | Compile/test Rust code | CSV with code | CSV with compilation results |
| `eval_data.py` | Generate refinement of the hold-out validation dataset | CSV dataset | Enhanced CSV |
| `validation_analysis_non-code.py` | Analyze validation results for non-code related samples | Validation results CSV | Analysis report |
| `error_analysis.py` | Analyze compilation errors for code-related samples | Code validation results | Error statistics |

**Note**: `retry_failure.py` functionality has been integrated into `llm_validation_non_code.py` - use `--retry_failed` flag instead.

---

## LLM Validation Scripts

### 1. Primary LLM Validation (`llm_validation_non_code.py`)

Validates non-code tasks (docstring generation, code explanation, etc.) using LLM models.

#### Basic Usage
```bash
python llm_validation_non_code.py \
  --input_filepath data/rust_dataset_100k.csv \
  --output_filepath data/validated_results.csv
```

#### Advanced Options

**Different Sampling Strategies:**
```bash
# Sample 10 items per task (default: 3)
python llm_validation_non_code.py \
  --input_filepath data/rust_dataset_100k.csv \
  --output_filepath data/sample_10_results.csv \
  --nsamples_per_task 10

# Validate ALL samples
python llm_validation_non_code.py \
  --input_filepath data/rust_dataset_100k.csv \
  --output_filepath data/all_validated.csv \
  --validate_all

# Validate all but limit to max 1000 per task
python llm_validation_non_code.py \
  --input_filepath data/rust_dataset_100k.csv \
  --output_filepath data/limited_all.csv \
  --validate_all \
  --max_samples_per_task 1000
```

**Different LLM Models:**
```bash
# Use Claude (default)
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/claude_results.csv \
  --llm_model claude

# Use Gemini
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/gemini_results.csv \
  --llm_model gemini

# Use GPT-4
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/gpt4_results.csv \
  --llm_model gpt4
```

**Performance Tuning:**
```bash
# High throughput (more workers)
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/output.csv \
  --max_workers 10

# Conservative (fewer API calls, more reliable)
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/output.csv \
  --max_workers 2 \
  --retry_attempts 5
```

**Resume and Checkpointing:**
```bash
# Resume automatically from checkpoint
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/output.csv

# Resume from specific row (ignores checkpoint)
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/output.csv \
  --resume_from 150

# Custom checkpoint frequency
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/output.csv \
  --checkpoint_every 25
```

**Custom Logging:**
```bash
# Custom log filename
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/output.csv \
  --log_filename run_001_claude.log

# Different runs with organized logs
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/gemini_run.csv \
  --llm_model gemini \
  --log_filename $(date +%Y%m%d_%H%M)_gemini_validation.log
```

### 2. Automatic Retry Functionality (Built-in)

The main validation script now includes built-in retry functionality! Failed samples are automatically treated as remaining samples when you resume.

#### Basic Retry Usage
```bash
# Automatically retry failed validations from previous run
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/results.csv \
  --retry_failed
```

#### Advanced Retry Options

**Selective Failure Types:**
```bash
# Retry only specific failure types
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/results.csv \
  --retry_failed \
  --failure_types TIMEOUT_ERROR RATE_LIMIT_ERROR

# Retry all retryable failures (default)
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/results.csv \
  --retry_failed \
  --failure_types FAILED_AFTER TIMEOUT_ERROR RATE_LIMIT_ERROR CONNECTION_ERROR PROCESSING_ERROR
```

**Combined with Other Options:**
```bash
# Retry failures with different model and settings
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/results.csv \
  --retry_failed \
  --llm_model gemini \
  --max_workers 2 \
  --retry_attempts 5 \
  --log_filename retry_with_gemini.log
```

---

## Analysis Scripts

### 3. Validation Results Analysis (`validation_analysis_non-code.py`)

Analyzes validation results and generates comprehensive reports.

#### Basic Usage
```bash
python validation_analysis_non-code.py \
  --filepath data/validated_results.csv
```

#### Advanced Analysis Options
```bash
# Save plots to files
python validation_analysis_non-code.py \
  --filepath data/validated_results.csv \
  --save_plots

# Generate detailed summary file
python validation_analysis_non-code.py \
  --filepath data/validated_results.csv \
  --output_summary analysis_report.txt \
  --save_plots
```

### 4. Error Analysis (`error_analysis.py`)

Analyzes compilation errors from code validation results.

#### Basic Usage
```bash
python error_analysis.py \
  --filepath data/code_validation_results.csv
```

#### Advanced Error Analysis
```bash
# Show top 10 errors instead of default 5
python error_analysis.py \
  --filepath data/code_validation_results.csv \
  --top_n 10
```

---

## Code Validation Scripts

### 5. Code Compilation and Testing (`code_validation.py`)

Compiles and tests Rust code samples using Docker.

#### Basic Usage
```bash
python code_validation.py \
  --filepath data/rust_dataset.csv \
  --output_file data/code_validation_results.csv
```

#### Advanced Code Validation
```bash
# Custom column names
python code_validation.py \
  --filepath data/custom_dataset.csv \
  --task_column my_task_col \
  --input_column my_input_col \
  --output_column my_output_col \
  --output_file data/results.csv

# Disable auto-parsing of dependencies
python code_validation.py \
  --filepath data/dataset.csv \
  --output_file data/results.csv \
  --autoparse_dependencies false
```

### 6. Dataset Quality Evaluation (`eval_data.py`)

Evaluates and potentially improves dataset quality using LLM.

#### Basic Usage
```bash
python eval_data.py \
  --input_filepath data/rust_dataset.csv \
  --output_filepath data/improved_dataset.csv
```

#### Advanced Evaluation
```bash
# More samples per category and different model
python eval_data.py \
  --input_filepath data/dataset.csv \
  --output_filepath data/improved.csv \
  --nsamples_per_cat 25 \
  --llm_model claude
```

---

## Common Usage Patterns

### Improved Workflow with Built-in Retry

The retry functionality is now integrated, making the workflow much simpler:

```bash
# 1. Run initial validation
python llm_validation_non_code.py \
  --input_filepath data/rust_dataset_100k.csv \
  --output_filepath data/validation_results.csv \
  --nsamples_per_task 10 \
  --log_filename validation_run.log

# 2. If interrupted or some samples failed, simply re-run with retry flag
# This will automatically:
# - Resume from checkpoint (incomplete samples)
# - Retry failed samples that match failure_types
python llm_validation_non_code.py \
  --input_filepath data/rust_dataset_100k.csv \
  --output_filepath data/validation_results.csv \
  --retry_failed \
  --log_filename retry_run.log

# 3. Analyze final results
python validation_analysis_non-code.py \
  --filepath data/validation_results.csv \
  --save_plots \
  --output_summary final_analysis.txt
```

### Full Pipeline Example (Legacy)
```bash
# 1. Initial validation with small sample
python llm_validation_non_code.py \
  --input_filepath data/rust_dataset_100k.csv \
  --output_filepath data/initial_validation.csv \
  --nsamples_per_task 5 \
  --log_filename initial_run.log

# 2. Analyze initial results
python validation_analysis_non-code.py \
  --filepath data/initial_validation.csv \
  --save_plots \
  --output_summary initial_analysis.txt

# 3. Retry any failures (now built-in)
python llm_validation_non_code.py \
  --input_filepath data/rust_dataset_100k.csv \
  --output_filepath data/initial_validation.csv \
  --retry_failed \
  --log_filename retry_run.log

# 4. Full validation if initial looks good
python llm_validation_non_code.py \
  --input_filepath data/rust_dataset_100k.csv \
  --output_filepath data/full_validation.csv \
  --validate_all \
  --max_samples_per_task 500 \
  --log_filename full_run.log
```

### Model Comparison Workflow
```bash
# Test different models on same sample
python llm_validation_non_code.py \
  --input_filepath data/dataset.csv \
  --output_filepath data/claude_results.csv \
  --llm_model claude \
  --nsamples_per_task 10 \
  --log_filename claude_comparison.log

python llm_validation_non_code.py \
  --input_filepath data/dataset.csv \
  --output_filepath data/gemini_results.csv \
  --llm_model gemini \
  --nsamples_per_task 10 \
  --log_filename gemini_comparison.log

# Analyze both results
python validation_analysis_non-code.py \
  --filepath data/claude_results.csv \
  --output_summary claude_analysis.txt

python validation_analysis_non-code.py \
  --filepath data/gemini_results.csv \
  --output_summary gemini_analysis.txt
```

### Code Validation Workflow
```bash
# 1. Validate code compilation
python code_validation.py \
  --filepath data/dataset_with_code.csv \
  --output_file data/compilation_results.csv

# 2. Analyze compilation errors
python error_analysis.py \
  --filepath data/compilation_results.csv \
  --top_n 15

# 3. Improve dataset based on errors
python eval_data.py \
  --input_filepath data/dataset_with_code.csv \
  --output_filepath data/improved_dataset.csv \
  --llm_model claude
```

---

## Logging and Debugging

### Log File Locations
- Log files are saved in the **same directory where you run the script**
- Default filenames: `llm_validation.log`, `retry_failures.log`
- Custom filenames can be specified with `--log_filename`

### Monitoring Progress
```bash
# Watch logs in real-time
tail -f your_log_file.log

# Watch with specific patterns
tail -f validation_run.log | grep -E "(ERROR|completed|FAILED)"

# Count progress
grep "completed" validation_run.log | wc -l
```

### Debugging Failed Runs
```bash
# Check what failed in validation
python validation_analysis_non-code.py \
  --filepath data/results.csv

# Analyze specific error types
grep "ERROR" your_log_file.log | head -20

# Resume from specific point if needed
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/output.csv \
  --resume_from 150 \
  --log_filename debug_resume.log
```

### Performance Optimization Tips

1. **Start Small**: Always test with small samples first (`--nsamples_per_task 3`)
2. **Monitor Resources**: Use fewer workers if hitting rate limits (`--max_workers 2`)
3. **Custom Logs**: Use descriptive log filenames for different runs
4. **Checkpointing**: Let automatic checkpointing handle interruptions
5. **Retry Strategy**: Use retry_failure.py for failed validations rather than rerunning everything

### Troubleshooting

**Common Issues:**
- **Rate Limiting**: Reduce `--max_workers`, increase `--retry_attempts`
- **Memory Issues**: Process in smaller batches using `--max_samples_per_task`
- **Network Issues**: Use retry_failure.py to handle transient failures
- **Disk Space**: Monitor log file sizes, use custom filenames to organize

**Emergency Recovery:**
```bash
# If process crashes, resume automatically
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/output.csv
  # Checkpoint will be loaded automatically

# If checkpoint is corrupted, resume from specific point
python llm_validation_non_code.py \
  --input_filepath data/input.csv \
  --output_filepath data/output.csv \
  --resume_from <last_known_good_row>
```

--- 