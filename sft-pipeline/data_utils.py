#!/usr/bin/env python3
"""
Data utilities for SFT training pipeline
Handles dataset loading, preprocessing, and formatting
"""

import json
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def parse_rust_dataset_format(input_data: str, output_data: str, task_category: str) -> Optional[List[Dict[str, str]]]:
    """
    Comprehensive parser for Rust dataset formats
    Task-category-driven parsing that handles all format combinations
    
    Args:
        input_data: JSON string containing input data
        output_data: JSON string containing output data  
        task_category: Task category to determine parsing strategy
        
    Returns:
        List of conversation messages or None if parsing fails
    """
    try:
        input_json = json.loads(input_data)
        output_json = json.loads(output_data)
        
        # Create user prompt based on task category and available input fields
        user_prompt = _create_task_based_user_prompt(input_json, task_category)
        if not user_prompt:
            return None
            
        # Create assistant response based on task category and available output fields
        assistant_response = _create_task_based_assistant_response(output_json, task_category)
        if not assistant_response:
            return None
            
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
        
    except Exception as e:
        logger.warning(f"Failed to parse Rust dataset format for task {task_category}: {e}")
        return None


def _create_task_based_user_prompt(input_json: Dict, task_category: str) -> Optional[str]:
    """Create user prompt based on task category, intelligently using available input fields"""
    
    # Task-specific prompt creation strategies
    if task_category == 'comment_generation':
        return _create_comment_generation_prompt(input_json)
    elif task_category == 'code_explanation':
        return _create_code_explanation_prompt(input_json)
    elif task_category == 'docstring_generation':
        return _create_docstring_generation_prompt(input_json)
    elif task_category == 'code_generation':
        return _create_code_generation_prompt(input_json)
    elif task_category == 'code_search':
        return _create_code_search_prompt(input_json)
    elif task_category == 'code_summarization':
        return _create_code_summarization_prompt(input_json)
    elif task_category == 'code_review':
        return _create_code_review_prompt(input_json)
    elif task_category == 'test_generation':
        return _create_test_generation_prompt(input_json)
    elif task_category == 'code_refactoring':
        return _create_code_refactoring_prompt(input_json)
    elif task_category == 'variable_naming':
        return _create_variable_naming_prompt(input_json)
    elif task_category == 'function_naming':
        return _create_function_naming_prompt(input_json)
    elif task_category == 'api_usage_prediction':
        return _create_api_usage_prompt(input_json)
    elif task_category == 'bug_detection':
        return _create_bug_detection_prompt(input_json)
    elif task_category == 'code_optimization':
        return _create_code_optimization_prompt(input_json)
    elif task_category == 'code_completion':
        return _create_code_completion_prompt(input_json)
    else:
        # Fallback for unknown task categories
        return _create_generic_prompt(input_json)


def _create_task_based_assistant_response(output_json: Dict, task_category: str) -> Optional[str]:
    """Create assistant response based on task category, intelligently using available output fields"""
    
    # Task-specific response creation strategies
    if task_category == 'comment_generation':
        return _create_comment_generation_response(output_json)
    elif task_category == 'code_explanation':
        return _create_code_explanation_response(output_json)
    elif task_category == 'docstring_generation':
        return _create_docstring_generation_response(output_json)
    elif task_category == 'code_generation':
        return _create_code_generation_response(output_json)
    elif task_category == 'code_search':
        return _create_code_search_response(output_json)
    elif task_category == 'code_summarization':
        return _create_code_summarization_response(output_json)
    elif task_category == 'code_review':
        return _create_code_review_response(output_json)
    elif task_category == 'test_generation':
        return _create_test_generation_response(output_json)
    elif task_category == 'code_refactoring':
        return _create_code_refactoring_response(output_json)
    elif task_category == 'variable_naming':
        return _create_variable_naming_response(output_json)
    elif task_category == 'function_naming':
        return _create_function_naming_response(output_json)
    elif task_category == 'api_usage_prediction':
        return _create_api_usage_response(output_json)
    elif task_category == 'bug_detection':
        return _create_bug_detection_response(output_json)
    elif task_category == 'code_optimization':
        return _create_code_optimization_response(output_json)
    elif task_category == 'code_completion':
        return _create_code_completion_response(output_json)
    else:
        # Fallback for unknown task categories
        return _create_generic_response(output_json)


# Task-specific prompt creation functions
def _create_comment_generation_prompt(input_json: Dict) -> str:
    """Create prompt for comment generation tasks"""
    code = input_json.get('code', '')
    context = input_json.get('code_context', '')
    
    prompt = "Add helpful comments to this Rust code:\n\n```rust\n" + code + "\n```"
    if context:
        prompt += f"\n\nContext:\n```rust\n{context}\n```"
    return prompt


def _create_code_explanation_prompt(input_json: Dict) -> str:
    """Create prompt for code explanation tasks"""
    code = input_json.get('code', '')
    context = input_json.get('code_context', '')
    
    prompt = "Explain what this Rust code does:\n\n```rust\n" + code + "\n```"
    if context:
        prompt += f"\n\nContext:\n```rust\n{context}\n```"
    return prompt


def _create_docstring_generation_prompt(input_json: Dict) -> str:
    """Create prompt for docstring generation tasks"""
    code = input_json.get('code', '')
    context = input_json.get('code_context', '')
    
    prompt = "Generate documentation for this Rust code:\n\n```rust\n" + code + "\n```"
    if context:
        prompt += f"\n\nContext:\n```rust\n{context}\n```"
    return prompt


def _create_code_generation_prompt(input_json: Dict) -> str:
    """Create prompt for code generation tasks"""
    if 'title' in input_json and 'description' in input_json:
        title = input_json.get('title', '')
        description = input_json.get('description', '')
        context = input_json.get('code_context', '')
        
        prompt = f"**{title}**\n\n{description}"
        if context:
            prompt += f"\n\nAvailable imports/context:\n```rust\n{context}\n```"
        return prompt
    else:
        return _create_generic_prompt(input_json)


def _create_code_search_prompt(input_json: Dict) -> str:
    """Create prompt for code search tasks"""
    query = input_json.get('query', '')
    context = input_json.get('code_context')
    
    if context:
        return f"{query}\n\nContext:\n```rust\n{context}\n```"
    else:
        return query


def _create_code_summarization_prompt(input_json: Dict) -> str:
    """Create prompt for code summarization tasks"""
    code = input_json.get('code', '')
    context = input_json.get('code_context', '')
    
    prompt = "Summarize this Rust code:\n\n```rust\n" + code + "\n```"
    if context:
        prompt += f"\n\nContext:\n```rust\n{context}\n```"
    return prompt


def _create_code_review_prompt(input_json: Dict) -> str:
    """Create prompt for code review tasks"""
    code = input_json.get('code', '')
    context = input_json.get('code_context', '')
    
    prompt = "Review this Rust code:\n\n```rust\n" + code + "\n```"
    if context:
        prompt += f"\n\nContext:\n```rust\n{context}\n```"
    return prompt


def _create_test_generation_prompt(input_json: Dict) -> str:
    """Create prompt for test generation tasks"""
    code_to_test = input_json.get('code_to_test', '')
    context = input_json.get('code_context', '')
    test_context = input_json.get('test_context')
    
    prompt = f"Generate comprehensive unit tests for this Rust code:\n\n```rust\n{code_to_test}\n```"
    if context:
        prompt += f"\n\nContext:\n```rust\n{context}\n```"
    if test_context:
        prompt += f"\n\nTest context: {test_context}"
    return prompt


def _create_code_refactoring_prompt(input_json: Dict) -> str:
    """Create prompt for code refactoring tasks"""
    code_before = input_json.get('code_before', input_json.get('code', ''))
    context = input_json.get('code_context', '')
    
    prompt = f"Refactor this Rust code to improve it:\n\n```rust\n{code_before}\n```"
    if context:
        prompt += f"\n\nContext:\n```rust\n{context}\n```"
    return prompt


def _create_variable_naming_prompt(input_json: Dict) -> str:
    """Create prompt for variable naming tasks"""
    code = input_json.get('code', '')
    context = input_json.get('code_context', '')
    
    prompt = f"Suggest a good variable name for this Rust code:\n\n```rust\n{code}\n```"
    if context:
        prompt += f"\n\nContext:\n```rust\n{context}\n```"
    return prompt


def _create_function_naming_prompt(input_json: Dict) -> str:
    """Create prompt for function naming tasks"""
    code = input_json.get('code', '')
    context = input_json.get('code_context', '')
    
    prompt = f"Suggest a good function name for this Rust code:\n\n```rust\n{code}\n```"
    if context:
        prompt += f"\n\nContext:\n```rust\n{context}\n```"
    return prompt


def _create_api_usage_prompt(input_json: Dict) -> str:
    """Create prompt for API usage prediction tasks"""
    code = input_json.get('code', '')
    context = input_json.get('code_context', '')
    
    prompt = f"What API call should come next in this Rust code?\n\n```rust\n{code}\n```"
    if context:
        prompt += f"\n\nContext:\n```rust\n{context}\n```"
    return prompt


def _create_bug_detection_prompt(input_json: Dict) -> str:
    """Create prompt for bug detection tasks"""
    buggy_code = input_json.get('buggy_code', input_json.get('code', ''))
    context = input_json.get('code_context', '')
    
    prompt = f"Find and fix the bug in this Rust code:\n\n```rust\n{buggy_code}\n```"
    if context:
        prompt += f"\n\nContext:\n```rust\n{context}\n```"
    return prompt


def _create_code_optimization_prompt(input_json: Dict) -> str:
    """Create prompt for code optimization tasks"""
    code = input_json.get('code', input_json.get('code_before', ''))
    context = input_json.get('code_context', '')
    
    prompt = f"Optimize this Rust code for better performance:\n\n```rust\n{code}\n```"
    if context:
        prompt += f"\n\nContext:\n```rust\n{context}\n```"
    return prompt


def _create_code_completion_prompt(input_json: Dict) -> str:
    """Create prompt for code completion tasks"""
    prefix = input_json.get('prefix', '')
    suffix = input_json.get('suffix', '')
    
    return f"Complete this Rust code:\n\n```rust\n{prefix}____{suffix}\n```"


def _create_generic_prompt(input_json: Dict) -> str:
    """Generic prompt creation for unknown formats"""
    if 'query' in input_json:
        return input_json['query']
    elif 'title' in input_json and 'description' in input_json:
        return f"{input_json['title']}: {input_json['description']}"
    elif 'title' in input_json:
        return input_json['title']
    elif 'description' in input_json:
        return input_json['description']
    elif 'code' in input_json:
        return f"Help me with this Rust code:\n\n```rust\n{input_json['code']}\n```"
    else:
        return "Help me with this Rust programming task."


# Task-specific response creation functions
def _create_comment_generation_response(output_json: Dict) -> str:
    """Create response for comment generation tasks"""
    if 'commented_code' in output_json:
        commented_code = output_json['commented_code']
        return f"Here's the code with helpful comments:\n\n```rust\n{commented_code}\n```"
    else:
        return _create_generic_response(output_json)


def _create_code_explanation_response(output_json: Dict) -> str:
    """Create response for code explanation tasks"""
    if 'explanation' in output_json:
        return output_json['explanation']
    else:
        return _create_generic_response(output_json)


def _create_docstring_generation_response(output_json: Dict) -> str:
    """Create response for docstring generation tasks"""
    if 'docstring' in output_json:
        docstring = output_json['docstring']
        return f"Here's the documentation:\n\n```rust\n{docstring}\n```"
    else:
        return _create_generic_response(output_json)


def _create_code_generation_response(output_json: Dict) -> str:
    """Create response for code generation tasks"""
    if 'code' in output_json:
        code = output_json['code']
        return f"Here's the solution:\n\n```rust\n{code}\n```"
    else:
        return _create_generic_response(output_json)


def _create_code_search_response(output_json: Dict) -> str:
    """Create response for code search tasks"""
    if 'code_snippet' in output_json:
        code_snippet = output_json['code_snippet']
        return f"```rust\n{code_snippet}\n```"
    else:
        return _create_generic_response(output_json)


def _create_code_summarization_response(output_json: Dict) -> str:
    """Create response for code summarization tasks"""
    if 'summary' in output_json:
        summary = output_json['summary']
        return f"**Summary:**\n{summary}"
    else:
        return _create_generic_response(output_json)


def _create_code_review_response(output_json: Dict) -> str:
    """Create response for code review tasks"""
    if 'code_after' in output_json and 'review_comment' in output_json:
        code_after = output_json['code_after']
        review_comment = output_json['review_comment']
        return f"**Review Comment:**\n{review_comment}\n\n**Improved Code:**\n```rust\n{code_after}\n```"
    elif 'review_comment' in output_json:
        return output_json['review_comment']
    else:
        return _create_generic_response(output_json)


def _create_test_generation_response(output_json: Dict) -> str:
    """Create response for test generation tasks"""
    if 'test_cases' in output_json:
        test_cases = output_json['test_cases']
        if isinstance(test_cases, list):
            formatted_tests = '\n\n'.join(test_cases)
            return f"Here are comprehensive unit tests:\n\n```rust\n{formatted_tests}\n```"
        else:
            return f"Here are the unit tests:\n\n```rust\n{test_cases}\n```"
    else:
        return _create_generic_response(output_json)


def _create_code_refactoring_response(output_json: Dict) -> str:
    """Create response for code refactoring tasks"""
    if 'code_after' in output_json and 'rationale' in output_json:
        code_after = output_json['code_after']
        rationale = output_json['rationale']
        return f"{rationale}\n\nHere's the improved code:\n\n```rust\n{code_after}\n```"
    elif 'code_after' in output_json:
        code_after = output_json['code_after']
        return f"Here's the refactored code:\n\n```rust\n{code_after}\n```"
    else:
        return _create_generic_response(output_json)


def _create_variable_naming_response(output_json: Dict) -> str:
    """Create response for variable naming tasks"""
    if 'variable_name' in output_json:
        var_name = output_json['variable_name']
        return f"A good variable name would be: `{var_name}`"
    else:
        return _create_generic_response(output_json)


def _create_function_naming_response(output_json: Dict) -> str:
    """Create response for function naming tasks"""
    if 'function_name' in output_json:
        func_name = output_json['function_name']
        return f"A good function name would be: `{func_name}`"
    else:
        return _create_generic_response(output_json)


def _create_api_usage_response(output_json: Dict) -> str:
    """Create response for API usage prediction tasks"""
    if 'next_api_call' in output_json:
        next_api = output_json['next_api_call']
        return f"Next API call: `{next_api}`"
    else:
        return _create_generic_response(output_json)


def _create_bug_detection_response(output_json: Dict) -> str:
    """Create response for bug detection tasks"""
    if 'bug_description' in output_json and 'fixed_code' in output_json:
        bug_desc = output_json['bug_description']
        fixed_code = output_json['fixed_code']
        return f"**Bug Description:**\n{bug_desc}\n\n**Fixed Code:**\n```rust\n{fixed_code}\n```"
    elif 'fixed_code' in output_json:
        fixed_code = output_json['fixed_code']
        return f"Here's the fixed code:\n\n```rust\n{fixed_code}\n```"
    else:
        return _create_generic_response(output_json)


def _create_code_optimization_response(output_json: Dict) -> str:
    """Create response for code optimization tasks"""
    if 'code_after' in output_json and 'rationale' in output_json:
        code_after = output_json['code_after']
        rationale = output_json['rationale']
        return f"{rationale}\n\nHere's the optimized code:\n\n```rust\n{code_after}\n```"
    elif 'code_after' in output_json:
        code_after = output_json['code_after']
        return f"Here's the optimized code:\n\n```rust\n{code_after}\n```"
    else:
        return _create_generic_response(output_json)


def _create_code_completion_response(output_json: Dict) -> str:
    """Create response for code completion tasks"""
    if 'completion' in output_json:
        completion = output_json['completion']
        return completion
    else:
        return _create_generic_response(output_json)


def _create_generic_response(output_json: Dict) -> str:
    """Generic response creation for unknown formats"""
    # Try common output fields in order of preference
    if 'code' in output_json:
        return f"```rust\n{output_json['code']}\n```"
    elif 'code_snippet' in output_json:
        return f"```rust\n{output_json['code_snippet']}\n```"
    elif 'code_after' in output_json:
        return f"```rust\n{output_json['code_after']}\n```"
    elif 'explanation' in output_json:
        return output_json['explanation']
    elif 'docstring' in output_json:
        return f"```rust\n{output_json['docstring']}\n```"
    elif 'summary' in output_json:
        return output_json['summary']
    else:
        return "I can help you with that Rust programming task."


class DatasetProcessor:
    """Dataset processor for SFT training"""
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_dataset_from_file(self, file_path: str, format: str = "json") -> Dataset:
        """Load dataset from various file formats"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        logger.info(f"Loading dataset from {file_path} (format: {format})")
        
        if format == "json" or format == "jsonl":
            if file_path.suffix == ".jsonl":
                dataset = load_dataset("json", data_files=str(file_path), split="train")
            else:
                dataset = load_dataset("json", data_files=str(file_path), split="train")
        elif format == "csv":
            dataset = load_dataset("csv", data_files=str(file_path), split="train")
        elif format == "parquet":
            dataset = load_dataset("parquet", data_files=str(file_path), split="train")
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        return dataset
        
    def format_conversation_data(self, dataset: Dataset) -> Dataset:
        """Format conversation data for SFT training"""
        def format_example(example):
            """Format a single example"""
            messages = example.get("messages", [])
            
            if not messages:
                # Handle different data formats
                if "conversation" in example:
                    messages = example["conversation"]
                elif "instruction" in example and "response" in example:
                    messages = [
                        {"role": "user", "content": example["instruction"]},
                        {"role": "assistant", "content": example["response"]}
                    ]
                elif "prompt" in example and "completion" in example:
                    messages = [
                        {"role": "user", "content": example["prompt"]},
                        {"role": "assistant", "content": example["completion"]}
                    ]
                elif "input_data" in example and "output_data" in example:
                    # Handle Rust dataset format with comprehensive parser
                    messages = parse_rust_dataset_format(
                        example["input_data"], 
                        example["output_data"], 
                        example.get("task_category", "unknown")
                    )
                    if not messages:
                        return {"text": ""}
                else:
                    logger.warning(f"Unknown data format in example: {example.keys()}")
                    return {"text": ""}
            
            # Apply chat template
            try:
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}")
                # Fallback formatting
                formatted_text = self._fallback_format(messages)
                
            return {"text": formatted_text}
        
        logger.info("Formatting dataset for SFT training")
        formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
        
        # Filter out empty examples
        formatted_dataset = formatted_dataset.filter(lambda x: len(x["text"].strip()) > 0)
        
        logger.info(f"Formatted dataset has {len(formatted_dataset)} examples")
        return formatted_dataset
        
    def _fallback_format(self, messages: List[Dict[str, str]]) -> str:
        """Fallback formatting when chat template fails"""
        formatted_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"Human: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
            else:
                formatted_parts.append(f"{role.title()}: {content}")
                
        return "\n\n".join(formatted_parts) + "\n"
        
    def filter_by_length(self, dataset: Dataset, max_length: Optional[int] = None) -> Dataset:
        """Filter dataset by sequence length"""
        if max_length is None:
            max_length = self.max_length
            
        def is_within_length(example):
            tokens = self.tokenizer.encode(example["text"], add_special_tokens=True)
            return len(tokens) <= max_length
            
        logger.info(f"Filtering dataset by max length: {max_length}")
        filtered_dataset = dataset.filter(is_within_length)
        
        logger.info(f"Filtered dataset has {len(filtered_dataset)} examples (was {len(dataset)})")
        return filtered_dataset
        
    def create_train_val_split(self, dataset: Dataset, val_ratio: float = 0.1) -> tuple:
        """Create train/validation split"""
        if val_ratio <= 0 or val_ratio >= 1:
            raise ValueError("val_ratio must be between 0 and 1")
            
        split_dataset = dataset.train_test_split(test_size=val_ratio, seed=42)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
        
        logger.info(f"Created train/val split: {len(train_dataset)} train, {len(val_dataset)} val")
        return train_dataset, val_dataset


class DatasetValidator:
    """Validate dataset format and content"""
    
    @staticmethod
    def validate_conversation_format(example: Dict[str, Any]) -> bool:
        """Validate that example has proper conversation format"""
        if "messages" not in example:
            return False
            
        messages = example["messages"]
        if not isinstance(messages, list):
            return False
            
        for message in messages:
            if not isinstance(message, dict):
                return False
            if "role" not in message or "content" not in message:
                return False
            if message["role"] not in ["system", "user", "assistant"]:
                return False
                
        return True
        
    @staticmethod
    def validate_dataset(dataset: Dataset) -> Dict[str, Any]:
        """Validate entire dataset and return statistics"""
        stats = {
            "total_examples": len(dataset),
            "valid_examples": 0,
            "invalid_examples": 0,
            "role_distribution": {"system": 0, "user": 0, "assistant": 0},
            "avg_conversation_length": 0,
            "max_conversation_length": 0,
            "min_conversation_length": float('inf')
        }
        
        total_length = 0
        
        for example in dataset:
            if DatasetValidator.validate_conversation_format(example):
                stats["valid_examples"] += 1
                
                messages = example["messages"]
                stats["max_conversation_length"] = max(stats["max_conversation_length"], len(messages))
                stats["min_conversation_length"] = min(stats["min_conversation_length"], len(messages))
                total_length += len(messages)
                
                for message in messages:
                    role = message["role"]
                    if role in stats["role_distribution"]:
                        stats["role_distribution"][role] += 1
            else:
                stats["invalid_examples"] += 1
                
        if stats["valid_examples"] > 0:
            stats["avg_conversation_length"] = total_length / stats["valid_examples"]
            
        if stats["min_conversation_length"] == float('inf'):
            stats["min_conversation_length"] = 0
            
        return stats


def create_sample_dataset(output_path: str, num_examples: int = 100):
    """Create a sample dataset for testing"""
    sample_data = []
    
    for i in range(num_examples):
        example = {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": f"Question {i+1}: What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        }
        sample_data.append(example)
        
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in sample_data:
            f.write(json.dumps(example) + '\n')
            
    logger.info(f"Created sample dataset with {num_examples} examples at {output_path}")


def convert_csv_to_jsonl(csv_path: str, output_path: str, 
                        instruction_col: str = "instruction", 
                        response_col: str = "response"):
    """Convert CSV dataset to JSONL format for SFT training"""
    df = pd.read_csv(csv_path)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            example = {
                "messages": [
                    {"role": "user", "content": str(row[instruction_col])},
                    {"role": "assistant", "content": str(row[response_col])}
                ]
            }
            f.write(json.dumps(example) + '\n')
            
    logger.info(f"Converted CSV to JSONL: {len(df)} examples -> {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample dataset
    create_sample_dataset("./data/sample_dataset.jsonl", 50)
    
    # Example of converting CSV to JSONL
    # convert_csv_to_jsonl("input.csv", "output.jsonl", "question", "answer")
