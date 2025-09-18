#!/usr/bin/env python3
"""
Data utilities for SFT training pipeline
Handles dataset loading, preprocessing, and formatting
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from prompts_and_responses import format_prompt_response_pair, add_context_to_prompt

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
        
        # Use paired prompt-response system
        user_prompt, assistant_response = _create_paired_prompt_response(input_json, output_json, task_category)
        if not user_prompt or not assistant_response:
            return None
            
        return [
            {"role": "system", "content": "You are an expert in Rust programming language."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
        
    except Exception as e:
        logger.warning(f"Failed to parse Rust dataset format for task {task_category}: {e}")
        return None


def _create_paired_prompt_response(input_json: Dict, output_json: Dict, task_category: str) -> tuple[Optional[str], Optional[str]]:
    """Create paired prompt and response using the new prompt-response system"""
    try:
        # Prepare variables for prompt formatting
        prompt_vars = {}
        response_vars = {}
        
        # Common input variables
        if 'code' in input_json:
            prompt_vars['code'] = input_json['code']

        if 'code_context' in input_json:
            context = input_json['code_context']
        else:
            context = ''
            
        # Task-specific variable mapping
        if task_category == 'comment_generation':
            if 'commented_code' in output_json:
                response_vars['commented_code'] = output_json['commented_code']
        
        elif task_category == 'code_explanation':
                response_vars['explanation'] = output_json['explanation']
        
        elif task_category == 'docstring_generation':
                response_vars['docstring'] = output_json['docstring']
        
        elif task_category == 'code_generation':
                prompt_vars['title'] = input_json['title']
                prompt_vars['description'] = input_json['description']
                response_vars['code'] = output_json['code']
        
        elif task_category == 'code_search':
                prompt_vars['query'] = input_json['query']
                response_vars['code_snippet'] = output_json['code_snippet']
        
        elif task_category == 'code_summarization':
                response_vars['summary'] = output_json['summary']
        
        elif task_category == 'code_review':
                response_vars['review_comment'] = output_json['review_comment']
                response_vars['code_after'] = output_json['code_after']
        
        elif task_category == 'test_generation':
                prompt_vars['code_to_test'] = input_json['code_to_test']
                test_cases = output_json['test_cases']
                if isinstance(test_cases, list):
                    response_vars['test_cases'] = '\n\n'.join(test_cases)
                else:
                    response_vars['test_cases'] = test_cases
        
        elif task_category == 'code_refactoring':
            if 'code_before' in input_json:
                prompt_vars['code_before'] = input_json['code_before']
            elif 'code' in input_json:
                prompt_vars['code_before'] = input_json['code']
                response_vars['code_after'] = output_json['code_after']
                response_vars['rationale'] = output_json['rationale']
        
        elif task_category == 'variable_naming':
                response_vars['variable_name'] = output_json['variable_name']
        
        elif task_category == 'function_naming':
                response_vars['function_name'] = output_json['function_name']
        
        elif task_category == 'api_usage_prediction':
                response_vars['next_api_call'] = output_json['next_api_call']
        
        elif task_category == 'bug_detection':
            if 'buggy_code' in input_json:
                prompt_vars['buggy_code'] = input_json['buggy_code']
            elif 'code' in input_json:
                prompt_vars['buggy_code'] = input_json['code']

                response_vars['fixed_code'] = output_json['fixed_code']
                response_vars['bug_description'] = output_json['bug_description']
        
        elif task_category == 'code_optimization':
            if 'code_before' in input_json:
                prompt_vars['code'] = input_json['code_before']
            elif 'code' in input_json:
                prompt_vars['code'] = input_json['code']

                response_vars['code_after'] = output_json['code_after']
                response_vars['rationale'] = output_json['rationale']
        
        elif task_category == 'code_completion':
                prompt_vars['prefix'] = input_json['prefix']
                prompt_vars['suffix'] = input_json['suffix']
                response_vars['completion'] = output_json['completion']
        
        # Get paired prompt and response
        prompt, response = format_prompt_response_pair(task_category, prompt_vars, response_vars)
        
        # Add context to prompt if available
        if context:
            prompt = add_context_to_prompt(prompt, context)
        
        return prompt, response
        
    except Exception as e:
        logger.warning(f"Failed to create paired prompt-response for task {task_category}: {e}")
        return None, None





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
                if "input_data" in example and "output_data" in example:
                    # Handle Rust dataset format with comprehensive parser
                    messages = parse_rust_dataset_format(
                        example["input_data"], 
                        example["output_data"], 
                        example.get("task_category", "unknown")
                    )
                    if not messages:
                        return {"text": ""}
                elif "conversation" in example:
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
        # First, try to get messages from the example
        messages = DatasetValidator._extract_messages(example)
        
        if not messages:
            return False
            
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
    def _extract_messages(example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        """Extract messages from various data formats"""
        # Format 1: Already has messages field
        if "messages" in example:
            return example["messages"]
        
        # Format 2: Rust dataset format (input_data, output_data, task_category)
        elif "input_data" in example and "output_data" in example:
            try:
                return parse_rust_dataset_format(
                    example["input_data"],
                    example["output_data"], 
                    example.get("task_category", "unknown")
                )
            except Exception as e:
                logger.warning(f"Failed to parse Rust dataset format: {e}")
                return None
        
        # Format 3: Conversation field
        elif "conversation" in example:
            return example["conversation"]
        
        # Format 4: Standard instruction-response format
        elif "instruction" in example and "response" in example:
            return [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["response"]}
            ]
        
        # Format 5: Prompt-completion format
        elif "prompt" in example and "completion" in example:
            return [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["completion"]}
            ]
        
        else:
            return None
        
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
            "min_conversation_length": float('inf'),
            "format_distribution": {
                "messages": 0,
                "instruction_response": 0, 
                "prompt_completion": 0,
                "rust_dataset": 0,
                "conversation": 0,
                "unknown": 0
            }
        }
        
        total_length = 0
        
        for example in dataset:
            # Extract messages using our flexible parser
            messages = DatasetValidator._extract_messages(example)
            
            # Track format types
            format_type = DatasetValidator._detect_format_type(example)
            if format_type in stats["format_distribution"]:
                stats["format_distribution"][format_type] += 1
            else:
                stats["format_distribution"]["unknown"] += 1
            
            if messages and DatasetValidator._validate_messages_structure(messages):
                stats["valid_examples"] += 1
                
                stats["max_conversation_length"] = max(stats["max_conversation_length"], len(messages))
                stats["min_conversation_length"] = min(stats["min_conversation_length"], len(messages))
                total_length += len(messages)
                
                for message in messages:
                    role = message.get("role", "unknown")
                    if role in stats["role_distribution"]:
                        stats["role_distribution"][role] += 1
            else:
                stats["invalid_examples"] += 1
                
        if stats["valid_examples"] > 0:
            stats["avg_conversation_length"] = total_length / stats["valid_examples"]
            
        if stats["min_conversation_length"] == float('inf'):
            stats["min_conversation_length"] = 0
            
        return stats
    
    @staticmethod
    def _detect_format_type(example: Dict[str, Any]) -> str:
        """Detect the format type of an example"""
        if "messages" in example:
            return "messages"
        elif "instruction" in example and "response" in example:
            return "instruction_response"
        elif "prompt" in example and "completion" in example:
            return "prompt_completion"
        elif "input_data" in example and "output_data" in example:
            return "rust_dataset"
        elif "conversation" in example:
            return "conversation"
        else:
            return "unknown"
    
    @staticmethod
    def _validate_messages_structure(messages: List[Dict[str, str]]) -> bool:
        """Validate the structure of extracted messages"""
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
