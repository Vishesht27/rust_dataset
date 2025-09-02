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
