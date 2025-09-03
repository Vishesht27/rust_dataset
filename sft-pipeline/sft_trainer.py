#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) Pipeline for Qwen2.5 32B Model
Based on foundation-model-stack/fms-hf-tuning implementation
Supports multi-GPU training with FSDP and no GPU memory constraints
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SFTTrainerPipeline:
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.accelerator = None
        
        # Initialize accelerator for multi-GPU support
        self.accelerator = Accelerator()
        
        # Set seeds
        seed = self.config.get("seed", 42)
        set_seed(seed)
        accelerate_set_seed(seed)
        
        logger.info(f"Initialized with {self.accelerator.num_processes} processes")
        logger.info(f"Using device: {self.accelerator.device}")
        
    def load_model_and_tokenizer(self):
        model_name = self.config.get("model_name", "Qwen/Qwen2.5-32B-Instruct")
        
        logger.info(f"Loading model: {model_name}")
        
        # Load model with full precision (no quantization for better performance)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better numerical stability
            trust_remote_code=True,
            device_map=None,  # Let accelerator handle device placement
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Model and tokenizer loaded successfully")
        
    def setup_lora(self):
        """Setup LoRA for efficient fine-tuning"""
        if not self.config.get("use_lora", True):
            return
            
        logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get("lora_r", 64),  # Higher rank for better performance
            lora_alpha=self.config.get("lora_alpha", 128),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            target_modules=self.config.get("lora_target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "lm_head"  # Include output layer
            ]),
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        if self.accelerator.is_main_process:
            self.model.print_trainable_parameters()
        
    def load_dataset(self) -> Dataset:
        """Load and prepare the training dataset"""
        dataset_path = self.config.get("dataset_path")
        dataset_format = self.config.get("dataset_format", "json")
        
        if not dataset_path:
            raise ValueError("Dataset path not specified in config")
            
        logger.info(f"Loading dataset from: {dataset_path}")
        
        if dataset_format == "json":
            if dataset_path.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=dataset_path, split="train")
            else:
                dataset = load_dataset("json", data_files=dataset_path, split="train")
        elif dataset_format == "csv":
            dataset = load_dataset("csv", data_files=dataset_path, split="train")
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")
            
        logger.info(f"Dataset loaded with {len(dataset)} examples")
        return dataset
        
    def format_dataset(self, dataset: Dataset) -> Dataset:
        """Format dataset for SFT training"""
        def format_conversation(example):
            """Format a single conversation example"""
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
                    # Handle CSV format with input_data/output_data
                    import json
                    try:
                        input_json = json.loads(example["input_data"])
                        output_json = json.loads(example["output_data"])
                        
                        # Extract query/question from input
                        query = input_json.get("query", "")
                        if not query:
                            query = input_json.get("title", "")
                        if not query:
                            query = input_json.get("description", "")
                        
                        # Extract code from output
                        code = output_json.get("code", "")
                        if not code:
                            code = output_json.get("code_snippet", "")
                        if not code:
                            code = output_json.get("code_after", "")
                        
                        if query and code:
                            messages = [
                                {"role": "user", "content": f"Please help me with this Rust programming task: {query}"},
                                {"role": "assistant", "content": f"Here's the solution:\n\n```rust\n{code}\n```"}
                            ]
                        else:
                            return {"text": ""}
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse CSV data: {e}")
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
        formatted_dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
        
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
        
    def create_training_args(self) -> SFTConfig:
        """Create training arguments with multi-GPU support"""
        # Calculate effective batch size
        per_device_batch_size = self.config.get("per_device_train_batch_size", 2)
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 4)
        num_processes = self.accelerator.num_processes
        
        effective_batch_size = per_device_batch_size * gradient_accumulation_steps * num_processes
        
        logger.info(f"Effective batch size: {effective_batch_size} "
                   f"(per_device: {per_device_batch_size}, "
                   f"grad_accum: {gradient_accumulation_steps}, "
                   f"processes: {num_processes})")
        
        # Check if multi-GPU training is enabled
        use_multi_gpu = self.config.get("use_multi_gpu", False) and num_processes > 1
        logger.info(f"Multi-GPU training: {'enabled' if use_multi_gpu else 'disabled'}")
        
        return SFTConfig(
            output_dir=self.config.get("output_dir", "./qwen2.5-32b-sft"),
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=self.config.get("per_device_eval_batch_size", 2),
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=self.config.get("learning_rate", 2e-5),
            num_train_epochs=self.config.get("num_train_epochs", 3),
            max_steps=self.config.get("max_steps", -1),
            warmup_steps=self.config.get("warmup_steps", 100),
            logging_steps=self.config.get("logging_steps", 10),
            save_steps=self.config.get("save_steps", 500),
            eval_steps=self.config.get("eval_steps", 500),
            save_strategy=self.config.get("save_strategy", "steps"),
            load_best_model_at_end=False,  # Disable since we don't have eval dataset
            metric_for_best_model=self.config.get("metric_for_best_model", "eval_loss"),
            greater_is_better=False,
            fp16=False,  # Use bf16 instead
            bf16=True,   # Better numerical stability
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            push_to_hub=self.config.get("push_to_hub", False),
            hub_model_id=self.config.get("hub_model_id"),
            hub_token=self.config.get("hub_token"),
            report_to=self.config.get("report_to", "wandb" if self.config.get("use_wandb", False) else "none"),
            run_name=self.config.get("run_name", "qwen2.5-32b-sft"),
            seed=self.config.get("seed", 42),
            data_seed=self.config.get("data_seed", 42),
            # Multi-GPU specific settings
            **({} if not use_multi_gpu else {
                "ddp_find_unused_parameters": False,
                "ddp_backend": "nccl",
                "fsdp": self.config.get("fsdp", "full_shard auto_wrap"),
                "fsdp_config": self.config.get("fsdp_config", {
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "fsdp_backward_prefetch": "BACKWARD_PRE",
                    "fsdp_cpu_ram_efficient_loading": True,
                    "fsdp_forward_prefetch": True,
                    "fsdp_offload_params": False,
                    "fsdp_sharding_strategy": "FULL_SHARD",
                    "fsdp_sync_module_states": True,
                    "fsdp_use_orig_params": False,
                }),
            }),
            # Gradient checkpointing for memory efficiency
            gradient_checkpointing=True,
            # Optimizer settings
            optim=self.config.get("optim", "adamw_torch_fused"),
            adam_beta1=self.config.get("adam_beta1",0.9),
            adam_beta2=self.config.get("adam_beta2",0.95),
            adam_epsilon=self.config.get("adam_epsilon",1e-8),
            weight_decay=self.config.get("weight_decay",0.01),
            max_grad_norm=self.config.get("max_grad_norm",1.0),
            # Learning rate scheduler
            lr_scheduler_type=self.config.get("lr_scheduler_type","cosine"),
            warmup_ratio=self.config.get("warmup_ratio",0.1),
            # SFT-specific parameters
            max_length=self.config.get("max_seq_length", 4096),
            dataset_text_field="text",
            packing=self.config.get("packing", True),
        )
        
    def create_trainer(self, dataset: Dataset) -> SFTTrainer:
        """Create the SFT trainer with multi-GPU support"""
        training_args = self.create_training_args()
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=None,  
            processing_class=self.tokenizer,
        )

        # Only add data_collator if packing is disabled
        if not self.config.get("packing", True):
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            trainer_kwargs["data_collator"] = data_collator
        
        return trainer
        
    def train(self):
        """Main training function with multi-GPU support"""
        if self.accelerator.is_main_process:
            logger.info("Starting SFT training pipeline")
            
            # Initialize wandb if enabled
            if self.config.get("use_wandb", False):
                wandb.init(
                    project=self.config.get("wandb_project", "qwen2.5-sft"),
                    name=self.config.get("run_name", "qwen2.5-32b-sft"),
                    config=self.config
                )
        
        try:
            # Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Setup LoRA if enabled
            self.setup_lora()
            
            # Load and format dataset
            dataset = self.load_dataset()
            formatted_dataset = self.format_dataset(dataset)
            
            # Create trainer
            self.trainer = self.create_trainer(formatted_dataset)
            
            # Prepare for multi-GPU training if enabled
            use_multi_gpu = self.config.get("use_multi_gpu", False) and self.accelerator.num_processes > 1
            if use_multi_gpu:
                self.trainer.model, self.trainer.optimizer, self.trainer.lr_scheduler = self.accelerator.prepare(
                    self.trainer.model, self.trainer.optimizer, self.trainer.lr_scheduler
                )
            
            # Start training
            if self.accelerator.is_main_process:
                logger.info("Starting training...")
            
            self.trainer.train()
            
            # Save model
            if self.accelerator.is_main_process:
                logger.info("Saving model...")
                self.trainer.save_model()
                self.tokenizer.save_pretrained(self.config.get("output_dir", "./qwen2.5-32b-sft"))
                logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            if self.accelerator.is_main_process and self.config.get("use_wandb", False):
                wandb.finish()


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SFT Training Pipeline for Qwen2.5 32B")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--dataset", type=str, help="Path to dataset file (overrides config)")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--model_name", type=str, help="Model name (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.dataset:
        config["dataset_path"] = args.dataset
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.model_name:
        config["model_name"] = args.model_name
    
    # Create and run training pipeline
    pipeline = SFTTrainerPipeline(config)
    pipeline.train()


if __name__ == "__main__":
    main()
