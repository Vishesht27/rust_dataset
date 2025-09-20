#!/usr/bin/env python3
"""
Fixed SFT Trainer with proper distributed training setup
Addresses NCCL and process group initialization issues
Includes smart DDP/FSDP detection based on configuration
"""

import os
import json
import logging
import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from data_utils import parse_rust_dataset_format

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Properly setup distributed training with NCCL"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Set the device before initializing process group
        torch.cuda.set_device(local_rank)
        
        # Initialize process group with explicit device_id
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                rank=rank,
                world_size=world_size,
                device_id=local_rank
            )
        
        logger.info(f"Distributed training setup: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return True, rank, world_size, local_rank
    else:
        logger.info("No distributed environment detected, using single GPU")
        return False, 0, 1, 0


class SFTTrainerPipeline:
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Setup distributed training first
        self.is_distributed, self.rank, self.world_size, self.local_rank = setup_distributed()
        
        # Set seeds
        seed = self.config.get("seed", 42)
        set_seed(seed)
        
        if self.rank == 0:
            logger.info("Fixed SFT Trainer Pipeline initialized")
        
    def load_model_and_tokenizer(self):
        model_name = self.config.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        
        if self.rank == 0:
            logger.info(f"Loading model: {model_name}")
        
        # Load model with proper device placement
        device_map = None
        if not self.is_distributed:
            device_map = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map,
        )
        
        # Move model to correct device for distributed training
        if self.is_distributed:
            self.model = self.model.to(f'cuda:{self.local_rank}')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        if self.rank == 0:
            logger.info("Model and tokenizer loaded successfully")
        
    def setup_lora(self):
        """Setup LoRA for efficient fine-tuning"""
        if not self.config.get("use_lora", True):
            return
            
        if self.rank == 0:
            logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get("lora_r", 64),
            lora_alpha=self.config.get("lora_alpha", 128),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            target_modules=self.config.get("lora_target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "lm_head"
            ]),
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        if self.rank == 0:
            self.model.print_trainable_parameters()
        
    def load_dataset(self) -> Dataset:
        """Load and prepare the training dataset"""
        dataset_path = self.config.get("dataset_path")
        dataset_format = self.config.get("dataset_format", "json")
        
        if not dataset_path:
            raise ValueError("Dataset path not specified in config")
            
        if self.rank == 0:
            logger.info(f"Loading dataset from: {dataset_path}")
        
        if dataset_format == "csv":
            dataset = load_dataset("csv", data_files=dataset_path, split="train")
        elif dataset_format == "json":
            dataset = load_dataset("json", data_files=dataset_path, split="train")
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")
            
        if self.rank == 0:
            logger.info(f"Dataset loaded with {len(dataset)} examples")
        return dataset
        
    def format_dataset(self, dataset: Dataset) -> Dataset:
        """Format dataset for SFT training"""
        def format_conversation(example):
            messages = example.get("messages", [])
            
            if not messages:
                if "input_data" in example and "output_data" in example:
                    messages = parse_rust_dataset_format(
                        example["input_data"],
                        example["output_data"], 
                        example.get("task_category", "unknown")
                    )
                    if not messages:
                        return {"text": ""}
                elif "instruction" in example and "response" in example:
                    messages = [
                        {"role": "user", "content": example["instruction"]},
                        {"role": "assistant", "content": example["response"]}
                    ]
                else:
                    return {"text": ""}
            
            try:
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception as e:
                formatted_text = self._fallback_format(messages)
                
            return {"text": formatted_text}
        
        if self.rank == 0:
            logger.info("Formatting dataset for SFT training")
        
        formatted_dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
        formatted_dataset = formatted_dataset.filter(lambda x: len(x["text"].strip()) > 0)
        
        if self.rank == 0:
            logger.info(f"Formatted dataset has {len(formatted_dataset)} examples")
        return formatted_dataset
    
    def _fallback_format(self, messages):
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
        return "\n\n".join(formatted_parts) + "\n"
        
    def create_training_args(self):
        """Create training arguments with smart DDP/FSDP detection"""
        per_device_batch_size = self.config.get("per_device_train_batch_size", 2)
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 4)
        
        effective_batch_size = per_device_batch_size * gradient_accumulation_steps * self.world_size
        
        # Auto-detect: if FSDP is configured, use FSDP; otherwise use DDP
        fsdp_setting = self.config.get("fsdp", "")
        use_fsdp = bool(fsdp_setting.strip())  # Check if FSDP is actually configured
        
        if self.rank == 0:
            logger.info(f"Effective batch size: {effective_batch_size}")
            logger.info(f"Multi-GPU training: {'enabled' if self.is_distributed else 'disabled'}")
            logger.info(f"Using {'FSDP' if use_fsdp else 'DDP'} for distributed training")
        
        # Base configuration common to both DDP and FSDP
        base_config = {
            "output_dir": self.config.get("output_dir", "./qwen2.5-7b-sft"),
            "per_device_train_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": self.config.get("learning_rate", 2e-5),
            "num_train_epochs": self.config.get("num_train_epochs", 3),
            "max_steps": self.config.get("max_steps", -1),
            "warmup_steps": self.config.get("warmup_steps", 100),
            "logging_steps": self.config.get("logging_steps", 10),
            "save_steps": self.config.get("save_steps", 500),
            "save_strategy": self.config.get("save_strategy", "steps"),
            "bf16": self.config.get("bf16", True),
            "dataloader_pin_memory": self.config.get("dataloader_pin_memory", True),
            "remove_unused_columns": False,
            "report_to": self.config.get("report_to", "wandb" if self.config.get("use_wandb", False) else "none"),
            "run_name": self.config.get("run_name", "qwen2.5-7b-sft"),
            "seed": self.config.get("seed", 42),
            "data_seed": self.config.get("data_seed", 42),
            # Optimizer settings
            "optim": self.config.get("optim", "adamw_torch_fused"),
            "weight_decay": self.config.get("weight_decay", 0.01),
            "max_grad_norm": self.config.get("max_grad_norm", 1.0),
            "lr_scheduler_type": self.config.get("lr_scheduler_type", "cosine"),
            "warmup_ratio": self.config.get("warmup_ratio", 0.1),
            # SFT-specific parameters
            "max_seq_length": self.config.get("max_seq_length", 2048),
            "dataset_text_field": "text",
            "packing": self.config.get("packing", False),
        }
        
        if use_fsdp:
            # FSDP configuration
            fsdp_config = self.config.get("fsdp_config", {})
            return SFTConfig(
                **base_config,
                # FSDP settings
                fsdp=fsdp_setting,
                fsdp_config=fsdp_config,
                # Additional FSDP-specific settings
                gradient_checkpointing=self.config.get("gradient_checkpointing", True),  # Recommended for FSDP
                local_rank=self.local_rank if self.is_distributed else -1,
            )
        else:
            # DDP configuration
            return SFTConfig(
                **base_config,
                # DDP settings
                local_rank=self.local_rank if self.is_distributed else -1,
                ddp_backend="nccl" if self.is_distributed else None,
                ddp_find_unused_parameters=self.config.get("ddp_find_unused_parameters", False),
                ddp_timeout=self.config.get("ddp_timeout", 1800),
                gradient_checkpointing=self.config.get("gradient_checkpointing", False),  # Optional for DDP
            )
        
    def train(self):
        """Main training function with fixed distributed setup"""
        if self.rank == 0:
            logger.info("Starting Fixed SFT training pipeline")
            
            if self.config.get("use_wandb", False):
                wandb.init(
                    project=self.config.get("wandb_project", "qwen2.5-sft"),
                    name=self.config.get("run_name", "qwen2.5-7b-sft"),
                    config=self.config
                )
        
        try:
            # Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Setup LoRA
            self.setup_lora()
            
            # Load and format dataset
            dataset = self.load_dataset()
            formatted_dataset = self.format_dataset(dataset)
            
            # Create training arguments
            training_args = self.create_training_args()
            
            # Create trainer with correct API
            self.trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=formatted_dataset,
                processing_class=self.tokenizer,
            )
            
            if self.rank == 0:
                logger.info("Starting training...")
            
            # Start training
            self.trainer.train()
            
            # Save model
            if self.rank == 0:
                logger.info("Saving model...")
                self.trainer.save_model()
                self.tokenizer.save_pretrained(self.config.get("output_dir"))
                logger.info("Training completed successfully!")
            
        except Exception as e:
            if self.rank == 0:
                logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            if self.rank == 0 and self.config.get("use_wandb", False):
                wandb.finish()
            
            # Clean up distributed
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed SFT Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    pipeline = SFTTrainerPipeline(config)
    pipeline.train()


if __name__ == "__main__":
    main()