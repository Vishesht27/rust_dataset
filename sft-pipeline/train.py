#!/usr/bin/env python3
"""
Main training script for SFT pipeline with multi-GPU support
Usage: 
  Single GPU: python train.py --config configs/training_config.json
  Multi-GPU:  accelerate launch --multi_gpu train.py --config configs/training_config.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from sft_trainer import SFTTrainerPipeline, load_config
from data_utils import DatasetProcessor, DatasetValidator, create_sample_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_config(config: dict) -> bool:
    """Validate configuration file"""
    required_fields = ["model_name", "dataset_path", "output_dir"]
    
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required field in config: {field}")
            return False
            
    # Validate dataset path
    dataset_path = config["dataset_path"]
    if not Path(dataset_path).exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        return False
        
    return True


def setup_directories(config: dict):
    """Setup necessary directories"""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")


def print_system_info():
    """Print system information for debugging"""
    import torch
    
    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    logger.info("=========================")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="SFT Training Pipeline for Qwen2.5 32B")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--dataset", type=str, help="Path to dataset file (overrides config)")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--model_name", type=str, help="Model name (overrides config)")
    parser.add_argument("--create_sample", action="store_true", help="Create sample dataset for testing")
    parser.add_argument("--validate_only", action="store_true", help="Only validate dataset, don't train")
    parser.add_argument("--dry_run", action="store_true", help="Dry run - validate config and dataset only")
    parser.add_argument("--system_info", action="store_true", help="Print system information and exit")
    
    args = parser.parse_args()
    
    # Print system info if requested
    if args.system_info:
        print_system_info()
        return
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.dataset:
            config["dataset_path"] = args.dataset
        if args.output_dir:
            config["output_dir"] = args.output_dir
        if args.model_name:
            config["model_name"] = args.model_name
            
        # Create sample dataset if requested
        if args.create_sample:
            sample_path = "data/sample_dataset.jsonl"
            create_sample_dataset(sample_path, 100)
            config["dataset_path"] = sample_path
            logger.info(f"Created sample dataset at: {sample_path}")
            
        # Validate configuration
        if not validate_config(config):
            logger.error("Configuration validation failed")
            sys.exit(1)
            
        # Setup directories
        setup_directories(config)
        
        # Print system information
        print_system_info()
        
        # Validate dataset if requested
        if args.validate_only or args.dry_run:
            logger.info("Validating dataset...")
            
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
            
            processor = DatasetProcessor(tokenizer)
            dataset = processor.load_dataset_from_file(config["dataset_path"])
            
            validator = DatasetValidator()
            stats = validator.validate_dataset(dataset)
            
            logger.info("Dataset validation results:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
                
            if args.validate_only:
                return
                
        # Dry run - just validate everything
        if args.dry_run:
            logger.info("Dry run completed successfully!")
            return
            
        # Create and run training pipeline
        logger.info("Starting SFT training pipeline...")
        pipeline = SFTTrainerPipeline(config)
        pipeline.train()
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
