#!/usr/bin/env python3
"""
Setup script for SFT Pipeline with Multi-GPU Support
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def check_gpu_availability():
    """Check GPU availability and configuration"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ CUDA available with {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  - GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("⚠️  CUDA not available - training will be very slow")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False


def setup_accelerate():
    """Setup accelerate configuration for multi-GPU training"""
    print("\n🚀 Setting up Accelerate for multi-GPU training...")
    
    # Check if accelerate config exists
    config_path = Path.home() / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"
    
    if not config_path.exists():
        print("Setting up accelerate configuration...")
        try:
            result = subprocess.run(["accelerate", "config"], input="\n\n\n\n\n\n\n\n", 
                                  text=True, capture_output=True, timeout=30)
            print("✓ Accelerate configuration completed")
        except subprocess.TimeoutExpired:
            print("⚠️  Accelerate config timed out - you may need to run 'accelerate config' manually")
        except FileNotFoundError:
            print("⚠️  Accelerate not found - install with: pip install accelerate")
    else:
        print("✓ Accelerate configuration already exists")


def main():
    """Main setup function"""
    print("🚀 Setting up SFT Pipeline for Qwen2.5 32B (Multi-GPU Optimized)")
    print("=" * 70)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    directories = ["data", "outputs", "logs", "checkpoints", "configs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ Created {directory}/")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("\n❌ Failed to install dependencies. Please check your Python environment.")
        sys.exit(1)
    
    # Check GPU availability
    print("\n🔍 Checking GPU availability...")
    gpu_available = check_gpu_availability()
    
    # Setup accelerate for multi-GPU training
    if gpu_available:
        setup_accelerate()
    
    # Create sample dataset
    print("\n📊 Creating sample dataset...")
    try:
        from data_utils import create_sample_dataset
        create_sample_dataset("data/sample_dataset.jsonl", 50)
        print("✓ Sample dataset created at data/sample_dataset.jsonl")
    except Exception as e:
        print(f"⚠️  Could not create sample dataset: {e}")
    
    # Validate installation
    print("\n🔍 Validating installation...")
    try:
        import torch
        import transformers
        import datasets
        import trl
        import peft
        import accelerate
        print("✓ All required packages imported successfully")
        
        # Test multi-GPU setup
        if gpu_available and torch.cuda.device_count() > 1:
            print(f"✓ Multi-GPU setup detected: {torch.cuda.device_count()} GPUs")
            print("  You can use: accelerate launch --multi_gpu train.py --config configs/training_config.json")
        elif gpu_available:
            print("✓ Single GPU setup detected")
            print("  You can use: python train.py --config configs/training_config.json")
        else:
            print("⚠️  No GPU detected - training will be very slow")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Prepare your dataset in JSONL format")
    print("2. Update configs/training_config.json with your settings")
    
    if gpu_available and torch.cuda.device_count() > 1:
        print("3. For multi-GPU training:")
        print("   accelerate launch --multi_gpu train.py --config configs/training_config.json")
    else:
        print("3. For single GPU training:")
        print("   python train.py --config configs/training_config.json")
    
    print("\nFor testing with sample data:")
    print("python train.py --config configs/training_config.json --create_sample")
    
    print("\nFor system information:")
    print("python train.py --system_info")


if __name__ == "__main__":
    main()
