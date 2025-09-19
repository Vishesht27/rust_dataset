#!/bin/bash

# Multi-GPU training script for SFT pipeline
# This script ensures proper multi-GPU setup using accelerate

echo "=== Multi-GPU SFT Training Script ==="
echo "Checking GPU availability..."

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

echo ""
echo "Starting multi-GPU training..."
echo "Using accelerate config: accelerate_config.yaml"
echo ""

# Run training with accelerate
accelerate launch \
    --config_file accelerate_config.yaml \
    train.py \
    --config configs/training_config.json

echo ""
echo "Training completed!"
