#!/bin/bash

# 2-GPU training script for SFT pipeline (for testing)
# This script uses only 2 GPUs to test multi-GPU functionality

echo "=== 2-GPU SFT Training Script ==="
echo "Checking GPU availability..."

# Check if CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

echo ""
echo "Setting up NCCL environment variables for 2-GPU training..."

# Set NCCL environment variables for better communication
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

# Set additional environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

echo ""
echo "Starting 2-GPU training..."
echo "Using accelerate config: configs/accelerate_config_2gpu.yaml"
echo ""

# Run training with accelerate
accelerate launch \
    --config_file configs/accelerate_config_2gpu.yaml \
    train.py \
    --config configs/training_config.json

echo ""
echo "Training completed!"
