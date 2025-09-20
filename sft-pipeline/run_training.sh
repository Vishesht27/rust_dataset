#!/bin/bash

# Multi-GPU SFT Training Script
# Optimized for cloud environments with proper NCCL configuration

echo "=== Multi-GPU SFT Training Script ==="
echo "Checking GPU availability..."

# Check if CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

echo ""
echo "Setting up environment variables for multi-GPU training..."

# Critical NCCL environment variables to prevent hanging
export NCCL_DEBUG=WARN  # Reduce verbosity to prevent log spam
export NCCL_SOCKET_IFNAME=^docker0,lo  # Exclude docker and loopback interfaces
export NCCL_IB_DISABLE=1  # Disable InfiniBand
export NCCL_P2P_DISABLE=1  # Disable P2P for cloud environments
export NCCL_TREE_THRESHOLD=0  # Force ring algorithm
export NCCL_ALGO=Ring  # Use ring algorithm explicitly
export NCCL_TIMEOUT=1800  # 30 minute timeout
export NCCL_BLOCKING_WAIT=1  # Enable blocking wait

# Additional environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export OMP_NUM_THREADS=1  # Prevent thread oversubscription
export TOKENIZERS_PARALLELISM=false  # Prevent tokenizer warnings

# Auto-detect GPU count and set CUDA_VISIBLE_DEVICES
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [ "$GPU_COUNT" -gt 0 ]; then
    # Create comma-separated list of GPU IDs
    GPU_IDS=$(python3 -c "import torch; print(','.join(str(i) for i in range(torch.cuda.device_count())))")
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
else
    echo "No GPUs detected!"
    exit 1
fi

echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "GPU_COUNT: $GPU_COUNT"

echo ""
echo "Starting multi-GPU training with $GPU_COUNT GPUs..."
echo "Using accelerate config: configs/accelerate_config.yaml"
echo ""

# Run training with accelerate
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --main_process_port $MASTER_PORT \
    train.py \
    --config configs/training_config.json

echo ""
echo "Training completed!"
