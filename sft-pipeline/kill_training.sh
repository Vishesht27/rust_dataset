#!/bin/bash

echo "Killing stuck training processes..."

# Kill all training-related processes
pkill -f "train.py"
pkill -f "accelerate"
pkill -f "torch.distributed"

# Wait a moment
sleep 2

# Force kill if still running
pkill -9 -f "train.py"
pkill -9 -f "accelerate"
pkill -9 -f "torch.distributed"

echo "Training processes killed."

# Show remaining python processes
echo ""
echo "Remaining Python processes:"
ps aux | grep python | grep -v grep
