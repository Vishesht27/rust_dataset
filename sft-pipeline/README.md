# SFT Pipeline for Qwen2.5 32B - Multi-GPU Optimized

A high-performance Supervised Fine-Tuning (SFT) pipeline for the Qwen2.5 32B model, inspired by the [foundation-model-stack/fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning) implementation. This pipeline supports multi-GPU training with FSDP and is optimized for maximum performance without GPU memory constraints.

## 🚀 Key Features

- **Multi-GPU Support**: Full support for distributed training across multiple GPUs
- **FSDP Integration**: Fully Sharded Data Parallel for efficient large model training
- **No Memory Constraints**: Optimized for maximum performance without quantization
- **High-Performance LoRA**: Higher rank LoRA (r=64) for better model quality
- **Advanced Optimizations**: Gradient checkpointing, mixed precision, and efficient data loading
- **Flexible Data Formats**: Support for JSON, JSONL, and CSV datasets
- **Comprehensive Monitoring**: Weights & Biases integration and detailed logging

## 📋 Requirements

### Hardware Requirements
- **Minimum**: 2x 24GB GPUs (RTX 4090, A100 40GB)
- **Recommended**: 4x 40GB+ GPUs (A100 80GB, H100)
- **System RAM**: 64GB+ recommended
- **Storage**: 500GB+ SSD storage

### Software Requirements
- Python 3.8+
- CUDA 11.8+ or 12.1+
- PyTorch 2.1+

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd sft-pipeline
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Setup the environment:**
```bash
python setup.py
```

## 🚀 Quick Start

### 1. Single GPU Training
```bash
python train.py --config configs/training_config.json
```

### 2. Multi-GPU Training (Recommended)
```bash
# Using accelerate (recommended)
accelerate launch --multi_gpu train.py --config configs/training_config.json

# Or using torchrun
torchrun --nproc_per_node=4 train.py --config configs/training_config.json
```

### 3. Create and Test with Sample Data
```bash
python train.py --config configs/training_config.json --create_sample
```

### 4. Validate Your Dataset
```bash
python train.py --config configs/training_config.json --validate_only
```

## 📊 Dataset Format

Your dataset should be in JSONL format with the following structure:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

### Alternative Formats Supported:
- **CSV**: With `instruction` and `response` columns
- **JSON**: With `prompt` and `completion` fields

## ⚙️ Configuration

The pipeline uses a single, comprehensive configuration file (`configs/training_config.json`):

### Key Configuration Sections:

#### Training Parameters:
```json
{
  "training": {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "max_seq_length": 4096,
    "packing": true
  }
}
```

#### Multi-GPU Settings:
```json
{
  "multi_gpu": {
    "fsdp": "full_shard auto_wrap",
    "fsdp_config": {
      "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
      "fsdp_sharding_strategy": "FULL_SHARD",
      "fsdp_cpu_ram_efficient_loading": true
    }
  }
}
```

#### LoRA Configuration:
```json
{
  "optimization": {
    "use_lora": true,
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.1
  }
}
```

## 🔧 Advanced Usage

### Multi-GPU Training with Custom Configuration

```bash
# 4-GPU training with custom settings
accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=bf16 \
  train.py \
  --config configs/training_config.json \
  --output_dir ./outputs/my-experiment
```

### Enable Weights & Biases Monitoring

Update your config:
```json
{
  "wandb": {
    "use_wandb": true,
    "wandb_project": "qwen2.5-sft",
    "run_name": "my-experiment"
  }
}
```

### Custom Dataset Processing

```python
from data_utils import DatasetProcessor, DatasetValidator
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
processor = DatasetProcessor(tokenizer)

# Load and process your dataset
dataset = processor.load_dataset_from_file("your_data.jsonl")
formatted_dataset = processor.format_conversation_data(dataset)

# Validate dataset
validator = DatasetValidator()
stats = validator.validate_dataset(dataset)
print(stats)
```

## 📈 Performance Optimizations

### Memory Efficiency:
- **FSDP**: Fully sharded data parallel for large models
- **Gradient Checkpointing**: Trades compute for memory
- **Mixed Precision**: BF16 for better numerical stability
- **Efficient Data Loading**: Pinned memory and optimized data loaders

### Training Efficiency:
- **Higher LoRA Rank**: r=64 for better model quality
- **Packing**: Efficient sequence packing for better GPU utilization
- **Optimized Optimizer**: AdamW with fused operations
- **Cosine Learning Rate**: Better convergence

### Multi-GPU Optimizations:
- **NCCL Backend**: Fast GPU-to-GPU communication
- **FSDP Auto-Wrap**: Automatic model sharding
- **Gradient Synchronization**: Efficient gradient aggregation

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable `fsdp_cpu_ram_efficient_loading`

2. **Multi-GPU Issues**:
   - Ensure all GPUs are visible: `nvidia-smi`
   - Check NCCL installation: `python -c "import torch; print(torch.distributed.is_nccl_available())"`
   - Use `accelerate config` to configure your setup

3. **Dataset Format Errors**:
   - Use `--validate_only` to check your dataset
   - Ensure proper JSON structure
   - Check for empty or malformed examples

### Performance Tips:

1. **Use SSD Storage**: Faster data loading
2. **Enable Mixed Precision**: Use `bf16: true` for modern GPUs
3. **Optimize Batch Size**: Balance memory usage and training speed
4. **Monitor GPU Utilization**: Ensure high GPU usage across all devices

## 📊 Monitoring and Logging

### Weights & Biases Integration:
```bash
# Enable in config
{
  "wandb": {
    "use_wandb": true,
    "wandb_project": "qwen2.5-sft"
  }
}
```

### Local Logging:
- Training logs: `outputs/your-model/logs/`
- Model checkpoints: `outputs/your-model/checkpoints/`
- TensorBoard logs: `outputs/your-model/runs/`

## 🔄 Comparison with Foundation Model Stack

This implementation is inspired by the [foundation-model-stack/fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning) project and includes:

- **Similar Architecture**: Based on their proven SFT trainer implementation
- **Enhanced Multi-GPU Support**: Improved FSDP configuration
- **No Memory Constraints**: Removed quantization for maximum performance
- **Higher LoRA Ranks**: Better model quality with r=64
- **Advanced Optimizations**: Latest training techniques and optimizations

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Foundation Model Stack](https://github.com/foundation-model-stack/fms-hf-tuning) for the excellent reference implementation
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the core framework
- [TRL Library](https://github.com/huggingface/trl) for SFT training utilities
- [Qwen Team](https://github.com/QwenLM/Qwen) for the excellent model

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the configuration options
3. Open an issue with detailed error logs and system information
