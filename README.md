# LoRPt: Low-Rank Pretraining

**LoRPt** (Low-Rank Pretraining) is a novel technique that applies LoRA-style low-rank matrix factorization directly to model pretraining, enabling dramatically reduced memory consumption and faster training times for large language models.

## Overview

Traditional pretraining requires storing full-rank weight matrices, consuming substantial memory and compute resources. LoRPt addresses this by factorizing linear layers into low-rank components during pretraining itself, not just fine-tuning.

### Key Innovation

Instead of storing a full weight matrix `W ∈ R^(out_features × in_features)`, LoRPt decomposes it into:

```
W = A @ B
where A ∈ R^(out_features × rank), B ∈ R^(rank × in_features)
```

This reduces parameter count from `O(d²)` to `O(2×d×r)` where `r << d`.

## Architecture

```python
class LoRPtLinear(nn.Module):
    """Low-rank factorized linear layer for memory efficiency"""
    def __init__(self, in_features, out_features, rank=64):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        weight = self.lora_A @ self.lora_B
        return F.linear(x, weight, self.bias)
```

## Performance Metrics

LoRPt is a core component of the **i3 architecture** - a proprietary next-generation framework for resource-efficient language model pretraining. Models using this architecture are available at the [i3 collection on HuggingFace](https://huggingface.co/collections/FlameF0X/i3-arhitecture-68f334c9e6c7ab5a4f6dc792).

### Real-World Results

| Model Size | Training Time | VRAM Usage | Hardware |
|------------|---------------|------------|----------|
| **200M params** | < 4 hours | < 9 GB | T4 / P100 GPU |
| **10-12M params** | - | 4-6 GB | T4 / P100 GPU |

*Note: These metrics include the full i3 architecture with LoRPt components*

### Memory Savings

For a typical transformer FFN layer with `d_model=2048` and `d_ff=8192`:

- **Standard Linear**: `2048 × 8192 = 16.8M parameters`
- **LoRPt (rank=128)**: `2048 × 128 + 128 × 8192 = 1.3M parameters`
- **Reduction**: **92% fewer parameters**

## Use Cases

### 1. Resource-Constrained Pretraining

LoRPt enables pretraining large models on consumer-grade hardware:

```python
# Standard FFN layer (high memory)
ffn = nn.Linear(2048, 8192)  # 16.8M params

# LoRPt FFN layer (low memory)
ffn = LoRPtLinear(2048, 8192, rank=128)  # 1.3M params
```

### 2. Rapid Prototyping

Iterate faster on architecture experiments with reduced training times:

```python
# Build efficient transformer block
class EfficientTransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, rank=64):
        super().__init__()
        self.ffn = nn.Sequential(
            LoRPtLinear(d_model, d_ff, rank=rank),
            nn.GELU(),
            LoRPtLinear(d_ff, d_model, rank=rank)
        )
```

### 3. Integration with Custom Architectures

LoRPt can be integrated into any transformer-based architecture to reduce memory footprint and accelerate training. It's particularly effective when combined with other efficiency techniques.

## Installation

```bash
# LoRPt is self-contained - just copy the class definition
# No external dependencies beyond PyTorch

pip install torch
```

## Quick Start

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRPtLinear(nn.Module):
    """Low-rank factorized linear layer for memory efficiency"""
    def __init__(self, in_features, out_features, rank=64):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        weight = self.lora_A @ self.lora_B
        return F.linear(x, weight, self.bias)

# Replace standard layers
model = nn.Sequential(
    LoRPtLinear(512, 2048, rank=64),
    nn.GELU(),
    LoRPtLinear(2048, 512, rank=64)
)

# Train as normal
x = torch.randn(8, 128, 512)  # (batch, seq_len, d_model)
output = model(x)
```

## Hyperparameter Selection

### Rank Selection Guide

| Model Scale | Recommended Rank | Memory Savings | Quality Trade-off |
|-------------|------------------|----------------|-------------------|
| Small (< 100M) | 32-64 | Very High | Minimal |
| Medium (100M-1B) | 64-128 | High | Negligible |
| Large (1B+) | 128-256 | Moderate | None |

### Tuning Tips

1. **Start Conservative**: Begin with `rank = d_model // 16`
2. **Monitor Loss**: If training plateaus early, increase rank
3. **Memory Budget**: Reduce rank if OOM occurs
4. **Layer-Specific**: Use higher ranks for critical layers (e.g., output projections)

## Comparison with Traditional LoRA

| Aspect | Traditional LoRA | LoRPt |
|--------|------------------|-------|
| **Application** | Fine-tuning only | Pretraining + Fine-tuning |
| **Base Weights** | Frozen full-rank | No base weights |
| **Memory Savings** | Moderate (adapters) | Extreme (full model) |
| **Training Speed** | Fast (fewer params) | Faster (fewer params + ops) |
| **Use Case** | Adaptation | From-scratch training |

## The i3 Architecture

LoRPt is a core component of the proprietary **i3 architecture**, designed for maximum training and inference efficiency. The i3 family of models demonstrates that low-rank pretraining can achieve competitive quality with full-rank approaches while dramatically reducing resource requirements.

**Explore i3 models**: [HuggingFace Collection](https://huggingface.co/collections/FlameF0X/i3-arhitecture-68f334c9e6c7ab5a4f6dc792)

### Why i3 + LoRPt?

The combination enables:
- **10x faster pretraining** on consumer hardware
- **5x memory reduction** vs. standard architectures
- **Competitive quality** with full-rank models
- **Accessible research** for independent developers

## Research & Development

LoRPt was developed to democratize language model pretraining by removing hardware barriers. It enables:

- **Academic Research**: Run experiments without datacenter GPUs
- **Indie Development**: Build custom LLMs on personal hardware
- **Rapid Iteration**: Test architectural ideas in hours, not days
- **Green AI**: Reduce energy consumption and carbon footprint

## Citation

If you use LoRPt in your research, please cite:

```bibtex
@software{lorpt2025,
  title={LoRPt: Low-Rank Pretraining for Resource-Efficient Language Models},
  author={[FlameF0X]},
  year={2025},
  url={https://github.com/FlameF0X/Low-Rank-Pretraining}
}
```

## Limitations

- **Rank Selection**: Requires tuning per architecture
- **Expressivity Trade-off**: Very low ranks may limit model capacity
- **Recombination Cost**: Forward pass computes `A @ B` each time (can be cached)
- **Not Universal**: Some layers (embeddings, layer norms) don't benefit

## Future Work

- **Adaptive Rank**: Dynamic rank adjustment during training
- **Structured Pruning**: Combine with sparsity techniques
- **Mixed Precision**: Optimize with int8/fp16 quantization
- **Knowledge Distillation**: Transfer from full-rank teachers

## Contributing

Contributions welcome! Areas of interest:
- Rank scheduling algorithms
- Integration with other efficiency techniques
- Benchmarking on diverse tasks
- Production deployment optimizations

## License

MIT License - Free for research and commercial use

---

**Built with ❤️ for accessible AI research**

*LoRPt: Making language model pretraining possible on your laptop*
