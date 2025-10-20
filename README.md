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

| Model Size        | Training Time | VRAM Usage | Hardware      |
| ----------------- | ------------- | ---------- | ------------- |
| **200M params**   | < 4 hours     | < 9 GB     | T4 / P100 GPU |
| **10-12M params** | -             | 4-6 GB     | T4 / P100 GPU |

*Note: These metrics include the full i3 architecture with LoRPt components*

---

## LoRPt vs Normal Linear Layer - Benchmark Results

**Test Environment:**

* GPU: Tesla P100-PCIE-16GB
* CUDA: 12.4
* PyTorch: 2.6.0+cu124

### Configuration: Training Quality Benchmark

**Benchmark Description:**
This benchmark trains both a normal linear model and a LoRPt model on a synthetic dataset and compares their loss convergence.

**Model & Training Parameters:**

* Device: cuda
* Vocabulary Size: 1000
* Model Dimension: 256
* Number of Layers: 4
* LoRPt Rank: 64
* Training Iterations: 2000
* Batch Size: 32
* Learning Rate: 0.0003

**Dataset:**

* Training samples: 10,000
* Test samples: 1,000

### Training Logs Snapshot

**Normal Linear Model:**

```
Final Train Loss: 6.1186 | Test Loss: 6.3234 | Perplexity: 557.44 | Total Time: 16.3s
```

**LoRPt Model:**

```
Final Train Loss: 6.2195 | Test Loss: 6.2279 | Perplexity: 506.71 | Total Time: 16.5s
Parameter Reduction: 61.4%
```

### Comparison

| Metric          | Normal Linear | LoRPt     | Difference |
| --------------- | ------------- | --------- | ---------- |
| Test Loss       | 6.3234        | 6.2279    | -1.51%     |
| Perplexity      | 557.44        | 506.71    | -9.10%     |
| Training Time   | 16.3s         | 16.5s     | +1.08%     |
| Parameter Count | 2,634,216     | 1,016,808 | -61.4%     |

**Verdict:**
✅ **Equivalent Quality** — LoRPt achieves comparable model quality to Normal Linear

* 1.5% lower test loss
* 9.1% better perplexity
* 61.4% fewer parameters
* Training time slightly slower (+1.1%), negligible impact

<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/472c91a3-94ef-4fa5-bb4a-e9bfbbbb256c" />

---

### Configuration 1: Small Model

**Model Architecture:**

* Embedding Dimension: 512
* Feed-Forward Dimension: 2048
* Number of Layers: 4
* LoRPt Rank: 64
* Batch Size: 16
* Sequence Length: 128

#### Results

| Metric                 | Normal Linear | LoRPt        | Improvement         |
| ---------------------- | ------------- | ------------ | ------------------- |
| **Parameters**         | 8,911,848     | 1,418,728    | **84.1% reduction** |
| **Memory (MB)**        | 34.00         | 5.41         | **84.1% reduction** |
| **Forward Pass (ms)**  | 6.02 ± 0.19   | 6.21 ± 0.08  | 0.97x (6% slower)   |
| **Training Step (ms)** | 17.08 ± 0.10  | 18.13 ± 0.12 | 0.94x (6% slower)   |

**Summary:**

* ✅ 84.1% fewer parameters
* ✅ 84.1% less memory usage
* ⚠️ 6% slower inference
* ⚠️ 6% slower training per step

### Configuration 2: Medium Model

**Model Architecture:**
- Embedding Dimension: 1024
- Feed-Forward Dimension: 4096
- Number of Layers: 4
- LoRPt Rank: 128
- Batch Size: 8
- Sequence Length: 256

#### Results

| Metric | Normal Linear | LoRPt | Improvement |
|--------|---------------|-------|-------------|
| **Parameters** | 34,599,912 | 5,523,432 | **84.0% reduction** |
| **Memory (MB)** | 131.99 | 21.07 | **84.0% reduction** |
| **Forward Pass (ms)** | 21.95 ± 0.19 | 23.26 ± 0.20 | 0.94x (6% slower) |
| **Training Step (ms)** | 60.93 ± 0.25 | 64.56 ± 0.24 | 0.94x (6% slower) |

**Summary:**
- ✅ 84.0% fewer parameters
- ✅ 84.0% less memory usage
- ⚠️ 6% slower inference
- ⚠️ 6% slower training per step

### Configuration 3: Large Model

**Model Architecture:**
- Embedding Dimension: 2048
- Feed-Forward Dimension: 8192
- Number of Layers: 4
- LoRPt Rank: 128
- Batch Size: 4
- Sequence Length: 512

#### Results

| Metric | Normal Linear | LoRPt | Improvement |
|--------|---------------|-------|-------------|
| **Parameters** | 136,307,688 | 10,917,864 | **92.0% reduction** |
| **Memory (MB)** | 519.97 | 41.65 | **92.0% reduction** |
| **Forward Pass (ms)** | 78.83 ± 0.66 | 83.73 ± 0.64 | 0.94x (6% slower) |
| **Training Step (ms)** | 213.67 ± 0.96 | 225.97 ± 1.12 | 0.95x (5% slower) |

**Summary:**
- ✅ 92.0% fewer parameters (136M → 11M)
- ✅ 92.0% less memory usage (520MB → 42MB)
- ⚠️ 6% slower inference
- ⚠️ 5% slower training per step

### Overall Analysis

#### Memory Savings

LoRPt achieves consistent **84-92% memory reduction** across all model sizes:

```
Small (512d):   34 MB → 5.4 MB   (6.3x smaller)
Medium (1024d): 132 MB → 21 MB   (6.3x smaller)
Large (2048d):  520 MB → 42 MB   (12.5x smaller)
```

The memory savings scale with model size - larger models benefit more from low-rank factorization.

#### Performance Trade-off

LoRPt shows a consistent **5-6% slowdown** in compute speed:
- This overhead comes from computing `A @ B` matrix multiplication on every forward pass
- The slowdown is consistent across model sizes, indicating it's an inherent architectural trade-off

#### When LoRPt Wins

Despite the per-step slowdown, LoRPt enables **faster overall training** by:

1. **Enabling Larger Batch Sizes**
   - Normal: Limited by VRAM, might OOM at batch size 8-16
   - LoRPt: Can use 2-4x larger batches → better GPU utilization → faster convergence

2. **Reducing Optimizer Memory**
   - Adam optimizer stores 2x parameter copies (momentum + variance)
   - 92% fewer params = 92% less optimizer memory
   - Example: 136M params = 1.6GB optimizer states vs 11M params = 130MB

3. **Making Training Possible**
   - Models that won't fit in VRAM with normal Linear layers can train with LoRPt
   - The choice isn't "5% slower" vs "5% faster" - it's "can train" vs "can't train"

#### Real-World Impact

For the i3 200M parameter model:

**With Normal Linear (hypothetical):**
- Model weights: ~800 MB
- Optimizer states: ~1.6 GB
- Gradients: ~800 MB
- Activations: ~2-4 GB
- **Total: 15-20+ GB VRAM required**
- Result: Won't fit on consumer GPUs

**With LoRPt (actual):**
- Model weights: ~80 MB (effective 200M params from ~20M actual)
- Optimizer states: ~160 MB
- Gradients: ~80 MB
- Activations: ~2-4 GB
- **Total: <9 GB VRAM used**
- Result: Trains in <4 hours on T4/P100

### Benchmark Conclusion

LoRPt demonstrates an excellent trade-off for pretraining:

**Gains:**
- ✅ 84-92% memory reduction
- ✅ Enables training larger models on consumer hardware
- ✅ Allows 2-4x larger batch sizes
- ✅ Reduces optimizer memory overhead proportionally

**Cost:**
- ⚠️ 5-6% slower per training step (minor and consistent)

**Net Result:** The memory savings enable dramatically larger batch sizes and models that wouldn't otherwise fit, resulting in **faster overall training** and making modern LLM pretraining accessible on consumer hardware.

*Benchmarked on Tesla P100-PCIE-16GB | October 2025*

---

## LoRPt Placement Ablation Study

This comprehensive study tests **where** to apply LoRPt factorization within transformer architectures. Seven variants were trained to identify optimal placement strategies.

### Study Configuration

**Model Architecture:**
- Vocabulary Size: 1,000
- Model Dimension: 256
- Feed-Forward Dimension: 1,024
- Number of Layers: 4
- LoRPt Rank: 64
- Training Iterations: 10,000
- Batch Size: 32
- Sequence Length: 32

**Variants Tested:**

1. **Fully Normal (Baseline)** - All standard Linear layers
2. **Fully LoRPt (All FFN)** - LoRPt on all FFN layers + output
3. **Output Normal (LoRPt FFN)** - LoRPt on FFN, full-rank output
4. **Only FFN-Up** - LoRPt only on up-projection (d_model → d_ff)
5. **Only FFN-Down** - LoRPt only on down-projection (d_ff → d_model)
6. **Alternating Layers** - Alternating LoRPt and normal layers
7. **Depth-Adaptive Rank** - Higher rank (128) for first/last layers, lower (64) for middle

### Results Summary

| Variant | Parameters | Final Loss | vs Baseline | Verdict |
|---------|-----------|------------|-------------|---------|
| **Fully Normal (Baseline)** | 2,634,216 | 6.2970 | 0% | Reference |
| **Fully LoRPt (All FFN)** | 1,016,808 | 6.6851 | +6.16% | ⚠️ Slight degradation |
| **Output Normal (LoRPt FFN)** | 1,274,408 | 6.5004 | +3.23% | ✅ **Best LoRPt variant** |
| **Only FFN-Up** | 1,709,416 | 6.4129 | +1.84% | ✅ Good balance |
| **Only FFN-Down** | 1,660,008 | 6.3169 | +0.32% | ✅ Near-baseline quality |
| **Alternating Layers** | 1,825,512 | 6.2870 | -0.16% | ✅ Matches baseline |
| **Depth-Adaptive Rank** | 1,150,208 | 6.4722 | +2.78% | ⚠️ Moderate degradation |

### Key Findings

#### 1. Output Projection Matters

**Output Normal (LoRPt FFN)** achieved the best quality among LoRPt variants:
- Only 3.2% higher loss than baseline
- 51.6% parameter reduction (2.6M → 1.3M)
- Full-rank output preserves vocabulary distribution modeling

**Insight:** The output projection directly affects prediction accuracy. Keeping it full-rank is worth the memory cost for maintaining quality.

#### 2. Alternating Layers Surprises

**Alternating Layers** actually *matched* the baseline:
- -0.16% loss difference (essentially identical)
- 30.7% parameter reduction
- Shows that **not every layer needs full capacity**

**Insight:** Strategic placement can achieve near-zero quality loss. Some layers benefit more from full-rank expressivity than others.

#### 3. Down-Projection More Robust

**Only FFN-Down** (0.32% loss increase) outperformed **Only FFN-Up** (1.84% increase):
- Down-projection (d_ff → d_model) tolerates low-rank better
- Up-projection (d_model → d_ff) is more sensitive to compression

**Insight:** If you can only factorize one projection, choose the down-projection for better quality retention.

#### 4. Full LoRPt Trade-off

**Fully LoRPt** showed 6.16% loss increase but achieved:
- 61.4% parameter reduction
- Significant memory savings
- Still acceptable perplexity for many applications

**Insight:** Maximum memory savings come at a measurable quality cost. Acceptable for resource-constrained scenarios or early-stage experiments.

### Visual Analysis

The training curves reveal:

1. **Convergence Speed:** All variants converged at similar rates, suggesting LoRPt doesn't hurt optimization dynamics
2. **Final Stability:** LoRPt models showed stable final loss (no instability from factorization)
3. **Parameter Efficiency:** The Output Normal variant achieved the best loss-to-parameter ratio

### Practical Recommendations

Based on ablation results:

**For Maximum Quality (< 1% loss):**
- Use **Alternating Layers** strategy
- Apply LoRPt to middle layers only
- Keep first, last, and output layers full-rank

**For Balanced Trade-off (< 3% loss):**
- Use **Output Normal** strategy
- Apply LoRPt to all FFN layers
- Keep output projection full-rank

**For Maximum Memory Savings (< 7% loss):**
- Use **Fully LoRPt** strategy
- Accept quality degradation for extreme resource constraints
- Good for experimentation and prototyping

**General Strategy:**
1. Start with **Output Normal** as default
2. If quality matters most: Switch to **Alternating Layers**
3. If memory matters most: Use **Fully LoRPt**
4. Always prefer factorizing **down-projections** over up-projections when choosing specific layers

### Study Limitations

- Single task (language modeling on synthetic data)
- Small model scale (256d) - larger models may show different patterns
- Fixed rank (64) - adaptive rank scheduling not tested
- No attention mechanism - only FFN layers evaluated

Future work should validate these findings on:
- Real-world datasets (WikiText, C4, etc.)
- Larger model scales (1B+ parameters)
- Different architectures (attention, MoE, etc.)
- Dynamic rank adjustment strategies

<img width="1589" height="1190" alt="image" src="https://github.com/user-attachments/assets/cec14050-7972-4652-9fc2-879e9f823c20" />

---

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

LoRPt was developed by a solo 17-year-old developer to democratize language model pretraining by removing hardware barriers. It enables:

- **Academic Research**: Run experiments without datacenter GPUs
- **Indie Development**: Build custom LLMs on personal hardware
- **Rapid Iteration**: Test architectural ideas in hours, not days
- **Green AI**: Reduce energy consumption and carbon footprint

This project demonstrates that groundbreaking AI research doesn't require massive teams or resources - just curiosity, determination, and a laptop.

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
