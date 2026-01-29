# Lyra

**Efficient Subquadratic Architecture for Biological Sequence Modeling**

Implementation of Lyra from ["Lyra: An Efficient and Expressive Subquadratic Architecture for Modeling Biological Sequences"](https://www.biorxiv.org/content/10.1101/2024.XX.XX.XXXXXX) by Ramesh, Siddiqui, Gu, Mitzenmacher & Sabeti.

## Key Ideas

Lyra achieves state-of-the-art performance on 100+ biological tasks with **orders of magnitude fewer parameters** than foundation models (up to 120,000× reduction) by aligning architecture with biology:

1. **Epistasis as Polynomials**: Biological sequence-to-function relationships can be represented as multilinear polynomials, where coefficients capture how positions interact (epistatic effects).

2. **SSMs as Polynomial Approximators**: State Space Models (specifically S4D) naturally approximate polynomials through their hidden state dynamics, making them ideal for capturing epistatic interactions.

3. **Hybrid Architecture**:
   - **PGC (Projected Gated Convolutions)**: Local feature extraction with multiplicative gating that explicitly encodes 2nd-order interactions
   - **S4D layers**: Global dependencies via FFT-based convolution, scaling O(N log N) vs O(N²) for attention

## Installation

```bash
pip install lyra-bio

# Or from source:
git clone https://github.com/dryingpaint/lyra.git
cd lyra
pip install -e .
```

## Quick Start

```python
import torch
from lyra import Lyra

# Create model for protein fitness prediction
model = Lyra(
    d_input=20,      # 20 amino acids (one-hot)
    d_model=64,      # Internal dimension
    d_output=1,      # Regression output
    num_pgc_layers=2,
    num_s4_layers=4,
    d_state=64,      # Controls polynomial degree
)

# Input: batch of protein sequences (one-hot encoded)
x = torch.randn(32, 100, 20)  # (batch, seq_len, features)

# Forward pass
y = model(x)  # (32, 1)

print(f"Parameters: {model.count_parameters():,}")
# Parameters: ~50,000 (vs billions for foundation models)
```

## Model Variants

### Sequence-level prediction (default)
For tasks like fitness prediction, binding affinity, etc:

```python
from lyra import Lyra

model = Lyra(
    d_input=20,
    d_model=64,
    d_output=1,
    pool='mean',  # 'mean', 'max', or 'first'
)
```

### Token-level prediction
For tasks like disorder prediction, secondary structure, etc:

```python
from lyra import LyraForTokenClassification

model = LyraForTokenClassification(
    d_input=20,
    d_model=64,
    d_output=3,  # e.g., 3 classes per position
)

x = torch.randn(32, 100, 20)
y = model(x)  # (32, 100, 3) - prediction per position
```

## Architecture Details

```
Input (B, L, d_input)
    │
    ▼
┌─────────────────────────┐
│   Linear Encoder        │  → (B, L, d_model)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   PGC Layers (×N)       │  Local epistasis via gated conv
│   ├─ Depthwise Conv     │  
│   ├─ Linear Projection  │
│   └─ Hadamard Gate      │  ← 2nd-order interactions
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   S4D Layers (×M)       │  Global epistasis via SSM
│   ├─ FFT Convolution    │  O(N log N) complexity
│   ├─ Skip Connection    │
│   └─ GLU Output         │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Pooling + Decoder     │  → (B, d_output)
└─────────────────────────┘
```

## Why It Works

### The Math
Epistatic interactions in proteins can be written as:

$$f(u) = \sum_{k=1}^{K} \sum_{i_1 < \cdots < i_k} c_{i_1 \cdots i_k} \prod_{j=1}^{k} u_{i_j}$$

This is a multilinear polynomial. The S4D hidden state evolution:

$$x_{t+1} = Ax_t + Bu_t$$

with diagonal A creates a Vandermonde structure that naturally approximates such polynomials. The hidden dimension controls the polynomial degree (epistatic order) that can be captured.

### Key Results from the Paper
- **Disorder prediction**: 0.931 accuracy with 56K params vs ProtT5's 0.855 with 3B params
- **Protein fitness**: SOTA on DMS benchmarks
- **Inference speed**: 64× faster than transformers on average
- **Training**: Most tasks trainable in <2 hours on 2 GPUs

## Configuration Guide

| Task Type | Recommended Config |
|-----------|-------------------|
| Short proteins (<200 aa) | `d_model=64, d_state=64, num_s4=4` |
| Long proteins (>500 aa) | `d_model=128, d_state=128, num_s4=6` |
| Nucleotide sequences | `d_input=4, d_model=64` |
| High epistasis tasks | Increase `d_state` (more polynomial terms) |

## Citation

```bibtex
@article{ramesh2024lyra,
  title={Lyra: An Efficient and Expressive Subquadratic Architecture for Modeling Biological Sequences},
  author={Ramesh, Krithik and Siddiqui, Sameed M and Gu, Albert and Mitzenmacher, Michael D and Sabeti, Pardis C},
  journal={bioRxiv},
  year={2024}
}
```

## License

MIT

## Acknowledgments

Based on the paper by Ramesh, Siddiqui, Gu, Mitzenmacher & Sabeti. This is an unofficial implementation for research purposes.
