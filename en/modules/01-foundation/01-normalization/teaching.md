---
title: Normalization Teaching Notes | MiniMind LLM Training
description: Understand why normalization is needed, how RMSNorm works, and why Pre-LN is preferred over Post-LN in modern LLMs.
keywords: Normalization, RMSNorm, LayerNorm, Pre-LN, Post-LN, gradient vanishing, Transformer
---

# Normalization Teaching Notes

> Understand why deep networks need normalization and how RMSNorm stabilizes training.

---

## 1. Why normalization?

### Problem: deep networks are hard to train

When stacking many layers, activations can shrink rapidly. A simple example with 8 layers:

**Problem 1: Vanishing activations**
```
Layer 1: activation std = 1.04
Layer 2: activation std = 0.85
Layer 3: activation std = 0.62
Layer 4: activation std = 0.38
Layer 5: activation std = 0.21
...
Layer 8: activation std = 0.016  ← almost 0
```

As the signal shrinks, gradients vanish and the model stops learning.

**Problem 2: Exploding activations**

In other cases, activations grow too large, leading to `NaN` and unstable training.

---

### Intuition: normalization as a stabilizer

**Key idea**: normalization keeps the scale of activations stable across layers.

- Without normalization: each layer amplifies or shrinks the signal unpredictably.
- With normalization: each layer receives inputs with controlled magnitude.

---

### Why this matters

Normalization helps with three critical issues:

1. **Stabilize activation scale** so deep stacks remain trainable.
2. **Improve gradient flow**, avoiding vanishing or exploding gradients.
3. **Make training more robust** to hyperparameters and initialization.

---

## 2. What is RMSNorm?

### Definition

**RMSNorm** stands for *Root Mean Square Normalization*.

It removes the mean subtraction of LayerNorm and only normalizes by the RMS:

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma$$

where

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

**Parameters**:
- $x$: input tensor with shape `[batch_size, seq_len, hidden_dim]`
- $d$: hidden_dim
- $\epsilon$: small constant, e.g. `1e-5`
- $\gamma$: learnable scale parameter with shape `[hidden_dim]`
- $\odot$: element-wise multiplication

---

### RMSNorm vs LayerNorm

| Feature | LayerNorm | RMSNorm |
|------|-----------|---------|
| Formula | $(x - \mu) / (\sigma + \epsilon)$ | $x / (\text{RMS}(x) + \epsilon)$ |
| Steps | subtract mean + divide by std | divide by RMS only |
| Parameters | $2d$ (weight + bias) | $d$ (weight only) |
| Speed | slower | **7–64% faster** |
| Half precision | less stable | more stable |
| Common usage | BERT, GPT-2 | Llama, GPT-3+, MiniMind |

---

### Why RMSNorm is preferred in LLMs

1. **Simpler computation** (no mean subtraction).
2. **Faster training** due to fewer ops.
3. **More stable** in half precision (BF16/FP16).

MiniMind uses **Pre-LN + RMSNorm** as the default configuration.

---

### Implementation (reference)

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # learnable scale

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
```

**Notes**:
- `rsqrt` is $1/\sqrt{x}$, faster than `1/sqrt(x)`.
- `mean(-1)` computes RMS over hidden_dim.
- `.float()` improves stability in half precision.
- `.type_as(x)` returns to original dtype.

---

## 3. Pre-LN vs Post-LN

Normalization can be placed **before** or **after** each sub-layer.

### Post-LN (older)
```python
x = x + Attention(x)
x = LayerNorm(x)
x = x + FFN(x)
x = LayerNorm(x)
```

**Issues**:
- Residual path is less clean.
- Deep networks (>12 layers) become unstable.
- Training often requires smaller learning rates.

---

### Pre-LN (modern)
```python
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

**Advantages**:
- Cleaner residual path → gradients flow directly.
- More stable for deep networks.
- More tolerant to larger learning rates.

**MiniMind choice**: Pre-LN + RMSNorm.

---

## 4. How to verify (experiments)

### Experiment 1: Gradient vanishing

**Goal**: show how normalization prevents vanishing activations.

```bash
python experiments/exp1_gradient_vanishing.py
```

**Expected**:
- **No normalization**: std shrinks to 0.016
- **With normalization**: std stays around 1.0

Output: `results/gradient_vanishing.png`

---

### Experiment 2: Compare configurations

**Goal**: compare NoNorm / Post-LN / Pre-LN / Pre-LN + RMSNorm.

```bash
python experiments/exp2_norm_comparison.py
# quick mode
python experiments/exp2_norm_comparison.py --quick
```

**Expected (summary)**:

| Config | Converges | NaN | Final loss | Stability |
|------|-----------|-----|------------|----------|
| NoNorm | ❌ | ~500 steps | NaN | poor |
| Post-LN | ✅ | - | ~3.5 | sensitive (LR < 1e-4) |
| Pre-LN + LayerNorm | ✅ | - | ~2.8 | good |
| Pre-LN + RMSNorm | ✅ | - | ~2.7 | best |

Output: `results/norm_comparison.png`

---

### Experiment 3: Pre-LN vs Post-LN

**Goal**: show Pre-LN is more stable in deeper networks.

```bash
python experiments/exp3_prenorm_vs_postnorm.py
```

**Expected**:
- **4 layers**: both converge.
- **8 layers**: Pre-LN remains stable, Post-LN becomes unstable.

Output: `results/prenorm_vs_postnorm.png`

---

## 5. Key takeaways

1. **Normalization is essential** to stabilize deep networks.
2. **RMSNorm is simpler and faster** than LayerNorm while remaining stable.
3. **Pre-LN** is the modern default because it stabilizes gradients.

---

### MiniMind’s block (Pre-LN)

```python
# Pre-LN Transformer Block
residual = x
x = self.norm1(x)
x = self.attention(x)
x = residual + x

residual = x
x = self.norm2(x)
x = self.feedforward(x)
x = residual + x
```

Order: **Norm → Compute → Residual**.

---

## 6. Further reading

### Papers
- [RMSNorm: Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)

### Blogs
- [Layer Normalization](https://leimao.github.io/blog/Layer-Normalization/)
- [Why does normalization help?](https://blog.paperspace.com/busting-the-myths-about-batch-normalization/)

### Code references
- MiniMind: `model/model_minimind.py:95-105` (RMSNorm)
- MiniMind: `model/model_minimind.py:359-380` (TransformerBlock)

### Quiz
- [quiz.md](./quiz.md) - 5 self-check questions

---

## ✅ Self-check

Before moving on, make sure you can:
- [ ] Explain vanishing/exploding gradients
- [ ] Write the RMSNorm formula
- [ ] Explain RMSNorm vs LayerNorm
- [ ] Explain Pre-LN vs Post-LN
- [ ] Sketch the Pre-LN Transformer block flow
- [ ] Identify where RMSNorm appears in the model
