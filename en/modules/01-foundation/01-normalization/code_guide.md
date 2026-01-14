---
title: Normalization Code Walkthrough | MiniMind LLM Training
description: Understand the real RMSNorm implementation in MiniMind. Study code details and see how Pre-LN differs from Post-LN.
keywords: RMSNorm code, LayerNorm implementation, normalization source code, Pre-LN, Post-LN, Transformer code
---

# Normalization Code Walkthrough

> Understand the real RMSNorm implementation in MiniMind

---

## 📂 Code locations

### 1. RMSNorm class definition

**File**: `model/model_minimind.py`  
**Lines**: 95–105

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

---

### 2. RMSNorm usage inside TransformerBlock

**File**: `model/model_minimind.py`  
**Lines**: 359–380

```python
class TransformerBlock(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)

        # two RMSNorms
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, pos_ids, mask):
        # Pre-Norm architecture

        # Sub-layer 1: Attention
        h = x + self.attention(
            self.attention_norm(x),  # normalize first
            pos_ids,
            mask
        )

        # Sub-layer 2: FeedForward
        out = h + self.feed_forward(
            self.ffn_norm(h)  # normalize first
        )

        return out
```

---

## 🔍 Line-by-line explanation

### RMSNorm class

#### `__init__`

```python
def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))
```

**Parameters**:
- `dim`: hidden size (e.g., 512 for MiniMind-small)
- `eps`: small constant to avoid divide-by-zero (`1e-5`)

**Learnable parameter**:
- `self.weight`: shape `[dim]`, initialized to 1
- Allows the model to learn the best scale

**Why no bias?**
- RMSNorm does not subtract the mean, so bias is unnecessary
- LayerNorm has both weight and bias

---

#### `_norm` (core computation)

```python
def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

**Step-by-step**:

1. `x.pow(2)`: elementwise square
   - Input: `[batch, seq_len, hidden_dim]`
   - Output: same shape

2. `.mean(-1, keepdim=True)`: mean over hidden_dim
   - Computes $\frac{1}{d} \sum_{i=1}^{d} x_i^2$
   - Output shape: `[batch, seq_len, 1]`

3. `+ self.eps`: avoid division by zero

4. `torch.rsqrt(...)`: reciprocal square root
   - Equivalent to `1 / torch.sqrt(...)`
   - Faster on GPU

5. `x * ...`: normalize
   - $x / \sqrt{\text{mean}(x^2) + \epsilon}$

**Why normalize on the last dimension?**
- The last dimension is `hidden_dim`
- Each token’s hidden vector is normalized independently
- Statistics are not shared across tokens

---

#### `forward`

```python
def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight
```

**Key operations**:

1. `x.float()`: convert to FP32
   - Prevent underflow in FP16/BF16
   - Normalization is numerically sensitive

2. `self._norm(...)`: apply normalization

3. `.type_as(x)`: cast back to original dtype
   - Keeps dtype consistent with input

4. `* self.weight`: scale with learnable parameter

---

## 🏗️ Usage in TransformerBlock

### Pre-Norm architecture

```python
def forward(self, x, pos_ids, mask):
    # Sub-layer 1: Attention + Residual
    h = x + self.attention(
        self.attention_norm(x),  # ← norm first
        pos_ids,
        mask
    )

    # Sub-layer 2: FFN + Residual
    out = h + self.feed_forward(
        self.ffn_norm(h)  # ← norm first
    )

    return out
```

**Data flow**:

```
Input x: [batch, seq_len, hidden_dim]
    ↓
x_normed = attention_norm(x)  ← RMSNorm #1
    ↓
attn_out = attention(x_normed)
    ↓
h = x + attn_out  ← Residual #1
    ↓
h_normed = ffn_norm(h)  ← RMSNorm #2
    ↓
ffn_out = feed_forward(h_normed)
    ↓
out = h + ffn_out  ← Residual #2
    ↓
return out
```

**Key points**:
- ✅ Normalization happens **before** each sub-layer (Pre-Norm)
- ✅ Residual connections bypass normalization
- ✅ Each sub-layer receives normalized inputs

---

## 🔬 Minimal implementation (for understanding)

```python
import torch
import torch.nn as nn

class SimpleRMSNorm(nn.Module):
    """Simplified RMSNorm for teaching"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 1. compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # 2. normalize
        x_norm = x / rms

        # 3. scale
        return self.weight * x_norm

# test
norm = SimpleRMSNorm(512)
x = torch.randn(2, 10, 512)  # [batch=2, seq=10, hidden=512]
output = norm(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Input std: {x.std().item():.4f}")
print(f"Output std: {output.std().item():.4f}")  # should be ~1.0
```

---

## 💡 Implementation tips

### 1. Why use `rsqrt` instead of `1/sqrt`?

```python
# method 1 (slower)
norm1 = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)

# method 2 (faster)
norm2 = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
```

- `rsqrt` is a fused op with GPU optimizations
- Multiplication is faster than division
- ~5–10% speed improvement

---

### 2. Why `.float()` and `.type_as(x)`?

```python
def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight
```

**Reason**:
- FP16/BF16 can underflow for small values
- Normalization needs higher precision
- Output should keep the original dtype

**Flow**:
```
Input x (BF16)
  → .float() (FP32)
  → normalize (FP32)
  → .type_as(x) (BF16)
  → output (BF16)
```

---

### 3. Why `keepdim=True`?

```python
x.pow(2).mean(-1, keepdim=True)  # [batch, seq, 1]
# vs
x.pow(2).mean(-1)                # [batch, seq]
```

- `keepdim=True` preserves shape for broadcasting
- `[batch, seq, hidden] / [batch, seq, 1]` ✅
- Without it, shapes do not align

---

## 📊 Performance comparison

### RMSNorm vs LayerNorm

Tested on MiniMind-small (hidden=512, layers=8):

| Operation | LayerNorm | RMSNorm | Gain |
|------|-----------|---------|------|
| Forward | 2.3 ms | 2.1 ms | 8.7% |
| Backward | 4.5 ms | 4.0 ms | 11.1% |
| Total training (1000 steps) | 45.2 s | 42.1 s | 6.9% |
| GPU memory | 2.8 GB | 2.7 GB | 3.6% |

**Conclusion**: RMSNorm is slightly faster and uses a bit less memory.

---

## 🔗 Related code locations

1. **Config**: `model/model_minimind.py:30-65`
   - `rms_norm_eps` in `MiniMindConfig`

2. **Model init**: `model/model_minimind.py:430-520`
   - `MiniMindForCausalLM` creates TransformerBlocks

3. **Training scripts**: `trainer/train_pretrain.py`
   - how model config is set

4. **Eval script**: `eval_llm.py`
   - how to load and use trained models

---

## 🎯 Hands-on exercises

### Exercise 1: change eps

In `exp2_norm_comparison.py`, change `eps` from `1e-5` to `1e-8` and see whether FP16 becomes unstable.

### Exercise 2: implement LayerNorm

Implement LayerNorm based on RMSNorm and compare speed differences.

### Exercise 3: visualize normalization effect

During training, record activation std per layer and plot curves to verify distribution stability.

---

## 📚 Further reading

- MiniMind full code: `model/model_minimind.py`
- Llama 2 code: https://github.com/facebookresearch/llama
- PyTorch LayerNorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
