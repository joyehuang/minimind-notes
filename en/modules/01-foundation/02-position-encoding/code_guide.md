---
title: Position Encoding Code Walkthrough | MiniMind LLM Training
description: Understand the real RoPE implementation in MiniMind. Learn the code details for rotary frequency computation and application.
keywords: RoPE code, positional encoding implementation, rotary position embedding source, Transformer positional encoding code, LLM positional encoding
---

# Position Encoding Code Walkthrough

> Understand the real RoPE implementation in MiniMind

---

## 📂 Code locations

### 1. Precompute rotary frequencies

**File**: `model/model_minimind.py`
**Lines**: 108-128

```python
def precompute_freqs_cis(dim: int, end: int, rope_base: float = 1e6, rope_scaling=None):
    """Precompute RoPE frequencies"""

    # frequency: 1 / (base^(2i/dim))
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # positions [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)

    # YaRN extrapolation (optional)
    if rope_scaling is not None:
        t = t / rope_scaling

    # rotation angles: position * frequency
    freqs = torch.outer(t, freqs)  # [end, dim//2]

    # complex form (cos + i*sin)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [end, dim//2]

    return freqs_cis
```

---

### 2. Apply rotary embeddings

**File**: `model/model_minimind.py`
**Lines**: 131-145

```python
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply RoPE to Query and Key"""

    # real → complex
    # [batch, seq, heads, head_dim] -> [batch, seq, heads, head_dim//2, 2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # reshape freqs_cis for broadcasting
    freqs_cis = freqs_cis[:, None, :]  # [seq, 1, head_dim//2]

    # complex multiplication = rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

---

### 3. Use inside Attention

**File**: `model/model_minimind.py`
**Lines**: 250-290

```python
class Attention(nn.Module):
    def forward(self, x, pos_ids, mask):
        batch, seq_len, _ = x.shape

        # compute Q, K, V
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # reshape into heads
        xq = xq.view(batch, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # ⭐ apply RoPE (only Q and K)
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis[pos_ids])

        # attention scores
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ... softmax and output
```

---

## 🔍 Step-by-step

### Frequency formula

```python
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
```

**Breakdown**:
1. `torch.arange(0, dim, 2)`: [0, 2, 4, ..., dim-2]
2. `[: (dim // 2)]`: keep the first dim/2 elements
3. `/ dim`: normalize to [0, 1)
4. `rope_base ** (...)`: exponentiation
5. `1.0 / ...`: take reciprocal

**MiniMind parameters** (head_dim=64, rope_base=1e6):
```
freqs[0]  = 1.0           # high frequency: one turn per 2π
freqs[15] = 0.001         # mid frequency: one turn per 6283 positions
freqs[31] = 0.000001      # low frequency: one turn per 6.28 million positions
```

---

### Why complex numbers?

```python
# real vector → complex
xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))

# complex multiply = rotation
xq_out = xq_ * freqs_cis
```

**Reason**: complex multiplication naturally represents 2D rotation.

$$e^{i\theta} = \cos\theta + i\sin\theta$$

$$(a + bi) \times e^{i\theta} = (a\cos\theta - b\sin\theta) + i(a\sin\theta + b\cos\theta)$$

That is exactly a rotation matrix.

**Equivalent matrix form**:
```python
# these are equivalent:
# 1. complex multiplication
result = (a + bi) * (cos_θ + i*sin_θ)

# 2. matrix multiplication
result = [[cos_θ, -sin_θ],   @  [[a],
          [sin_θ,  cos_θ]]      [b]]
```

Complex form is shorter and faster.

---

### Pairing dimensions

```python
xq_ = xq.reshape(*xq.shape[:-1], -1, 2)
# [batch, seq, heads, head_dim] → [batch, seq, heads, head_dim//2, 2]
```

**Why pair?**
- 2D rotation needs two coordinates
- each pair shares one rotation angle
- head_dim=64 → 32 pairs → 32 frequencies

**Diagram**:
```
head_dim = 64

[x0, x1,  x2, x3,  ..., x62, x63]
  ↓   ↓    ↓   ↓         ↓    ↓
 pair0   pair1   ...   pair31

Each pair uses its own frequency
```

---

## 💡 Implementation tips

### 1. Precompute freqs_cis

```python
# precompute at initialization
self.freqs_cis = precompute_freqs_cis(
    dim=self.head_dim,
    end=self.max_seq_len,
    rope_base=config.rope_theta
)
```

**Benefits**:
- avoid recomputing every forward pass
- allow arbitrary position indices (pos_ids)

---

### 2. Use torch.polar

```python
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
```

**Equivalent to**:
```python
freqs_cis = torch.exp(1j * freqs)
# or
freqs_cis = torch.cos(freqs) + 1j * torch.sin(freqs)
```

`torch.polar(r, θ)` builds complex numbers from polar coordinates efficiently.

---

### 3. Apply only to Q and K

```python
xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
# V does not need positional encoding
```

**Why not V?**
- Q and K determine attention scores (need position info)
- V is the content being fetched
- position info is already captured in Q·K

---

### 4. Preserve dtype

```python
return xq_out.type_as(xq), xk_out.type_as(xk)
```

**Why?**
- complex math requires float32
- model may run in BF16/FP16
- cast back to keep dtype consistent

---

## 📊 Performance considerations

### Memory efficiency

```python
# good: precompute and store
self.register_buffer('freqs_cis', precompute_freqs_cis(...))

# bad: recompute every forward
freqs_cis = precompute_freqs_cis(...)  # wasted compute
```

### Compute efficiency

Complex multiply is cheaper than matrix multiply:
- Matrix: 4 muls + 2 adds
- Complex: 2 muls + 2 adds (GPU-friendly)

---

## 🔬 Experimental check

### Verify relative position property

```python
# positions 5 and 8
q5 = apply_rotary_emb(q, freqs_cis[5])
k8 = apply_rotary_emb(k, freqs_cis[8])
score_5_8 = q5 @ k8.T

# positions 100 and 103 (same distance = 3)
q100 = apply_rotary_emb(q, freqs_cis[100])
k103 = apply_rotary_emb(k, freqs_cis[103])
score_100_103 = q100 @ k103.T

# scores should match (relative distance only)
assert torch.allclose(score_5_8, score_100_103)
```

---

## 🔗 Related code locations

1. **Config**: `model/model_minimind.py:30-65`
   - `rope_theta`: base frequency (default 1e6)
   - `max_position_embeddings`: max sequence length

2. **YaRN support**: `model/model_minimind.py:120-125`
   - `inference_rope_scaling`: extrapolation factor

3. **Full Attention**: `model/model_minimind.py:250-330`
   - includes GQA (Grouped Query Attention)

---

## 🎯 Hands-on exercises

### Exercise 1: visualize rotation

Modify `exp2_multi_frequency.py` to plot rotations at different frequencies:
```python
import matplotlib.pyplot as plt

freqs = precompute_freqs_cis(dim=64, end=100)
for i in [0, 15, 31]:
    plt.plot(freqs[:, i].real, label=f'freq_{i}')
plt.legend()
plt.show()
```

### Exercise 2: verify relative position

Write code to verify that attention scores for positions (5, 8) and (100, 103) are equal.

### Exercise 3: compare absolute positional encoding

Implement a simple absolute positional embedding and compare its extrapolation ability to RoPE.

---

## 📚 Further reading

- MiniMind full code: `model/model_minimind.py`
- Llama 2 code: [facebookresearch/llama](https://github.com/facebookresearch/llama)
- RoFormer paper: [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
