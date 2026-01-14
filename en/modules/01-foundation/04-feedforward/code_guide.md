---
title: FeedForward Code Walkthrough | MiniMind LLM Training
description: Understand the real FeedForward implementation in MiniMind. Learn expand-compress structure and SwiGLU activation details from the source.
keywords: FeedForward code, FFN implementation, SwiGLU code, Transformer feedforward source, LLM feedforward
---

# FeedForward Code Walkthrough

> Understand the real FeedForward implementation in MiniMind

---

## 📂 Code locations

### 1. FeedForward class

**File**: `model/model_minimind.py`
**Lines**: 330-380

```python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()

        hidden_dim = config.hidden_size
        intermediate_dim = config.intermediate_size

        # SwiGLU: three projection matrices
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU formula
        # output = down(SiLU(gate(x)) * up(x))
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )
```

---

### 2. Usage inside TransformerBlock

**File**: `model/model_minimind.py`
**Lines**: 400-450

```python
class TransformerBlock(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, pos_ids, mask):
        # Attention + residual
        h = x + self.attention(self.attention_norm(x), pos_ids, mask)

        # FeedForward + residual
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
```

---

## 🔍 Step-by-step

### The three SwiGLU projections

```python
# input x: [batch, seq, hidden_dim]

# 1. gate signal
gate = self.gate_proj(x)  # [batch, seq, intermediate_dim]

# 2. value signal
up = self.up_proj(x)      # [batch, seq, intermediate_dim]

# 3. SiLU activation + gating
hidden = F.silu(gate) * up  # [batch, seq, intermediate_dim]

# 4. compress back
output = self.down_proj(hidden)  # [batch, seq, hidden_dim]
```

**Dimension flow** (MiniMind 512 config):
```
Input:  [batch, seq, 512]
gate:   [batch, seq, 2048]  (expand)
up:     [batch, seq, 2048]  (expand)
hidden: [batch, seq, 2048]  (gate × up)
Output: [batch, seq, 512]   (compress)
```

---

### SiLU activation

```python
# F.silu(x) = x * torch.sigmoid(x)

x = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float)
silu = F.silu(x)
# tensor([-0.2384, -0.2689,  0.0000,  0.7311,  1.7616])

# compare with ReLU
relu = F.relu(x)
# tensor([0., 0., 0., 1., 2.])
```

**Properties**:
- smooth: differentiable everywhere
- non-monotonic: negative values not fully zeroed
- self-gating: $x \cdot \sigma(x)$

---

### Why three projections instead of two?

**Standard FFN (two projections)**:
```python
hidden = ReLU(W1(x))  # 768 → 2048
output = W2(hidden)   # 2048 → 768
```

**SwiGLU (three projections)**:
```python
gate = SiLU(W_gate(x))  # 768 → 2048
up = W_up(x)            # 768 → 2048
hidden = gate * up      # elementwise
output = W_down(hidden) # 2048 → 768
```

**Advantages**:
1. gating: dynamic control of information flow
2. stronger expressiveness: two paths provide different views
3. better empirical results on LLM benchmarks

**Parameter comparison**:
- Standard FFN: 2 × d × 4d = 8d²
- SwiGLU: 3 × d × (8d/3) = 8d² (adjusted intermediate)

---

### Gating mechanism explained

```python
gate = F.silu(self.gate_proj(x))  # gate signal
up = self.up_proj(x)              # value signal
hidden = gate * up                # elementwise gate

# gate behavior:
# - gate ≈ 0: close, suppress up
# - gate ≈ 1: open, pass up fully
# - 0 < gate < 1: partial pass
```

**Intuition**:
- gate is like a volume knob
- each dimension has its own volume
- the model learns what to amplify or suppress

---

## 💡 Implementation tips

### 1. No bias (bias=False)

```python
self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
```

**Why no bias?**
- bias has limited impact in large models
- fewer parameters
- works well with RMSNorm (already normalizes)

---

### 2. Choosing intermediate_size

```python
# MiniMind config
hidden_size = 512
intermediate_size = 2048  # 4x expansion

# for SwiGLU, some implementations adjust:
# intermediate_size = int(hidden_size * 4 * 2 / 3)
# to keep parameter count constant
```

**Llama’s choice**:
- intermediate_size = 2.7 × hidden_size (adjusted)
- or simply 4x with more parameters

---

### 3. Fused operations

```python
# naive implementation
gate = self.gate_proj(x)
up = self.up_proj(x)
hidden = F.silu(gate) * up

# can fuse gate_proj and up_proj
# to reduce memory IO

gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
gate, up = gate_up.chunk(2, dim=-1)
hidden = F.silu(gate) * up
```

---

## 📊 Performance considerations

### Compute cost

```python
# FeedForward FLOPs
# assume batch=1, seq=512, hidden=512, intermediate=2048

# gate_proj: 512 × 512 × 2048 = 536M FLOPs
# up_proj:   512 × 512 × 2048 = 536M FLOPs
# down_proj: 512 × 2048 × 512 = 536M FLOPs
# elementwise mul: 512 × 2048 ≈ 1M FLOPs

# total: ≈ 1.6G FLOPs per block
```

**Compared with Attention**:
- Attention: ≈ 1G FLOPs (seq=512)
- FeedForward: ≈ 1.6G FLOPs
- FeedForward dominates (~60%)

---

### Memory usage

```python
# intermediate activations
# gate: batch × seq × intermediate = batch × 512 × 2048 floats
# up:   batch × seq × intermediate = batch × 512 × 2048 floats

# total activation memory ≈ 2 × batch × 512 × 2048 × 4 bytes
#                         = batch × 8 MB
```

**Optimization tips**:
- use checkpointing: recompute activations instead of storing
- mixed precision: BF16/FP16

---

## 🔬 Experimental checks

### Verify dimension changes

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFeedForward(nn.Module):
    def __init__(self, dim=512, intermediate=2048):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        print(f"gate: {gate.shape}")

        up = self.up_proj(x)
        print(f"up: {up.shape}")

        hidden = F.silu(gate) * up
        print(f"hidden: {hidden.shape}")

        output = self.down_proj(hidden)
        print(f"output: {output.shape}")

        return output

# test
ffn = SimpleFeedForward()
x = torch.randn(2, 10, 512)  # [batch=2, seq=10, dim=512]
print(f"input: {x.shape}")
output = ffn(x)
```

### Verify gating effect

```python
# visualize gate signal
import matplotlib.pyplot as plt

x = torch.randn(1, 5, 512)  # 5 tokens
gate = F.silu(ffn.gate_proj(x))  # [1, 5, 2048]

# view gate activations
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.hist(gate[0, i].detach().numpy(), bins=50)
    plt.title(f"Token {i}")
plt.suptitle("Gate Activations")
plt.show()
```

---

## 🔗 Related code locations

1. **Config**: `model/model_minimind.py:30-65`
   - `intermediate_size`: middle dimension
   - `hidden_size`: hidden dimension

2. **MoE FeedForward**: `model/model_minimind.py:380-450`
   - mixture-of-experts variant
   - each expert is a FeedForward

3. **Full TransformerBlock**: `model/model_minimind.py:450-500`
   - Attention + FFN combination

---

## 🎯 Hands-on exercises

### Exercise 1: compare activations

Implement different FFNs and compare output distributions:
```python
def ffn_relu(x):
    return W2(F.relu(W1(x)))

def ffn_gelu(x):
    return W2(F.gelu(W1(x)))

def ffn_swiglu(x):
    return W_down(F.silu(W_gate(x)) * W_up(x))
```

### Exercise 2: visualize gating

Modify the code to store and plot gate activations:
```python
# save during forward
self.last_gate = F.silu(self.gate_proj(x))

# plot heatmap
plt.imshow(model.ffn.last_gate[0].detach().numpy())
```

### Exercise 3: compute actual FLOPs

Write code to compute real FLOPs:
```python
from thop import profile
flops, params = profile(ffn, inputs=(x,))
print(f"FLOPs: {flops/1e6:.2f}M, Params: {params/1e6:.2f}M")
```

---

## 📚 Further reading

- MiniMind full code: `model/model_minimind.py`
- Llama 2 code: [facebookresearch/llama](https://github.com/facebookresearch/llama)
- GLU paper: [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
