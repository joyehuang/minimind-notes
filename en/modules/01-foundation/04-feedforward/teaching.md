---
title: FeedForward Teaching Notes | MiniMind LLM Training
description: Understand the expand-compress structure of FeedForward networks and the SwiGLU activation. Learn how FFN stores knowledge and why expansion is needed.
keywords: FeedForward, FFN, SwiGLU, activation function, Transformer feedforward, LLM training
---

# FeedForward Teaching Notes

> Understand the "expand-compress" structure and the SwiGLU activation

---

## 🤔 1. Why (Why)

### Problem scenario: limited expressiveness

**Attention’s limitation**:
- Attention handles information exchange
- but it is mostly linear (weighted averages)
- it cannot express complex non-linear transformations

**Example**:
```
Input: [0.5, 1.0, 0.8]  → a token vector
Goal: learn "is this token a verb or a noun?"

You need a non-linear decision boundary, not a simple linear combination.
```

---

### Intuition: kitchen processing

🍳 **Analogy**: FeedForward is like a kitchen process

1. **Input**: raw ingredients (768-d vector)
2. **Expand**: chop and spread (768 → 2048)
   - finer granularity
   - more workspace
3. **Activate**: cook (non-linear transform)
   - create chemical reactions
   - irreversible change
4. **Compress**: plate the dish (2048 → 768)
   - return to original dimension
   - but content is transformed

**Key**: input/output dims match, but the content has been “processed.”

---

### Mathematical essence

FeedForward is a **universal function approximator**:

1. **Expand**: map to high-dimensional space
2. **Non-linear activation**: create complex decision boundaries
3. **Compress**: keep useful information and return to original size

**Theory** (Universal Approximation Theorem):
- a neural network with one hidden layer can approximate any continuous function
- wider hidden layers give stronger approximation power

---

## 📐 2. What (What)

### Standard FeedForward structure

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2$$

**Components**:
- $W_1$: hidden_size → intermediate_size (expansion)
- ReLU: activation
- $W_2$: intermediate_size → hidden_size (compression)

**Typical setup**:
- intermediate_size = 4 × hidden_size
- MiniMind: 512 → 2048 → 512

---

### Why expand then compress?

**Comparison**:

```python
# Option A: direct 768 → 768
output_A = W_direct(x)  # one linear layer

# Option B: expand-compress 768 → 2048 → 768
h = ReLU(W1(x))  # 768 → 2048
output_B = W2(h)  # 2048 → 768
```

**Option A issues**:
- only linear transform
- decision boundary is a hyperplane
- cannot separate complex patterns

**Option B advantages**:
- linear separability is more likely in higher dimensions
- non-linear activation builds complex boundaries
- compression keeps discriminative features

**Intuition**:
- in 2D, one line cannot separate a ring-shaped distribution
- map to 3D, a plane can separate it

---

### SwiGLU activation

MiniMind uses **SwiGLU** instead of ReLU:

$$\text{SwiGLU}(x) = W_{\text{down}} \cdot (\text{SiLU}(W_{\text{gate}} \cdot x) \odot W_{\text{up}} \cdot x)$$

**Three projections**:
1. `gate_proj`: $W_{\text{gate}} \cdot x$ (gate signal)
2. `up_proj`: $W_{\text{up}} \cdot x$ (value signal)
3. `down_proj`: compress back to original dim

**Gating mechanism**:
```python
gate = SiLU(gate_proj(x))  # [batch, seq, intermediate]
up = up_proj(x)            # [batch, seq, intermediate]
hidden = gate * up         # elementwise gate
output = down_proj(hidden)
```

**Why gating?**
- dynamically control information flow
- gate decides how much passes through
- more flexible than a single activation

---

### SiLU (Swish) activation

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**Properties**:
- smooth: differentiable everywhere
- non-monotonic: negative values not zeroed out
- self-gating: input multiplied by its own sigmoid

**Compared to ReLU**:
```
ReLU(x)  = max(0, x)
SiLU(x)  = x * sigmoid(x)

x = -1:
  ReLU(-1) = 0
  SiLU(-1) ≈ -0.27  # keeps some negative info
```

**Why SiLU is better**:
- smoother gradients → more stable training
- avoids “dead” negative region
- empirically better for LLMs

---

### GLU variant comparison

| Variant | Formula | Features |
|------|------|------|
| **GLU** | $\sigma(W_1 x) \odot W_2 x$ | sigmoid gating |
| **ReGLU** | $\text{ReLU}(W_1 x) \odot W_2 x$ | ReLU gating |
| **GeGLU** | $\text{GELU}(W_1 x) \odot W_2 x$ | GELU gating |
| **SwiGLU** | $\text{SiLU}(W_1 x) \odot W_2 x$ | SiLU gating |

**LLM best practice**: SwiGLU (Llama, MiniMind) or GeGLU (GPT-J)

---

### FeedForward vs Attention

**Transformer block**:
```
x → Norm → Attention → + → Norm → FeedForward → +
         (exchange)    residual   (local processing) residual
```

**Division of labor**:

| Component | Role | Analogy |
|------|------|------|
| **Attention** | global information exchange | team meeting, gather others’ input |
| **FeedForward** | local feature transform | individual thinking, digest info |

**Key difference**:
- Attention: seq × seq interactions
- FeedForward: per-position independent processing

**Why independent?**
- Attention already mixes information
- FeedForward does “deep thinking” per position
- separation reduces complexity

---

### Parameter count

**Standard FFN**:
```
W1: hidden_size × intermediate_size
W2: intermediate_size × hidden_size

total = 2 × hidden_size × intermediate_size
     = 2 × 512 × 2048 = 2M params
```

**SwiGLU** (three projections):
```
gate_proj: hidden_size × intermediate_size
up_proj:   hidden_size × intermediate_size
down_proj: intermediate_size × hidden_size

total = 3 × hidden_size × intermediate_size
     = 3 × 512 × (2048 × 2/3)  # usually shrink intermediate
     ≈ 2M params
```

**Note**: to keep params constant, SwiGLU typically uses an intermediate size scaled to 2/3.

---

## 🔬 3. How to Verify

### Experiment 1: FeedForward basics

**Goal**: understand expand-compress and SwiGLU

**Run**:
```bash
python experiments/exp1_feedforward.py
```

**Expected output**:
- show dimension changes
- compare activation functions
- visualize gating

---

## 💡 4. Key takeaways

### Core formulas

**Standard FFN**:
$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x)$$

**SwiGLU**:
$$\text{SwiGLU}(x) = W_{\text{down}} \cdot (\text{SiLU}(W_{\text{gate}} \cdot x) \odot W_{\text{up}} \cdot x)$$

### Core concepts

| Concept | Role | Key point |
|------|------|--------|
| Expansion | map to higher dimension | increase expressiveness |
| Activation | non-linear transform | create complex boundaries |
| Compression | back to original dimension | keep useful info |
| Gating | dynamic control | selective information flow |

### Design principle

```python
# MiniMind FeedForward config
hidden_size = 512
intermediate_size = 2048  # 4x expansion (or adjusted for SwiGLU)

# SwiGLU implementation
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

---

## 📚 5. Further reading

### Must-read papers
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - GLU family comparison
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - original Transformer FFN

### Recommended blogs
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - visualize FFN

### Code references
- MiniMind: `model/model_minimind.py:330-380` - FeedForward implementation

### Quiz
- 📝 [quiz.md](./quiz.md) - reinforce your understanding

---

## 🎯 Self-check

After finishing this note, make sure you can:

- [ ] Explain why expansion-compression is needed
- [ ] Explain the role of the three SwiGLU projections
- [ ] Explain SiLU vs ReLU
- [ ] Explain the gating mechanism
- [ ] Explain the division of labor between FeedForward and Attention
- [ ] Implement a SwiGLU FeedForward from scratch

If anything is unclear, return to the experiments and verify by hand.
