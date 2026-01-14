---
title: Position Encoding Teaching Notes | MiniMind LLM Training
description: Understand how Transformers handle positional information and how RoPE works. Compare absolute vs relative positional encoding and learn length extrapolation.
keywords: Position Encoding, RoPE, Rotary Position Embedding, absolute positional encoding, relative positional encoding, length extrapolation, Transformer positional encoding
---

# Position Encoding Teaching Notes

> Understand how Transformers handle positional information and how RoPE works

---

## 🤔 1. Why (Why)

### Problem scenario: Attention is permutation-invariant

**Problem**: Self-Attention is inherently **permutation-invariant**.

**Example**:
```
Sentence 1: "I like you"  → {I, like, you}
Sentence 2: "You like I"  → {You, like, I}
```

Without positional encoding, Attention sees:
- the same set of words
- identical attention weights
- **no way to tell the order**

But the meanings are clearly different.

---

### Intuition: give each word a “seat number”

🏷️ **Analogy**: positional encoding is like assigning each word a seat number.

- **No seat numbers**:
  - Teacher calls: "Zhang San, Li Si, Wang Wu"
  - Students sit anywhere
  - Seats change every day, hard to remember who sits where

- **With seat numbers**:
  - Zhang San → Seat 1
  - Li Si → Seat 2
  - Wang Wu → Seat 3
  - Positions are fixed and easier to remember

---

### The evolution of positional encoding

| Gen | Approach | Used by | Pros | Cons |
|----|------|---------|------|------|
| 1️⃣ | Absolute positional encoding | BERT, GPT-2 | Simple | Cannot extrapolate to longer sequences |
| 2️⃣ | Relative positional encoding | T5, XLNet | More flexible | Complex, harder to optimize |
| 3️⃣ | **RoPE** | Llama, MiniMind | Efficient + extrapolatable | - |

**MiniMind’s choice**: RoPE (Rotary Position Embedding)

---

## 📐 2. What (What)

### RoPE core idea

**One-line summary**: encode position as a rotation angle and rotate vectors by that angle.

**Basic intuition**:
```
Position 0 → rotate 0°
Position 1 → rotate θ°
Position 2 → rotate 2θ°
Position 3 → rotate 3θ°
...
Position m → rotate mθ°
```

---

### Mathematical definition

For a vector $\mathbf{x} = [x_0, x_1, ..., x_{d-1}]$, RoPE at position $m$ is:

$$\text{RoPE}(\mathbf{x}, m) = \begin{bmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) & & & \\
\sin(m\theta_0) & \cos(m\theta_0) & & & \\
& & \cos(m\theta_1) & -\sin(m\theta_1) & \\
& & \sin(m\theta_1) & \cos(m\theta_1) & \\
& & & & \ddots
\end{bmatrix} \begin{bmatrix}
x_0 \\ x_1 \\ x_2 \\ x_3 \\ \vdots
\end{bmatrix}$$

**Key points**:
- every two dimensions form a pair and share one rotation matrix
- different pairs use different frequencies $\theta_i$

---

### Frequency computation

$$\theta_i = \frac{1}{\text{base}^{2i/d}}$$

Where:
- $\text{base} = 10000$ (standard) or $1000000$ (MiniMind)
- $d$: head dimension (head_dim)
- $i = 0, 1, 2, ..., d/2-1$

**MiniMind settings** (head_dim=64):
```python
rope_base = 1000000.0  # larger base → better length extrapolation
freqs = [1/1000000^(2i/64) for i in range(32)]
```

This generates 32 frequencies:
- **High frequency** (i=0): $\theta_0 = 1.0$, one full turn every $2\pi$ (~6 tokens)
- **Mid frequency** (i=15): $\theta_{15} = 0.001$, one turn every 6283 tokens
- **Low frequency** (i=31): $\theta_{31} = 0.000001$, one turn every 6.28 million tokens

---

### Why multiple frequencies?

🕰️ **Clock analogy**:

| Hand | Speed | Purpose | RoPE mapping |
|------|---------|------|-----------|
| Second hand | Fast (1 min per turn) | Fine-grained | High frequency (local positions) |
| Minute hand | Medium (1 hr per turn) | Mid-range | Mid frequency |
| Hour hand | Slow (12 hrs per turn) | Global | Low frequency (global positions) |

**Problems with a single frequency**:
- Only high frequency: long sequences “wrap” too many times → position becomes ambiguous
- Only low frequency: short sequences rotate too slowly → poor discrimination

**Benefits of multiple frequencies**:
- High frequency: distinguish nearby tokens
- Low frequency: distinguish far-apart tokens
- Together: uniquely encode up to millions of positions

---

### Relative position emerges naturally

**Key property**: after RoPE, dot products depend only on relative position.

**Simplified proof**:

Position $m$ Query: $\mathbf{q}_m = \text{RoPE}(\mathbf{q}, m)$
Position $n$ Key: $\mathbf{k}_n = \text{RoPE}(\mathbf{k}, n)$

Dot product:
$$\mathbf{q}_m \cdot \mathbf{k}_n = f(\mathbf{q}, \mathbf{k}, m-n)$$

It depends only on the **relative distance** $m-n$, not absolute positions $m$ or $n$.

**Practical effect**:
- The relation between positions 5 and 8 = the relation between positions 100 and 103
- The model learns “distance = 3” patterns, not absolute indices

---

### Length extrapolation (YaRN)

**Question**: train on length 256, can we infer at length 512?

**RoPE advantage**:
- Absolute position embeddings: ❌ cannot extrapolate
- RoPE: ✅ can extrapolate (just rotate further)

**YaRN in MiniMind**:
- dynamically rescales rotation frequencies
- improves long-context generalization
- config: `inference_rope_scaling=True`

---

## 🔬 3. How to Verify

### Experiment 1: prove permutation invariance

**Goal**: show that without positional encoding, Attention can’t distinguish order

**Method**:
- Compare outputs for [A, B, C] vs [C, B, A]
- Check if Attention outputs match

**Run**:
```bash
python experiments/exp1_why_position.py
```

**Expected**:
- No positional encoding: outputs are **identical**
- With RoPE: outputs are **different**

---

### Experiment 2: RoPE basics

**Goal**: visualize how RoPE rotates vectors

**Method**:
- Show rotations in 2D
- Different positions rotate by different angles

**Run**:
```bash
python experiments/exp2_rope_basics.py
```

**Expected output**:
- rotation animation or static plot
- vectors for positions 0, 1, 2, 3

---

### Experiment 3: multi-frequency mechanism

**Goal**: show how different frequencies combine to encode position

**Method**:
- plot multiple frequency curves
- show the combined unique pattern

**Run**:
```bash
python experiments/exp3_multi_frequency.py
```

**Expected output**:
- multiple sine curves (different frequencies)
- a combined complex pattern

---

### Experiment 4: full implementation (optional)

**Goal**: understand the real MiniMind implementation

**Method**:
- full RoPE code
- includes YaRN extrapolation

**Run**:
```bash
python experiments/exp4_rope_explained.py
```

---

## 💡 4. Key takeaways

### Core conclusions

1. **Why positional encoding matters**:
   - Attention is permutation-invariant
   - extra info is required to encode order

2. **Why RoPE**:
   - naturally encodes relative positions
   - efficient (rotation-based)
   - supports extrapolation

3. **Why multiple frequencies**:
   - high frequency: local positions
   - low frequency: global positions
   - combined: uniquely identify long positions

---

### Design principle

In MiniMind, RoPE is applied to Attention’s Q and K:

```python
def forward(self, x, pos_ids):
    # compute Q, K, V
    q, k, v = self.split_heads(x)

    # apply RoPE (only Q and K)
    q = apply_rotary_emb(q, freqs_cis[pos_ids])
    k = apply_rotary_emb(k, freqs_cis[pos_ids])

    # attention (V does not need positional encoding)
    attn = softmax(q @ k.T / sqrt(d))
    output = attn @ v

    return output
```

**Remember**: RoPE applies only to Q and K, not V.

---

## 📚 5. Further reading

### Must-read papers
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (2021)
  - introduces RoPE and proves relative-position properties

### Recommended blogs
- [Understanding Rotary Position Embedding](https://blog.eleuther.ai/rotary-embeddings/)
- [The Illustrated RoPE](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

### Code references
- MiniMind: `model/model_minimind.py:108-200` - RoPE implementation
- MiniMind: `model/model_minimind.py:250-330` - usage inside Attention

### Quiz
- 📝 [quiz.md](./quiz.md) - reinforce your understanding

---

## 🎯 Self-check

After finishing this note, make sure you can:

- [ ] Explain why Attention is permutation-invariant
- [ ] Write the RoPE formula
- [ ] Explain why multiple frequencies are required
- [ ] Explain why RoPE yields relative position information
- [ ] Sketch a RoPE rotation diagram
- [ ] Implement a simple RoPE from scratch

If anything is unclear, return to the experiments and verify by hand.
