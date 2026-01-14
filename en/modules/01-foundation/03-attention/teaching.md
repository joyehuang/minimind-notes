---
title: Attention Teaching Notes | MiniMind LLM Training
description: Understand how Self-Attention helps models focus on relevant information. Learn QKV, multi-head attention, and scaled dot-product attention with experiments.
keywords: attention mechanism, Attention, Self-Attention, QKV, multi-head attention, Transformer attention, LLM attention
---

# Attention Teaching Notes

> Understand how Self-Attention helps models focus on relevant information

---

## 🤔 1. Why (Why)

### Problem scenario: relations between words

**Example**:
```
Sentence: "Xiao Ming really loves his cat, and it always sleeps by the window."

Question: What does "it" refer to?
```

Humans easily know that “it” refers to “cat,” not “Xiao Ming” or “window.” But how does a model know?

**We need a mechanism** that lets the model focus on relevant parts of the sentence.

---

### Intuition: library search

📚 **Analogy**: Self-Attention is like searching for books in a library.

1. **Query**: what are you looking for?
   - "I want info about cats"

2. **Key**: tags/keywords for each book
   - *Biography of Xiao Ming* → tags: person
   - *The Life of Cats* → tags: animal, cat
   - *Window Design* → tags: architecture

3. **Value**: the actual content
   - after matching, you read the content

4. **Attention**: the matching score decides how much content to take from each book

---

### Mathematical essence

Attention does three things:

1. **Similarity**: dot product between Query and Key
2. **Normalization**: softmax into a probability distribution
3. **Weighted sum**: apply probabilities to Value

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

---

## 📐 2. What (What)

### Self-Attention formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Step-by-step**:

#### Step 1: similarity scores $QK^T$

```
Q: [batch, seq_len, d_k]
K: [batch, seq_len, d_k]

QK^T: [batch, seq_len, seq_len]
      ↑
    correlation between every pair of tokens
```

Result: a seq_len × seq_len matrix where element (i, j) is the relevance between token_i and token_j.

---

#### Step 2: scaling by $/ \sqrt{d_k}$

**Why scale?**
- dot product variance grows with $d_k$
- large values saturate softmax (gradients shrink)
- divide by $\sqrt{d_k}$ to stabilize variance

**Example** (d_k=64):
- unscaled score could be ~64 (too large)
- scaled score is ~8 (reasonable)

---

#### Step 3: softmax normalization

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

- converts scores to a probability distribution
- all weights sum to 1
- higher score → higher weight

---

#### Step 4: weighted sum $\times V$

```
attention_weights: [batch, seq_len, seq_len]
V: [batch, seq_len, d_v]

output: [batch, seq_len, d_v]
```

**Effect**: each token output is a weighted average of all Values.

---

### How Q, K, V are created

In Self-Attention, Q, K, V all come from the same input:

```python
# input x: [batch, seq_len, hidden_dim]

Q = x @ W_Q  # [batch, seq_len, d_k]
K = x @ W_K  # [batch, seq_len, d_k]
V = x @ W_V  # [batch, seq_len, d_v]
```

**Three projection matrices** $W_Q, W_K, W_V$ are learnable.

**Why not use x directly?**
- projections let the model learn different “views” of the token
- Q: what to query for
- K: how to present itself to be queried
- V: what information to pass along

---

### Multi-Head Attention

**Problem**: a single head can learn only one type of relation

**Solution**: multiple heads in parallel, each learning a different pattern

```python
# 8 heads
heads = []
for i in range(8):
    head_i = Attention(Q_i, K_i, V_i)
    heads.append(head_i)

# concat + projection
output = Concat(heads) @ W_O
```

**Different heads can learn**:
- Head 1: syntactic relations (subject-verb-object)
- Head 2: semantic similarity (synonyms)
- Head 3: positional relations (nearby words)
- Head 4: coreference (pronouns)
- ...

---

### GQA (Grouped Query Attention)

**MHA problem**: KV cache is huge

```
MHA: independent K, V per head
     8 heads × 512 seq × 64 dim = 262,144 values/token
```

**GQA solution**: multiple Q heads share KV

```
GQA: 8 Q heads, 2 KV heads
     every 4 Q heads share one KV
     memory reduced by 75%
```

**MiniMind config**:
```python
n_heads = 8        # Q heads
n_kv_heads = 2     # KV heads
# 4 Q heads share 1 KV
```

---

### Causal mask

**Problem**: language models can only see the past

```
When generating "The cat sat":
- "cat" can see "The"
- "sat" can see "The", "cat"
- but "The" cannot see "cat" (not generated yet)
```

**Solution**: mask out future positions

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
# upper triangle = 1 (masked)

scores = scores.masked_fill(mask == 1, float('-inf'))
# softmax(−∞) = 0, fully ignored
```

---

## 🔬 3. How to Verify

### Experiment 1: Attention basics

**Goal**: understand Attention computation

**Run**:
```bash
python experiments/exp1_attention_basics.py
```

**Expected output**:
- demonstrate permutation invariance
- visualize attention weight matrix

---

### Experiment 2: Q, K, V explained

**Goal**: see how Q, K, V work

**Run**:
```bash
python experiments/exp2_qkv_explained.py
```

**Expected output**:
- show Q, K, V generation
- compare different projection matrices

---

### Experiment 3: Multi-Head Attention

**Goal**: understand the advantage of multi-heads

**Run**:
```bash
python experiments/exp3_multihead_attention.py
```

**Expected output**:
- compare single-head vs multi-head
- visualize patterns learned by different heads

---

## 💡 4. Key takeaways

### Core formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Core concepts

| Concept | Role | Analogy |
|------|------|------|
| Q (Query) | what I want | search keywords |
| K (Key) | what tags I have | index labels |
| V (Value) | actual content | document content |
| $\sqrt{d_k}$ | scaling factor | prevent softmax saturation |
| Multi-Head | multiple relation patterns | multiple viewpoints |
| Causal Mask | see only past | language model constraint |

### Design principle

```python
# MiniMind Attention config
n_heads = 8           # 8 attention heads
n_kv_heads = 2        # GQA: 2 KV heads
head_dim = 64         # 64 dims per head
# hidden_size = 8 × 64 = 512
```

---

## 📚 5. Further reading

### Must-read papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - original Transformer paper
- [GQA Paper](https://arxiv.org/abs/2305.13245) - Grouped Query Attention

### Recommended blogs
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)

### Code references
- MiniMind: `model/model_minimind.py:250-330` - Attention implementation
- MiniMind: `model/model_minimind.py:180-210` - GQA implementation

### Quiz
- 📝 [quiz.md](./quiz.md) - reinforce your understanding

---

## 🎯 Self-check

After finishing this note, make sure you can:

- [ ] Write the full Attention formula
- [ ] Explain the roles of Q, K, V and how they are produced
- [ ] Explain why scaling is needed
- [ ] Explain the advantage of Multi-Head
- [ ] Explain how GQA reduces memory
- [ ] Explain the role of causal masking
- [ ] Implement Scaled Dot-Product Attention from scratch

If anything is unclear, return to the experiments and verify by hand.
