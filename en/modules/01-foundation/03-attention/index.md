---
title: Attention Module | MiniMind LLM Training
description: Module navigation for Attention. Understand Self-Attention and the intuition behind Q, K, V.
keywords: attention, self-attention, QKV, Transformer attention
---

# 03. Attention

> How does Self-Attention work? What is the intuition behind Q, K, V?

---

## 🎯 Learning goals

After this module, you will be able to:
- ✅ Understand the math of Self-Attention
- ✅ Understand the intuition of Q, K, V
- ✅ Understand the benefits of Multi-Head Attention
- ✅ Understand GQA (Grouped Query Attention)
- ✅ Implement Scaled Dot-Product Attention from scratch

---

## 📚 Learning path

### 1️⃣ Quick experience (15 min)

```bash
cd experiments

# Exp 1: Attention basics
python exp1_attention_basics.py

# Exp 2: Q, K, V explained
python exp2_qkv_explained.py
```

---

## 🔬 Experiment list

| Experiment | Purpose | Time |
|------|------|------|
| exp1_attention_basics.py | permutation invariance + basic computation | 5 min |
| exp2_qkv_explained.py | intuition for Q, K, V | 5 min |
| exp3_multihead_attention.py | multi-head mechanism | 10 min |

---

## 💡 Key points

### 1. Core Self-Attention formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Intuition**:
- $QK^T$: relevance scores (who relates to whom)
- $\sqrt{d_k}$: scaling factor (avoid softmax saturation)
- softmax: normalize to a probability distribution
- $\times V$: weighted sum (extract relevant info)

---

### 2. Intuition for Q, K, V

| Role | Full name | Question | Analogy |
|------|------|------|------|
| **Q** | Query | What am I looking for? | Library search |
| **K** | Key | What labels do I provide? | Book keywords |
| **V** | Value | What content do I return? | Book contents |

**Example**: sentence "The cat sat on the mat"
- Query of "cat": “find info about animals”
- Key of "sat": “I am an action word”
- If Q·K matches → use the Value of "sat"

---

### 3. Why Multi-Head?

**Problem**: single-head attention learns only one “relation pattern.”

**Solution**: multiple heads in parallel, each learns a different pattern
- Head 1: syntax (subject–verb–object)
- Head 2: semantics (synonyms)
- Head 3: position (neighboring words)
- ...

**Formula**:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

---

### 4. GQA (Grouped Query Attention)

**MHA**: each head has its own Q, K, V
- Params: 3 × n_heads × head_dim

**GQA**: multiple Q heads share a set of K, V
- Params: (n_heads + 2 × n_kv_heads) × head_dim
- Smaller KV cache, faster inference

**MiniMind**: `n_heads=8, n_kv_heads=2`
- 8 Q heads share 2 KV heads
- Each 4 Q heads share a KV group

---

## 📖 Docs

- 📘 [teaching.md](./teaching.md) - full concept explanation
- 💻 [code_guide.md](./code_guide.md) - MiniMind code walkthrough
- 📝 [quiz.md](./quiz.md) - self-check

---

## ✅ Completion check

After finishing, you should be able to:

### Theory
- [ ] Write the attention formula
- [ ] Explain the roles of Q, K, V
- [ ] Explain the role of $\sqrt{d_k}$
- [ ] Explain why multi-head helps

### Practice
- [ ] Implement Scaled Dot-Product Attention from scratch
- [ ] Visualize attention weights
- [ ] Understand causal masking

---

## 🔗 Resources

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - original Transformer
- [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245)

### Code reference
- MiniMind: `model/model_minimind.py:250-330`

---

## 🎓 Next step

After finishing this module, go to:
👉 [04. FeedForward](/en/modules/01-foundation/04-feedforward)
