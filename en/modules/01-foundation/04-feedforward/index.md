---
title: FeedForward Module | MiniMind LLM Training
description: Module navigation for FeedForward. Learn why expand–compress exists and what advantages SwiGLU brings.
keywords: feedforward, FFN, SwiGLU, Transformer feedforward
---

# 04. FeedForward

> Why expand–compress? What advantages does SwiGLU provide?

---

## 🎯 Learning goals

After this module, you will be able to:
- ✅ Understand the expand–compress structure of FeedForward
- ✅ Understand why a high-dimensional intermediate layer is needed
- ✅ Understand the advantages of SwiGLU
- ✅ Understand the division of labor between FeedForward and Attention
- ✅ Implement SwiGLU FeedForward from scratch

---

## 📚 Learning path

### 1️⃣ Quick experience (10 min)

```bash
cd experiments

# Exp 1: FeedForward basics
python exp1_feedforward.py
```

---

## 🔬 Experiment list

| Experiment | Purpose | Time |
|------|------|------|
| exp1_feedforward.py | Understand expand–compress and SwiGLU | 10 min |

---

## 💡 Key points

### 1. Core structure of FeedForward

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 \cdot x)$$

**Standard FFN**:
- $W_1$: hidden_size → intermediate_size (expand)
- $\sigma$: activation (ReLU, GELU, ...)
- $W_2$: intermediate_size → hidden_size (compress)

**MiniMind SwiGLU**:
$$\text{SwiGLU}(x) = W_{\text{down}} \cdot (\text{SiLU}(W_{\text{gate}} \cdot x) \odot W_{\text{up}} \cdot x)$$

---

### 2. Why expand–compress?

**Problem with direct transform**:
- 768 → 768: only linear, limited expressiveness
- cannot fit complex nonlinear functions

**Advantage of expand–compress**:
- 768 → 2048 → 768: move into a higher-dimensional space
- easier to separate patterns in high dimensions
- compress back while preserving useful info

**Analogy**:
- Cooking: ingredients → chopped/processed → plated
- Photos: pixels → features → optimized pixels

---

### 3. SwiGLU activation

**Why not ReLU?**
- ReLU is simple but often weaker
- GLU variants perform better in LLMs

**SwiGLU formula**:
```python
hidden = SiLU(gate) * up  # gating
output = down_proj(hidden)
```

**Three projections**:
- `gate_proj`: compute gate signal
- `up_proj`: compute value signal
- `down_proj`: project back to hidden size

**SiLU (Swish)**:
$$\text{SiLU}(x) = x \cdot \sigma(x)$$

---

### 4. Division of labor: FeedForward vs Attention

| Component | Role | Analogy |
|------|------|------|
| **Attention** | exchange information between tokens | meeting discussion |
| **FeedForward** | process each token independently | individual thinking |

**Key difference**:
- Attention: seq_len × seq_len interactions
- FeedForward: independent per position

**Transformer block flow**:
```
x → RMSNorm → Attention → + → RMSNorm → FeedForward → +
              (info exchange)           (independent processing)
```

---

## 📖 Docs

- 📘 [teaching.md](./teaching.md) - full concept explanation
- 💻 [code_guide.md](./code_guide.md) - MiniMind code walkthrough
- 📝 [quiz.md](./quiz.md) - self-check

---

## ✅ Completion check

After finishing, you should be able to:

### Theory
- [ ] Explain why expand–compress is needed
- [ ] Explain the three projections in SwiGLU
- [ ] Explain SiLU activation
- [ ] Explain the division of labor with Attention

### Practice
- [ ] Implement standard FFN from scratch
- [ ] Implement SwiGLU from scratch
- [ ] Compare different activations

---

## 🔗 Resources

### Papers
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Code reference
- MiniMind: `model/model_minimind.py:330-380`

---

## 🎓 Next step

After finishing this module, go to:
👉 [05. Residual Connection](../../02-architecture/05-residual-connection)
