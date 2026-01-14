---
title: Position Encoding Module | MiniMind LLM Training
description: Module navigation for Position Encoding. Learn why Transformer needs position and how RoPE works.
keywords: position encoding, RoPE, rotary embeddings
---

# 02. Position Encoding

> Why does Transformer need position information? How does RoPE work?

---

## 🎯 Learning goals

After this module, you will be able to:
- ✅ Understand Attention’s permutation invariance issue
- ✅ Understand RoPE’s core idea (rotary encoding)
- ✅ Understand why multi-frequency is needed
- ✅ Understand how RoPE supports length extrapolation
- ✅ Implement RoPE from scratch

---

## 📚 Learning path

### 1️⃣ Quick experience (10 min)

```bash
cd experiments

# Exp 1: prove Attention is permutation-invariant
python exp1_why_position.py

# Exp 2: RoPE basics
python exp2_rope_basics.py --quick
```

---

## 🔬 Experiment list

| Experiment | Purpose | Time | Data |
|------|------|------|------|
| exp1_why_position.py | Show why position encoding is needed | 30 sec | synthetic |
| exp2_rope_basics.py | RoPE core idea | 2 min | synthetic |
| exp3_multi_frequency.py | Multi-frequency mechanism | 2 min | synthetic |
| exp4_rope_explained.py | Full implementation (optional) | 5 min | synthetic |

---

## 💡 Key points

### 1. Why position encoding?

**Problem**: Attention is permutation-invariant
```python
# Same words, different order
Sentence 1: "I like you" → Attention can’t tell order
Sentence 2: "You like I" → different meaning!
```

**Fix**: attach position information to each token

---

### 2. Core idea of RoPE

**Rotary encoding**: encode position via rotation matrices
- position 0: rotate 0°
- position 1: rotate θ°
- position 2: rotate 2θ°
- ...

**Advantages**:
- ✅ Relative position emerges naturally (angle differences)
- ✅ Supports length extrapolation (train 256, infer 512+)
- ✅ Efficient (apply rotation directly)

---

### 3. Multi-frequency mechanism

**Why multiple frequencies?**
- Low frequency (slow rotation): long-range positions (paragraph level)
- High frequency (fast rotation): short-range positions (token level)

**Analogy**:
- Clock hands: seconds (high), minutes (mid), hours (low)
- Combined to represent time accurately

---

## 📖 Docs

- 📘 [teaching.md](./teaching.md) - full concept explanation
- 💻 [code_guide.md](./code_guide.md) - MiniMind code walkthrough
- 📝 [quiz.md](./quiz.md) - self-check

---

## ✅ Completion check

After finishing, you should be able to:

### Theory
- [ ] Explain why Attention needs position information
- [ ] State the RoPE formula
- [ ] Explain the role of multi-frequency

### Practice
- [ ] Implement RoPE from scratch
- [ ] Draw the RoPE rotation diagram
- [ ] Explain how RoPE supports extrapolation

---

## 🔗 Resources

### Paper
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

### Code reference
- MiniMind: `model/model_minimind.py:108-200`

---

## 🎓 Next step

After finishing this module, go to:
👉 [03. Attention](/en/modules/01-foundation/03-attention)
