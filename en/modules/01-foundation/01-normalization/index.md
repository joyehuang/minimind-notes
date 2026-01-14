---
title: Normalization Module | MiniMind LLM Training
description: Module navigation for Normalization. Learn why deep nets need normalization and how RMSNorm stabilizes training.
keywords: normalization, RMSNorm, LayerNorm, Pre-LN, Post-LN
---

# 01. Normalization

> Why do deep networks need normalization? How does RMSNorm stabilize training?

---

## 🎯 Learning goals

After this module, you will be able to:
- ✅ Understand the root cause of vanishing/exploding gradients
- ✅ Understand how normalization stabilizes activation distributions
- ✅ Distinguish RMSNorm vs LayerNorm
- ✅ Distinguish Pre-LN vs Post-LN
- ✅ Implement RMSNorm from scratch

---

## 📚 Learning path

### 1️⃣ Quick experience (10 min)

Run experiments first to build intuition:

```bash
cd experiments

# Exp 1: See what happens without normalization
python exp1_gradient_vanishing.py

# Exp 2: Compare four configurations
python exp2_norm_comparison.py --quick
```

**You will see**:
- No normalization: activation std quickly collapses to 0 (vanishing gradients)
- Post-LN: unstable loss in early training
- Pre-LN + RMSNorm: stable convergence ✅

---

### 2️⃣ Theory (30 min)

Read the teaching doc:
- 📖 [teaching.md](./teaching.md)

**Core structure**:
- **Why**: why normalization is needed
- **What**: what RMSNorm is
- **How**: how to validate it

---

### 3️⃣ Implementation (15 min)

Read real code:
- 💻 [code_guide.md](./code_guide.md)

**Key files**:
- `model/model_minimind.py:95-105` - RMSNorm implementation
- `model/model_minimind.py:359-380` - usage inside TransformerBlock

---

### 4️⃣ Self-check (5 min)

Finish the quiz:
- 📝 [quiz.md](./quiz.md)

---

## 🔬 Experiment list

| Experiment | Purpose | Time | Data |
|------|------|------|------|
| exp1_gradient_vanishing.py | Show why normalization is necessary | 10 sec | synthetic |
| exp2_norm_comparison.py | Compare four configs | 5 min | TinyShakespeare |
| exp3_prenorm_vs_postnorm.py | Pre-LN vs Post-LN | 8 min | TinyShakespeare |

### Run all experiments

```bash
cd experiments
bash run_all.sh
```

---

## 💡 Key points

### 1. Why normalization?

```
Deep net without normalization:
Layer 1: std = 1.00
Layer 2: std = 0.85
Layer 3: std = 0.62
Layer 4: std = 0.38
...
Layer 8: std = 0.016  ← gradients almost vanish!
```

**Intuition**: like adding a “pressure regulator” to a faucet—output stays stable even when input changes.

---

### 2. RMSNorm vs LayerNorm

| Feature | LayerNorm | RMSNorm |
|------|-----------|---------|
| Steps | subtract mean + divide by std | divide by RMS |
| Params | 2d (γ, β) | d (γ) |
| Speed | slower | faster by 7–64% |
| Half precision stability | weaker | better |

**Conclusion**: RMSNorm is simpler, faster, and more stable.

---

### 3. Pre-LN vs Post-LN

```python
# Post-LN (older)
x = x + Attention(x)          # compute
x = LayerNorm(x)              # normalize after

# Pre-LN (modern)
x = x + Attention(Norm(x))    # normalize before
x = x + FFN(Norm(x))          # then compute
```

**Advantages**:
- ✅ Cleaner residual path (gradients flow directly)
- ✅ More stable for deep networks (>12 layers)
- ✅ More tolerant to learning rate

---

## 📊 Expected results

After running experiments, you should see under `experiments/results/`:

1. **gradient_vanishing.png**
   - No norm: std decays
   - With norm: std stays stable

2. **norm_comparison.png**
   - Four loss curves
   - NoNorm becomes NaN after ~500 steps
   - Pre-LN is most stable

3. **prenorm_vs_postnorm.png**
   - Convergence comparison

---

## ✅ Completion check

After finishing, you should be able to:

### Theory
- [ ] Explain gradient vanishing in your own words
- [ ] State the RMSNorm formula
- [ ] Explain why Pre-LN is better than Post-LN

### Practice
- [ ] Implement RMSNorm from scratch
- [ ] Draw the Pre-LN Transformer block flow
- [ ] Debug normalization-related training issues

### Intuition
- [ ] Explain normalization with an analogy
- [ ] Predict what happens without normalization
- [ ] Explain why modern LLMs choose Pre-LN + RMSNorm

---

## 🔗 Resources

### Papers
- [RMSNorm](https://arxiv.org/abs/1910.07467) - Root Mean Square Layer Normalization
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - Pre-LN vs Post-LN

### Blogs
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Understanding Layer Normalization](https://leimao.github.io/blog/Layer-Normalization/)

### Videos
- [Andrej Karpathy - Let’s build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

## 🎓 Next step

After finishing this module, go to:
👉 [02. Position Encoding](/en/modules/01-foundation/02-position-encoding)

Learn how Transformer handles position and why RoPE works.
