---
title: Systematic Study (8 hours) | MiniMind LLM Training
description: Master all core Transformer components. Study normalization, position encoding, attention, and feedforward to understand LLM training principles.
keywords: LLM systematic study, Transformer tutorial, LLM fundamentals
---

# 📚 Systematic Study (8 hours)

> Master all core Transformer components

## 🎯 Learning goals

After 8 hours you will be able to:
- ✅ Understand all core Transformer components
- ✅ Explain design choices via controlled experiments
- ✅ Implement a simple Transformer from scratch

## 📋 Learning path

### Stage 1: Foundation (5.5 hours)

Study the four core modules in order:

#### 1. Normalization (1 hour)

**What to do**:
- 📖 Read [teaching.md](/modules/01-foundation/01-normalization/teaching) (30 min)
- 🔬 Run all experiments (20 min)
- 📝 Finish [quiz.md](/modules/01-foundation/01-normalization/quiz) (10 min)

**Completion criteria**:
- [ ] Explain gradient vanishing/explosion
- [ ] Implement RMSNorm from scratch
- [ ] Understand Pre-LN vs Post-LN

[Start learning →](/en/modules/01-foundation/01-normalization/)

---

#### 2. Position Encoding (1.5 hours)

**What to do**:
- 📖 Read [teaching.md](/modules/01-foundation/02-position-encoding/teaching) (40 min)
- 🔬 Run experiments 1-3 (40 min)
- 📝 Self-check (10 min)

**Completion criteria**:
- [ ] Understand permutation invariance in Attention
- [ ] Explain the rotation idea behind RoPE
- [ ] Understand the role of multi-frequency components

[Start learning →](/en/modules/01-foundation/02-position-encoding/)

---

#### 3. Attention (2 hours)

**What to do**:
- 🔬 Run all experiments (1.5 hours)
- 💻 Read the source code (30 min)

**Completion criteria**:
- [ ] Understand the roles of Q, K, and V
- [ ] Understand the benefits of multi-head attention
- [ ] Understand GQA (Grouped Query Attention)

[Start learning →](/en/modules/01-foundation/03-attention/)

---

#### 4. FeedForward (1 hour)

**What to do**:
- 🔬 Run the experiments (40 min)
- 💻 Understand the SwiGLU activation (20 min)

**Completion criteria**:
- [ ] Understand the expand-compress pattern in FFN
- [ ] Understand the division of labor: Attention vs FFN
- [ ] Implement SwiGLU from scratch

[Start learning →](/en/modules/01-foundation/04-feedforward/)

---

### Stage 2: Architecture (2.5 hours)

**What to do**:
- 📖 Complete the [Residual Connection module](/en/modules/02-architecture/01-residual-connection/) (30 min)
- 📖 Complete the [Transformer Block module](/en/modules/02-architecture/02-transformer-block/) (1.5 hours)
- 📖 Review the [Architecture overview](/en/modules/02-architecture/) and finish self-checks (30 min)

**Completion criteria**:
- [ ] Draw the data flow of a Pre-LN Transformer block
- [ ] Understand the role of residual connections
- [ ] Implement a Transformer block from scratch

---

## 🎯 Checklist

After finishing Systematic Study, make sure you can:

### Foundation modules
- [ ] ✅ Complete Normalization
- [ ] ✅ Complete Position Encoding
- [ ] ✅ Complete Attention
- [ ] ✅ Complete FeedForward

### Practical skills
- [ ] ✅ Implement a Transformer block from scratch
- [ ] ✅ Pass all module quizzes
- [ ] ✅ Explain each design choice

---

## 📚 Next steps

Want to go deeper?
- 🎓 [Deep Mastery (30 hours)](/en/docs/guide/mastery) - train a full LLM from scratch
- 📝 [Record notes](/learning_log) - track your learning progress
- 🗺️ [Full roadmap](/en/ROADMAP) - view the complete learning path
