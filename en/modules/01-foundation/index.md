---
title: Foundation Modules | MiniMind LLM Training
description: "Understand Transformer building blocks: normalization, position encoding, attention, and feedforward. Learn through experiments and theory."
keywords: Transformer fundamentals, normalization, position encoding, attention, feedforward
---

# Tier 1: Foundation

> Understand the basic building blocks of a Transformer

---

## 🎯 Learning goals

After finishing this tier, you will be able to:
- ✅ Explain the role and math behind each core component
- ✅ Understand why modern LLMs choose these designs (over alternatives)
- ✅ Validate design choices via experiments
- ✅ Implement these components from scratch

---

## 📚 Module list

### [01. Normalization](/en/modules/01-foundation/01-normalization)

**Core questions**:
- Why do deep networks need normalization?
- RMSNorm vs LayerNorm: what is different?
- Pre-LN vs Post-LN: why is Pre-LN more stable?

**Key experiments**:
- Exp 1: visualize gradient vanishing (no norm vs norm)
- Exp 2: compare four configs (NoNorm / Post-LN / Pre-LN / RMSNorm)
- Exp 3: precision impact (FP32 / FP16 / BF16)

**Estimated time**: 1 hour

---

### [02. Position Encoding](/en/modules/01-foundation/02-position-encoding)

**Core questions**:
- Why does Attention need position?
- How does RoPE work?
- Why does RoPE extrapolate better?

**Key experiments**:
- Exp 1: permutation invariance without position encoding
- Exp 2: RoPE vs absolute position encoding
- Exp 3: visualize multi-frequency mechanism
- Exp 4: length extrapolation test

**Estimated time**: 1.5 hours

---

### [03. Attention](/en/modules/01-foundation/03-attention)

**Core questions**:
- What is the intuition behind QKV?
- Why do we need Multi-Head Attention?
- How does GQA improve efficiency?

**Key experiments**:
- Exp 1: visualize attention weights
- Exp 2: single-head vs multi-head
- Exp 3: GQA efficiency test
- Exp 4: causal mask effect

**Estimated time**: 2 hours

---

### [04. FeedForward](/en/modules/01-foundation/04-feedforward)

**Core questions**:
- What role does FFN play in Transformer?
- Why the expand–compress structure?
- SwiGLU vs ReLU: what is different?

**Key experiments**:
- Exp 1: expansion ratio impact
- Exp 2: activation function comparison
- Exp 3: ablation (remove FFN)

**Estimated time**: 1 hour

---

## 🚀 Learning advice

### Recommended order

**Follow this order**:
1. Normalization → stabilize training
2. Position Encoding → encode position
3. Attention → core mechanism
4. FeedForward → knowledge storage

Reason: later modules rely on earlier concepts.

### Learning method

For each module:
1. **Run experiments first** (20 min)
   - Build intuition: “so that’s what happens”
   - No need to fully understand code yet

2. **Read theory** (20 min)
   - Read `teaching.md`
   - Understand math and intuition

3. **Read code** (10 min)
   - Read `code_guide.md`
   - Linked to MiniMind original implementation

4. **Self-check** (10 min)
   - Complete `quiz.md`
   - Check your understanding

### Common questions

**Q: What if experiments fail?**

A: Check the following:
1. Did you activate the virtual environment? `source venv/bin/activate`
2. Did you download data? `cd modules/common && python datasets.py --download-all`
3. Are you in the correct folder? Experiments must run inside `experiments/`

**Q: Experiments are too slow?**

A: Use quick mode:
```bash
python exp_xxx.py --quick
```
Quick mode reduces steps (100) to verify the concept fast.

**Q: Want to go deeper on a topic?**

A: Each module’s `teaching.md` ends with “Further Reading”:
- Original papers
- Blogs
- Video tutorials

---

## 📊 Completion check

After finishing this tier, you should be able to answer:

### Theory
- [ ] Why do deep networks need normalization?
- [ ] How does RoPE encode relative position?
- [ ] What do Q, K, V represent in attention?
- [ ] Why does FFN expand then compress?

### Practice
- [ ] Implement RMSNorm from scratch
- [ ] Implement RoPE from scratch
- [ ] Implement Scaled Dot-Product Attention
- [ ] Implement SwiGLU FFN

### Design intuition
- [ ] Explain why Pre-LN is more stable than Post-LN
- [ ] Explain why RoPE beats absolute position encoding
- [ ] Explain why Multi-Head beats Single-Head
- [ ] Explain why FFN cannot be removed

---

## 🎓 Next step

After finishing this tier, go to:
👉 [Tier 2: Architecture](/en/modules/02-architecture)

Learn how to assemble these components into a full Transformer block.
