---
title: Architecture Modules | MiniMind LLM Training
description: Assemble core components into a full Transformer. Understand residuals, Transformer block order, and architecture design.
keywords: Transformer architecture, residual connection, Transformer block
---

# Tier 2: Architecture

> Assemble the core components into a complete Transformer

---

## 🎯 Learning goals

After finishing this tier, you will be able to:
- ✅ Understand how residual connections work
- ✅ Understand the Pre-Norm Transformer block layout
- ✅ Understand gradient flow in deep networks
- ✅ Implement a complete Transformer block from scratch

---

## 📚 Module list

### [01. Residual Connection](/en/modules/02-architecture/01-residual-connection)

**Core questions**:
- Why do we need residual connections?
- How do residuals solve gradient vanishing?
- What is the mathematical form of residuals?

**Key experiments**:
- Exp 1: training with/without residuals
- Exp 2: visualize gradient flow
- Exp 3: depth impact (shallow vs deep)

**Estimated time**: 1 hour

**Prerequisites**:
- Tier 1: Foundation (all modules)

---

### [02. Transformer Block](/en/modules/02-architecture/02-transformer-block)

**Core questions**:
- How do we assemble Norm, Attention, FFN, Residual?
- Why this order?
- Pre-Norm vs Post-Norm: what’s the difference?

**Key experiments**:
- Exp 1: component ablation
- Exp 2: ordering impact
- Exp 3: depth impact (2 layers vs 8 layers)

**Estimated time**: 1.5 hours

**Prerequisites**:
- 01. Residual Connection

---

## 🏗️ From components to architecture

### Assembly logic of a Transformer block

```
Input x
  ├─ save as residual
  ↓
RMSNorm (normalize, stabilize)
  ↓
Attention (attend) + RoPE (position)
  ↓
+ residual (residual #1)
  ├─ save current state
  ↓
RMSNorm (normalize again)
  ↓
FeedForward (process, store knowledge)
  ↓
+ residual (residual #2)
  ↓
Output
```

**Design points**:
1. **Pre-Norm**: normalize before each sub-layer
   - Stable input distribution per sub-layer
   - Smoother gradient flow

2. **Double residuals**: one for Attention, one for FFN
   - Gradients can skip complex sub-layers
   - Incremental learning: only learn the “delta”

3. **Fixed order**: Attention → FFN
   - Attention: find relevant information
   - FFN: process the information

---

## 🚀 Learning advice

### Recommended order

**Follow this order**:
1. Residual Connection (understand residuals)
2. Transformer Block (understand full assembly)

### Practice projects (optional)

After finishing this tier, try:

**Project 1: Implement a Transformer Decoder from scratch**
```python
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=8, hidden_size=512, ...):
        # stack multiple Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(...) for _ in range(num_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
```

**Project 2: Train a tiny model**
- Data: TinyShakespeare (1MB)
- Model: 4-layer Transformer, 256 hidden size
- Goal: perplexity < 2.5
- Time: ~30 minutes (CPU)

---

## 📊 Completion check

After finishing this tier, you should be able to answer:

### Theory
- [ ] What is the mathematical form of residual connection?
- [ ] Why do residuals solve gradient vanishing?
- [ ] How does gradient flow differ in Pre-Norm vs Post-Norm?
- [ ] Why does Attention come before FFN?

### Practice
- [ ] Implement residual connections from scratch
- [ ] Implement a Pre-Norm Transformer block
- [ ] Stack blocks into a full model
- [ ] Train a simple language model

### Design intuition
- [ ] Explain why deep nets need residuals
- [ ] Explain why Pre-Norm is more stable than Post-Norm
- [ ] Explain why double residuals are necessary
- [ ] Draw the full Transformer block dataflow

---

## 🎓 Next step

After finishing this tier, go to:
👉 **Tier 3: Training** (planned)

You will learn how to train these models:
- Data preparation and tokenization
- Optimizers and LR scheduling
- Distributed training
- Evaluation and debugging

---

## 🔗 Resources

### Papers
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - ResNet
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - Pre-LN vs Post-LN

### Code references
- MiniMind: `model/model_minimind.py:359-380` (TransformerBlock)
- MiniMind: `model/model_minimind.py:430-520` (full model)

### Visualization
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
