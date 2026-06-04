---
title: Residual Connection Teaching Notes | MiniMind LLM Training
description: Understand residual connections through information preservation, delta learning, and gradient flow.
keywords: residual connection theory, gradient flow, deep Transformer
---

# Residual Connection Teaching Notes

> A residual connection keeps a direct path outside the complex sub-layer.

## 1. Why residual connections are needed

A plain deep stack repeatedly computes:

$$
x_{l+1} = F_l(x_l)
$$

Every layer must rewrite the full representation, and backward gradients must pass through every sub-layer Jacobian. Repeated multiplication can make gradients vanish or explode.

A residual layer computes:

$$
x_{l+1} = x_l + F_l(x_l)
$$

Now the branch only needs to learn the required change:

$$
F_l(x_l) = x_{l+1} - x_l
$$

If the branch initially produces almost zero, the layer still approximates the identity mapping.

## 2. Forward and backward views

Plain layer:

$$
\frac{\partial y}{\partial x} = J_F
$$

Residual layer:

$$
\frac{\partial y}{\partial x} = I + J_F
$$

The identity term gives information and gradients a direct route across the layer.

::: warning Residuals are not a universal guarantee
Residual connections improve optimization, but very deep models still require normalization, appropriate initialization, sensible learning rates, and controlled residual scale.
:::

## 3. Why a Transformer block has two residual paths

A block has two major sub-layers:

1. Attention exchanges information between tokens.
2. FeedForward transforms each token representation.

Each sub-layer gets its own identity path:

$$
h = x + \operatorname{Attention}(\operatorname{Norm}(x))
$$

$$
y = h + \operatorname{FFN}(\operatorname{Norm}(h))
$$

This lets Attention and FFN each learn a delta without forcing either one to reconstruct the complete state.

## 4. Residuals and Pre-Norm

MiniMind uses:

$$
x_{l+1} = x_l + F_l(\operatorname{Norm}(x_l))
$$

The identity path does not pass through normalization. In Post-Norm:

$$
x_{l+1} = \operatorname{Norm}(x_l + F_l(x_l))
$$

the cross-layer path still passes through the normalization Jacobian. Pre-Norm is usually easier to optimize at depth.

## 5. Shape requirements

Elementwise addition requires matching shapes:

```python
x.shape == delta.shape
```

Attention returns `hidden_size`, and the FFN expands internally but projects back to `hidden_size`. If shapes differ, a projected shortcut such as `projection(x) + F(x)` is required.

## 6. Residual scale still matters

Across many layers:

$$
x_L = x_0 + \sum_{l=0}^{L-1} F_l(x_l)
$$

Large branch outputs can still accumulate. Normalization, initialization, residual scaling, learning rate, and gradient clipping all help keep the sum controlled.

## 7. How the experiments verify the idea

- **Experiment 1** trains a near-identity target. The residual model only learns the delta.
- **Experiment 2** compares input-side and output-side gradient norms.
- **Experiment 3** increases depth and measures cosine similarity between final output and original input.

## Key takeaway

> Residual connections make “keep the current representation” an easy default and let each layer focus on useful changes.
