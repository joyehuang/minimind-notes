---
title: Transformer Block Teaching Notes | MiniMind LLM Training
description: Understand how a Pre-Norm decoder block assembles Attention, FeedForward, RMSNorm, causal masking, and residuals.
keywords: Transformer block theory, decoder block, Pre-Norm, causal attention
---

# Transformer Block Teaching Notes

> A Transformer block defines the cooperation protocol between information routing, per-token processing, scale control, and direct paths.

## 1. Why a block is needed

Each component solves one problem:

| Component | Role |
|---|---|
| RMSNorm | control sub-layer input scale |
| Attention | route information between tokens |
| RoPE | inject position into Q/K |
| FeedForward | apply per-token nonlinear processing |
| Residual | preserve information and direct gradients |

A block combines them into a shape-preserving update unit that can be stacked repeatedly.

## 2. MiniMind's Pre-Norm decoder block

$$
h = x + \operatorname{Attention}(\operatorname{RMSNorm}(x))
$$

$$
y = h + \operatorname{FFN}(\operatorname{RMSNorm}(h))
$$

Each sub-layer normalizes its input, computes a delta, and adds the delta to a direct path.

## 3. Attention and FFN roles

- **Attention** mixes information across tokens while a causal mask prevents reading the future.
- **FeedForward** applies the same nonlinear transformation independently at every position.

The canonical order is Attention followed by FFN: gather context, then process the gathered representation.

However, the order is not a mathematical law. These nonlinear operators generally do not commute:

$$
\operatorname{FFN}(\operatorname{Attention}(x))
\ne
\operatorname{Attention}(\operatorname{FFN}(x))
$$

Other architectures may use parallel or alternative arrangements.

## 4. Why two norms and two residuals

Attention and FFN are independent sub-layers with different inputs and learned deltas. Each benefits from its own normalized input and identity path.

## 5. External block inputs

- `position_embeddings`: precomputed RoPE cos/sin shared across layers
- `attention_mask`: controls allowed token-to-token attention
- `past_key_value`: the current layer's cached K/V
- `use_cache`: whether to return updated cache during generation

Each layer needs separate K/V because each layer sees a different hidden representation.

## 6. Stacking blocks

```python
for layer, past_key_value in zip(self.layers, past_key_values):
    hidden_states, present = layer(
        hidden_states,
        position_embeddings,
        past_key_value=past_key_value,
        use_cache=use_cache,
        attention_mask=attention_mask,
    )
```

The input and output remain `[batch, seq, hidden_size]`. A final RMSNorm stabilizes the representation before the language-model head.

## 7. What the experiments verify

- Component ablation uses a task requiring both cross-token mixing and per-token nonlinearity.
- The ordering experiment proves that swapping Mixer and FFN changes the function.
- The stack experiment checks shape preservation, causal behavior, and finite gradients.

## Key takeaway

> A Transformer block is a shape-preserving incremental update unit: Attention routes, FFN processes, Norm controls scale, and residuals preserve direct paths.
