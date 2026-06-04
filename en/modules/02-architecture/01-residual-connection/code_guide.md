---
title: Residual Connection Code Walkthrough | MiniMind LLM Training
description: Walk through the two residual paths in MiniMindBlock and their relationship with Pre-Norm.
keywords: MiniMindBlock source, residual connection code, Pre-Norm
---

# Residual Connection Code Walkthrough

The relevant implementation is in `model/model_minimind.py:441-476`.

```python
residual = hidden_states
hidden_states, present_key_value = self.self_attn(
    self.input_layernorm(hidden_states),
    position_embeddings,
    past_key_value,
    use_cache,
    attention_mask,
)
hidden_states += residual
hidden_states = hidden_states + self.mlp(
    self.post_attention_layernorm(hidden_states)
)
```

## 1. Attention residual

```python
residual = hidden_states
```

This preserves the state before Attention. The saved path does not pass through normalization.

```python
hidden_states, present_key_value = self.self_attn(
    self.input_layernorm(hidden_states),
    ...
)
```

Only the Attention branch receives normalized input:

```text
hidden_states → RMSNorm → Attention → attention delta
```

```python
hidden_states += residual
```

This implements:

$$
h = x + \operatorname{Attention}(\operatorname{RMSNorm}(x))
$$

## 2. MLP residual

```python
hidden_states = hidden_states + self.mlp(
    self.post_attention_layernorm(hidden_states)
)
```

Expanded for clarity:

```python
residual = hidden_states
normed = self.post_attention_layernorm(hidden_states)
delta = self.mlp(normed)
hidden_states = residual + delta
```

The source does not name a second `residual`, but the left-hand `hidden_states` is the direct path.

## 3. Full data flow

```text
x
├────────────────────────────────────────┐
└→ input_layernorm → self_attn → delta ─+→ h

h
├────────────────────────────────────────┐
└→ post_attention_layernorm → mlp → delta+→ y
```

Despite its name, `post_attention_layernorm` is applied before the MLP. The block is still Pre-Norm.

## 4. Shape contract

Residual addition requires:

```python
attention_output.shape == residual.shape
mlp_output.shape == hidden_states.shape
```

The MLP may expand internally, but `down_proj` returns to `hidden_size`.

## 5. Minimal implementation

```python
import torch.nn as nn


class PreNormResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.branch = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.branch(self.norm(x))
```

## 6. Debugging checks

```python
assert delta.shape == residual.shape
assert torch.isfinite(delta).all()
assert torch.isfinite(output).all()
```

For unstable deep models, inspect branch RMS values, input/output gradient ratios, initialization scale, and learning rate.
