---
title: Transformer Block Code Walkthrough | MiniMind LLM Training
description: Walk through MiniMindBlock initialization, forward flow, layer stacking, RoPE, and KV Cache interfaces.
keywords: MiniMindBlock source, Transformer block code, KV Cache, Pre-Norm
---

# Transformer Block Code Walkthrough

## 1. `MiniMindBlock.__init__`

The current implementation is around `model/model_minimind.py:441-455`:

```python
self.self_attn = Attention(config)
self.layer_id = layer_id
self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.post_attention_layernorm = RMSNorm(
    config.hidden_size, eps=config.rms_norm_eps
)
self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

`post_attention_layernorm` is after the Attention residual but before the MLP. The block is still Pre-Norm.

## 2. Forward flow

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
return hidden_states, present_key_value
```

Expanded:

```python
attention_delta, present_key_value = self.self_attn(
    self.input_layernorm(hidden_states),
    ...
)
hidden_states = hidden_states + attention_delta

mlp_delta = self.mlp(self.post_attention_layernorm(hidden_states))
hidden_states = hidden_states + mlp_delta
```

## 3. Why return a tuple

- `hidden_states` goes to the next block.
- `present_key_value` stores the current layer's updated K/V during generation.

Each layer owns a different cache because each layer projects a different hidden representation.

## 4. Precomputed RoPE

`MiniMindModel` precomputes RoPE cos/sin, slices the current sequence range, and passes the same position range to every layer. Each layer applies it to its own Q/K.

## 5. Layer stacking

```python
for layer_idx, (layer, past_key_value) in enumerate(
    zip(self.layers, past_key_values)
):
    hidden_states, present = layer(
        hidden_states,
        position_embeddings,
        past_key_value=past_key_value,
        use_cache=use_cache,
        attention_mask=attention_mask,
    )
    presents.append(present)

hidden_states = self.norm(hidden_states)
```

The final Norm stabilizes accumulated Pre-Norm block outputs before projection to vocabulary logits.

## 6. MoE preserves the block contract

```python
self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

Both implementations preserve:

```text
[batch, seq, hidden_size] → [batch, seq, hidden_size]
```

## 7. Debugging checklist

```python
assert attention_delta.shape == hidden_states.shape
assert mlp_delta.shape == hidden_states.shape
assert output.shape == hidden_states.shape
assert torch.isfinite(output).all()
```

For a decoder block, also test causality by changing a future token and verifying that past outputs do not change.
