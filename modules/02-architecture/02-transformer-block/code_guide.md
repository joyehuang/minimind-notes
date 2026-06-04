---
title: Transformer Block 源码导读 | minimind从零理解llm训练
description: 逐行理解 MiniMindBlock 的初始化、前向传播、多层堆叠、RoPE 与 KV Cache 接口。
keywords: MiniMindBlock源码, Transformer Block代码, KV Cache, Pre-Norm
---

# Transformer Block 源码导读

> 对照当前 `model/model_minimind.py` 理解单个 Block 和完整模型如何连接。

## 1. `MiniMindBlock.__init__`

当前源码约在 `model/model_minimind.py:441-455`：

```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

### 组件映射

| 属性 | 作用 |
|---|---|
| `self.self_attn` | 因果自注意力，内部应用 RoPE 和 GQA |
| `self.input_layernorm` | Attention 前的 RMSNorm |
| `self.post_attention_layernorm` | MLP 前的 RMSNorm |
| `self.mlp` | SwiGLU FFN 或 MoE FFN |
| `self.layer_id` | 保存层编号，便于层级元数据和扩展 |

`post_attention_layernorm` 的名字表示它位于 Attention 之后，但它实际在 MLP 之前执行，因此整个 Block 仍是 Pre-Norm。

## 2. `MiniMindBlock.forward`

当前源码约在 `model/model_minimind.py:456-476`：

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

逐步展开：

```python
# 子层 1：Attention
attention_input = self.input_layernorm(hidden_states)
attention_delta, present_key_value = self.self_attn(
    attention_input,
    position_embeddings,
    past_key_value,
    use_cache,
    attention_mask,
)
hidden_states = hidden_states + attention_delta

# 子层 2：FFN / MoE
mlp_input = self.post_attention_layernorm(hidden_states)
mlp_delta = self.mlp(mlp_input)
hidden_states = hidden_states + mlp_delta
```

## 3. 为什么返回二元组

```python
return hidden_states, present_key_value
```

- `hidden_states`：传给下一层
- `present_key_value`：推理时保存当前层新的 K/V Cache

即使训练时不使用 Cache，统一返回接口也让模型循环更简单。

## 4. RoPE 为什么由模型层预计算

`MiniMindModel.__init__`：

```python
freqs_cos, freqs_sin = precompute_freqs_cis(
    dim=config.hidden_size // config.num_attention_heads,
    end=config.max_position_embeddings,
    rope_base=config.rope_theta,
    rope_scaling=config.rope_scaling,
)
```

`MiniMindModel.forward` 根据当前序列范围切片：

```python
position_embeddings = (
    self.freqs_cos[start_pos : start_pos + seq_length],
    self.freqs_sin[start_pos : start_pos + seq_length],
)
```

所有 Block 使用同一位置范围，但每层在自己的 Q/K 上应用旋转。这样避免每层重复计算 cos/sin。

## 5. 多层堆叠

当前源码约在 `model/model_minimind.py:526-539`：

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

### 为什么每层 Cache 独立

每个 Block 的输入表示不同，所以每层投影出的 K/V 也不同：

```text
layer 0 cache ≠ layer 1 cache ≠ ... ≠ layer N cache
```

`past_key_values` 因此是按层组织的列表。

## 6. MoE 不改变 Block 接口

```python
self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

普通 FFN 与 MoE 内部计算不同，但都必须：

```text
[batch, seq, hidden_size] → [batch, seq, hidden_size]
```

这说明良好的 Block 接口可以隐藏内部实现变化。

## 7. 最小 Decoder Block

```python
class DecoderBlock(nn.Module):
    def __init__(self, dim, attention, ffn, norm):
        super().__init__()
        self.attention = attention
        self.ffn = ffn
        self.attention_norm = norm(dim)
        self.ffn_norm = norm(dim)

    def forward(self, x, mask=None):
        x = x + self.attention(self.attention_norm(x), mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

## 8. 调试检查清单

```python
assert attention_delta.shape == hidden_states.shape
assert mlp_delta.shape == hidden_states.shape
assert output.shape == hidden_states.shape
assert torch.isfinite(output).all()
```

Decoder Block 还应验证因果性：

1. 保持过去 token 不变
2. 只修改未来 token
3. 过去位置的输出必须不变
