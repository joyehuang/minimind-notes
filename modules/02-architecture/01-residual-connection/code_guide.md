---
title: Residual Connection 源码导读 | minimind从零理解llm训练
description: 逐行理解 MiniMindBlock 中 Attention 和 MLP 的两条残差连接，以及它们与 Pre-Norm 的配合。
keywords: MiniMindBlock源码, 残差连接代码, Pre-Norm代码
---

# Residual Connection 源码导读

> 对照当前 `model/model_minimind.py`，理解 MiniMind 如何实现两条残差路径。

## 1. 代码位置

残差连接位于 `MiniMindBlock.forward`，当前源码约在：

- `model/model_minimind.py:441-476`

核心代码：

```python
def forward(
    self,
    hidden_states,
    position_embeddings,
    past_key_value=None,
    use_cache=False,
    attention_mask=None,
):
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

## 2. 第一条残差：Attention

```python
residual = hidden_states
```

这一步保存 Attention 之前的表示。赋值本身不会复制张量数据，而是让 `residual` 引用当前张量对象。

```python
hidden_states, present_key_value = self.self_attn(
    self.input_layernorm(hidden_states),
    ...
)
```

数据流是：

```text
hidden_states → RMSNorm → Attention → attention_output
```

注意 `self.input_layernorm(hidden_states)` 只进入 Attention 分支，保存的 `residual` 没有经过 Norm。这正是 Pre-Norm 的直接路径。

```python
hidden_states += residual
```

Attention 输出与原始输入相加：

$$
h = x + \operatorname{Attention}(\operatorname{RMSNorm}(x))
$$

此时 `hidden_states` 已经是第一条残差之后的状态。

## 3. 第二条残差：MLP

```python
hidden_states = hidden_states + self.mlp(
    self.post_attention_layernorm(hidden_states)
)
```

这一行可拆成：

```python
residual = hidden_states
normed = self.post_attention_layernorm(hidden_states)
delta = self.mlp(normed)
hidden_states = residual + delta
```

源码没有显式写第二个 `residual` 变量，但加法左侧的 `hidden_states` 就是这条直接路径：

$$
y = h + \operatorname{MLP}(\operatorname{RMSNorm}(h))
$$

## 4. 完整数据流

```text
x
├──────────────────────────────────────────────┐
└→ input_layernorm → self_attn → attention Δ ─+→ h

h
├──────────────────────────────────────────────┐
└→ post_attention_layernorm → mlp → mlp Δ ────+→ y
```

两个 Norm 名称容易造成误解：

- `input_layernorm`：Attention 之前的 RMSNorm
- `post_attention_layernorm`：位于 Attention 残差之后、MLP 之前

`post_attention_layernorm` 并不是整个 Block 的 Post-Norm。整个 Block 仍是 Pre-Norm 架构。

## 5. 为什么 Attention 和 MLP 输出维度不变

残差加法要求形状一致：

```python
attention_output.shape == residual.shape
mlp_output.shape == hidden_states.shape
```

MiniMind 的核心维度流：

```text
Attention:   hidden_size → hidden_size
FeedForward: hidden_size → intermediate_size → hidden_size
```

FFN 中间可以扩张，但 `down_proj` 必须压缩回 `hidden_size`，否则无法与残差路径相加。

## 6. `+=` 是否会影响自动求导

这里的 `hidden_states += residual` 是对 Attention 返回的新张量执行原地加法，而 `residual` 保存的是 Attention 输入。当前实现可正常自动求导。

不过在自定义网络中，原地操作仍要谨慎：

- 如果某个张量的旧值还被反向传播需要，原地修改可能触发 autograd 错误。
- 更保守、易读的教学实现通常写成 `hidden_states = hidden_states + residual`。

## 7. 最小可运行实现

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

这个最小版本保留了 MiniMind Block 的关键结构：

1. 子层前归一化
2. 子层输入输出维度相同
3. 输出与原输入相加

## 8. 调试检查清单

```python
assert delta.shape == residual.shape
assert torch.isfinite(delta).all()
assert torch.isfinite(output).all()
```

遇到深层网络不稳定时，进一步检查：

- 每层残差分支输出的 RMS 是否持续增长
- 输入端和输出端梯度范数的比例
- 学习率与初始化尺度
- 是否错误地把 Norm 放到了直接路径上
