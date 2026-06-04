---
title: Transformer Block 教学文档 | minimind从零理解llm训练
description: 理解 Pre-Norm Decoder Block 如何组装 Attention、FeedForward、RMSNorm、RoPE、因果掩码与残差连接。
keywords: Transformer Block原理, Decoder Block, Pre-Norm, 因果注意力
---

# Transformer Block 教学文档

> Transformer Block 的价值不只是“把组件放在一起”，而是定义信息交换、局部处理和直接路径的协作协议。

## 1. 为什么需要 Block

基础组件各自只解决一个问题：

| 组件 | 解决的问题 |
|---|---|
| RMSNorm | 控制子层输入尺度 |
| Attention | 在 token 之间路由信息 |
| RoPE | 把位置信息注入 Q/K |
| FeedForward | 对每个 token 做非线性变换 |
| Residual | 保留信息和直接梯度路径 |

Block 把它们组合成一个输入输出形状一致的“更新单元”。只要 Block 保持 `[batch, seq, hidden]` 接口，模型就可以重复堆叠它。

## 2. MiniMind 的 Pre-Norm Decoder Block

核心公式：

$$
h = x + \operatorname{Attention}(\operatorname{RMSNorm}(x))
$$

$$
y = h + \operatorname{FFN}(\operatorname{RMSNorm}(h))
$$

数据流：

```text
x
├─────────────────────────────┐
└→ RMSNorm → Attention ───────+→ h

h
├─────────────────────────────┐
└→ RMSNorm → FeedForward ─────+→ y
```

每个子层执行三步：

1. 归一化输入
2. 计算一个增量
3. 把增量加回直接路径

## 3. Attention 与 FFN 的分工

### Attention：跨 token 信息交换

位置 $i$ 的输出可以读取位置 $j$ 的内容。Decoder-only 模型使用因果掩码，使当前位置只能读取自己和过去：

$$
j \le i
$$

### FeedForward：逐 token 非线性处理

FFN 对每个位置独立应用同一组参数：

$$
\operatorname{FFN}(x_i)
$$

它不直接混合不同 token，但会处理 Attention 已经汇聚的信息。

### 为什么经典顺序是 Attention → FFN

常见直觉是：

1. Attention 先收集上下文
2. FFN 再处理汇聚后的表示

但这不是数学上的唯一正确顺序。Attention 和 FFN 是非线性、非交换算子：

$$
\operatorname{FFN}(\operatorname{Attention}(x))
\ne
\operatorname{Attention}(\operatorname{FFN}(x))
$$

交换顺序会改变模型函数。部分架构也使用并行分支或不同编排，因此应把经典顺序理解为成熟、有效的设计选择，而不是绝对定律。

## 4. 为什么有两个 Norm 和两条残差

Attention 与 FFN 是两个独立子层：

- 它们的输入分布不同
- 它们学习的增量不同
- 它们都需要自己的直接路径

如果只在整个 Block 外围放一条残差，两个子层会被绑定为一个大函数，失去独立的稳定接口。

## 5. Block 接收哪些外部信息

MiniMind 的 `MiniMindBlock.forward` 除了 `hidden_states`，还接收：

### `position_embeddings`

`MiniMindModel` 预计算 RoPE 的 cos/sin，并传给每一层 Attention。Block 本身不重复计算位置频率。

### `attention_mask`

控制哪些 token 可以互相注意。Decoder-only 模型需要保持因果性；批处理时还可能包含 padding mask。

### `past_key_value`

推理时，每一层都有自己的 K/V Cache。第 3 层的 Key/Value 来自第 3 层表示，不能与其他层共享。

### `use_cache`

决定是否返回当前层更新后的 KV Cache。训练通常不需要，逐 token 生成时需要。

## 6. 多层堆叠

完整核心模型循环：

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

每层输入输出形状相同，但表示逐层更新。最后再执行一次 RMSNorm：

```python
hidden_states = self.norm(hidden_states)
```

Pre-Norm Block 内部的直接路径不会强制每层输出都归一化，因此模型末尾需要最终 Norm 为输出投影提供稳定尺度。

## 7. 形状不变量

设输入形状为：

```text
[batch, seq_len, hidden_size]
```

关键形状：

```text
Attention 输出: [batch, seq_len, hidden_size]
FFN 中间层:    [batch, seq_len, intermediate_size]
FFN 输出:      [batch, seq_len, hidden_size]
Block 输出:    [batch, seq_len, hidden_size]
```

`hidden_size` 是 Block 的公共接口；`intermediate_size` 只存在于 FFN 内部。

## 8. 如何验证组装是否正确

### 组件消融

构造同时需要“跨 token 汇聚”和“逐 token 非线性”的目标，对比完整 Block、去掉 Mixer、去掉 FFN 和去掉残差的结果。

### 顺序实验

对同一输入执行 `Mixer → FFN` 与 `FFN → Mixer`，验证输出不同。实验说明顺序会改变函数，不直接证明某一顺序对所有任务都更好。

### 堆叠与因果性

实现一个最小 Decoder Block，检查：

- 2、8、16 层后形状不变
- 修改未来 token 不会改变过去位置输出
- 输入梯度保持有限且非零

## 9. 核心结论

> Transformer Block 是一个保持形状的增量更新单元：Attention 负责路由，FFN 负责处理，Norm 控制尺度，Residual 保留直接路径。
