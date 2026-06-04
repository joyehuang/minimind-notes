---
title: Transformer Block（Transformer 块）模块
description: 将 RMSNorm、Attention、FeedForward 和残差连接组装为完整的 MiniMind Transformer Block。
---

# 02. Transformer Block（Transformer 块）

> 单个组件解决局部问题，Transformer Block 定义它们如何协作。

## 学习目标

完成本模块后，你将能够：

- 从零实现一个 Pre-Norm Decoder Block
- 解释 Attention、FFN、Norm 和双残差的分工
- 说明经典 `Attention → FFN` 顺序为何常见，以及它并非唯一可行顺序
- 理解 `position_embeddings`、`attention_mask` 和 KV Cache 如何进入 Block
- 堆叠多个 Block，并验证形状、因果性和梯度

## 学习路径

1. 阅读 [教学文档](./teaching.md)
2. 运行 3 个组装实验
3. 阅读 [MiniMind 源码导读](./code_guide.md)
4. 完成 [自测题](./quiz.md)

```bash
cd modules/02-architecture/02-transformer-block/experiments
bash run_all.sh
```

## 实验列表

| 实验 | 核心问题 | 验证方式 |
|---|---|---|
| `exp1_component_ablation.py` | Attention/Mixer、FFN、残差各自贡献什么？ | 对比合成任务最终 MSE |
| `exp2_order_matters.py` | 交换组件顺序是否还是同一个函数？ | 测量两种顺序输出差异 |
| `exp3_stack_and_causality.py` | 完整 Decoder Block 能否安全堆叠？ | 验证形状、因果性和输入梯度 |

## 核心结构

```python
h = x + attention(attention_norm(x))
out = h + feed_forward(ffn_norm(h))
```

对应 MiniMind：

```text
x → RMSNorm → Attention → +x → RMSNorm → FeedForward → +h → output
```

## 完成检查

- [ ] 能独立实现 Pre-Norm Transformer Block
- [ ] 能解释为什么 Block 中有两个 Norm 和两条残差
- [ ] 能说明 Attention 与 FFN 顺序会改变函数，但不应写成绝对定律
- [ ] 能解释每层独立 KV Cache 的原因
- [ ] `bash experiments/run_all.sh` 全部通过

## 下一步

返回 [架构组装总览](../)，继续学习完整模型和训练流程。
