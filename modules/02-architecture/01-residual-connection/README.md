---
title: Residual Connection（残差连接）模块
description: 通过可执行实验理解残差连接为什么能保留信息、改善梯度流，并帮助深层 Transformer 稳定训练。
---

# 01. Residual Connection（残差连接）

> 让深层网络学习“增量”，同时为信息和梯度保留一条直接通路。

## 学习目标

完成本模块后，你将能够：

- 解释 `y = x + F(x)` 与普通堆叠 `y = F(x)` 的差异
- 用雅可比矩阵解释残差连接如何改善梯度流
- 区分“保留恒等路径”和“彻底消除梯度问题”
- 对照 MiniMind 源码找到两条残差路径
- 运行实验验证训练、梯度流和深度扩展差异

## 学习路径

1. 阅读 [教学文档](./teaching.md)
2. 运行 3 个对照实验
3. 阅读 [MiniMind 源码导读](./code_guide.md)
4. 完成 [自测题](./quiz.md)

```bash
cd modules/02-architecture/01-residual-connection/experiments
bash run_all.sh
```

## 实验列表

| 实验 | 核心问题 | 验证方式 |
|---|---|---|
| `exp1_with_vs_without.py` | 残差网络是否更容易学习接近恒等映射的任务？ | 对比相同深度网络的最终 MSE |
| `exp2_gradient_flow.py` | 梯度能否从深层稳定传回输入端？ | 对比首尾梯度比例 |
| `exp3_depth_scaling.py` | 网络变深时是否还能保留输入信息？ | 对比输出与输入的余弦相似度 |

实验结果会保存到 `experiments/results/*.json`。脚本包含确定性断言，结论不成立时会以非零状态退出。

## 核心公式

普通子层：

$$
y = F(x), \qquad \frac{\partial y}{\partial x} = J_F
$$

残差子层：

$$
y = x + F(x), \qquad \frac{\partial y}{\partial x} = I + J_F
$$

即使残差分支的局部梯度很小，恒等路径仍提供了一个直接项 $I$。

## MiniMind 中的位置

```python
residual = hidden_states
hidden_states, present_key_value = self.self_attn(
    self.input_layernorm(hidden_states),
    ...
)
hidden_states += residual

hidden_states = hidden_states + self.mlp(
    self.post_attention_layernorm(hidden_states)
)
```

Attention 和 MLP 各有一条残差连接，因此每个 `MiniMindBlock` 有两条直接路径。

## 完成检查

- [ ] 能写出残差连接的前向和反向公式
- [ ] 能解释为什么残差分支初始接近 0 不是坏事
- [ ] 能说明残差连接为什么要求输入输出形状一致
- [ ] 能在 `MiniMindBlock.forward` 中指出两条残差路径
- [ ] `bash experiments/run_all.sh` 全部通过

## 下一步

完成本模块后，继续学习 [02. Transformer Block](../02-transformer-block/)。
