---
title: Residual Connection（残差连接）自测题 | minimind从零理解llm训练
description: 检验对残差公式、梯度流、Pre-Norm 和 MiniMind 源码实现的理解。
keywords: 残差连接自测, Transformer测验, 梯度流
---

# Residual Connection（残差连接）自测题

## 🎮 交互式自测

<script setup>
const quizData = [
  {
    question: '残差连接的标准形式是什么？',
    type: 'single',
    options: [
      { label: 'A', text: 'y = F(x)' },
      { label: 'B', text: 'y = x + F(x)' },
      { label: 'C', text: 'y = x × F(x)' },
      { label: 'D', text: 'y = Norm(F(x))' }
    ],
    correct: [1],
    explanation: '<strong>正确答案：B</strong><br>残差分支学习增量 F(x)，恒等路径直接保留 x。'
  },
  {
    question: '对 y = x + F(x)，局部梯度 ∂y/∂x 是什么？',
    type: 'single',
    options: [
      { label: 'A', text: 'J_F' },
      { label: 'B', text: 'I + J_F' },
      { label: 'C', text: 'I × J_F' },
      { label: 'D', text: '0' }
    ],
    correct: [1],
    explanation: '<strong>正确答案：B</strong><br>恒等路径贡献 I，残差分支贡献 J_F。'
  },
  {
    question: '为什么残差分支初始接近 0 通常是可接受的？',
    type: 'single',
    options: [
      { label: 'A', text: '因为网络不需要训练' },
      { label: 'B', text: '因为输出仍可近似等于输入' },
      { label: 'C', text: '因为梯度也必须为 0' },
      { label: 'D', text: '因为可以删除输入' }
    ],
    correct: [1],
    explanation: '<strong>正确答案：B</strong><br>当 F(x)≈0 时，y=x+F(x)≈x，网络至少保留恒等映射。'
  },
  {
    question: 'MiniMind 的一个 MiniMindBlock 有几条主要残差连接？',
    type: 'single',
    options: [
      { label: 'A', text: '0 条' },
      { label: 'B', text: '1 条' },
      { label: 'C', text: '2 条' },
      { label: 'D', text: '8 条' }
    ],
    correct: [2],
    explanation: '<strong>正确答案：C</strong><br>Attention 和 MLP 子层各有一条残差连接。'
  },
  {
    question: '为什么 Attention 和 FFN 最终都要输出 hidden_size？',
    type: 'single',
    options: [
      { label: 'A', text: '为了使用 Softmax' },
      { label: 'B', text: '为了与残差路径进行逐元素加法' },
      { label: 'C', text: '为了减少 token 数量' },
      { label: 'D', text: '为了生成位置编码' }
    ],
    correct: [1],
    explanation: '<strong>正确答案：B</strong><br>残差加法要求两条路径形状一致。'
  },
  {
    question: '关于残差连接，哪项说法最准确？',
    type: 'single',
    options: [
      { label: 'A', text: '使用残差后永远不会梯度爆炸或消失' },
      { label: 'B', text: '残差连接改善直接信息和梯度路径，但仍需归一化与合理尺度' },
      { label: 'C', text: '残差连接只在 CNN 中有效' },
      { label: 'D', text: '残差连接会让所有层学到相同函数' }
    ],
    correct: [1],
    explanation: '<strong>正确答案：B</strong><br>残差连接很重要，但不是对任意深度和超参数的稳定性保证。'
  }
]
</script>

<InteractiveQuiz :questions="quizData" quiz-id="residual-connection" />

## 综合题

### Q7：为什么 MiniMind 的第二条残差没有显式保存 `residual`？

<details>
<summary>查看参考答案</summary>

源码写成：

```python
hidden_states = hidden_states + self.mlp(
    self.post_attention_layernorm(hidden_states)
)
```

加法左侧的 `hidden_states` 本身就是直接路径。展开后等价于先保存 `residual = hidden_states`，再计算 `residual + mlp(norm(residual))`。

</details>

### Q8：如果残差分支输出 RMS 随深度持续增大，会发生什么？

<details>
<summary>查看参考答案</summary>

残差连接会累计每层增量：

$$
x_L = x_0 + \sum_l F_l(x_l)
$$

如果每层增量过大，激活尺度仍可能不断增长，导致数值不稳定或梯度爆炸。应检查初始化、学习率、归一化、残差缩放和梯度裁剪。

</details>

### Q9：什么时候需要对残差路径做投影？

<details>
<summary>查看参考答案</summary>

当 $F(x)$ 的输出形状与 $x$ 不一致时，不能直接相加，需要使用 `projection(x) + F(x)` 对齐形状。Transformer Block 通常保持 `hidden_size` 不变，因此一般不需要投影。

</details>
