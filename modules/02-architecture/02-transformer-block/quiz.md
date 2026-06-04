---
title: Transformer Block 自测题 | minimind从零理解llm训练
description: 检验对 Pre-Norm Decoder Block、双残差、组件顺序、因果性和 KV Cache 的理解。
keywords: Transformer Block自测, Decoder Block测验, KV Cache
---

# Transformer Block 自测题

## 🎮 交互式自测

<script setup>
const quizData = [
  {
    question: 'MiniMindBlock 的核心前向顺序是什么？',
    type: 'single',
    options: [
      { label: 'A', text: 'Attention → Norm → FFN → Norm' },
      { label: 'B', text: 'Norm → Attention → Residual → Norm → FFN → Residual' },
      { label: 'C', text: 'FFN → Attention → Softmax' },
      { label: 'D', text: 'Norm → FFN → Attention → Norm' }
    ],
    correct: [1],
    explanation: '<strong>正确答案：B</strong><br>MiniMind 使用 Pre-Norm，并在 Attention 和 FFN 后各执行一次残差加法。'
  },
  {
    question: '为什么一个 Block 需要两个 RMSNorm？',
    type: 'single',
    options: [
      { label: 'A', text: '因为每个子层都需要稳定自己的输入尺度' },
      { label: 'B', text: '为了改变序列长度' },
      { label: 'C', text: '为了替代残差连接' },
      { label: 'D', text: '为了缓存 Key 和 Value' }
    ],
    correct: [0],
    explanation: '<strong>正确答案：A</strong><br>Attention 和 FFN 是两个独立子层，各自接收归一化输入。'
  },
  {
    question: '关于 Attention → FFN 顺序，哪项最准确？',
    type: 'single',
    options: [
      { label: 'A', text: '这是数学上唯一正确的顺序' },
      { label: 'B', text: '交换顺序一定完全等价' },
      { label: 'C', text: '这是成熟的经典顺序，但其他架构也可能采用不同编排' },
      { label: 'D', text: '顺序只影响推理速度，不影响函数' }
    ],
    correct: [2],
    explanation: '<strong>正确答案：C</strong><br>Attention 和 FFN 通常不交换，顺序会改变函数；但经典顺序并非唯一可行设计。'
  },
  {
    question: '为什么每层需要独立的 KV Cache？',
    type: 'single',
    options: [
      { label: 'A', text: '因为每层隐藏表示和 K/V 投影不同' },
      { label: 'B', text: '因为每层序列长度不同' },
      { label: 'C', text: '因为 RMSNorm 会删除 Cache' },
      { label: 'D', text: '因为 FFN 需要读取未来 token' }
    ],
    correct: [0],
    explanation: '<strong>正确答案：A</strong><br>不同层处理不同隐藏表示，因此产生不同的 Key 和 Value。'
  },
  {
    question: 'Decoder Block 的因果性要求是什么？',
    type: 'single',
    options: [
      { label: 'A', text: '过去位置可以读取未来 token' },
      { label: 'B', text: '修改未来 token 不应改变过去位置输出' },
      { label: 'C', text: '所有位置输出必须相同' },
      { label: 'D', text: 'FFN 必须混合所有 token' }
    ],
    correct: [1],
    explanation: '<strong>正确答案：B</strong><br>因果掩码阻止当前位置读取未来信息。'
  },
  {
    question: '为什么 FFN 可以在内部扩张，但 Block 输出仍是 hidden_size？',
    type: 'single',
    options: [
      { label: 'A', text: '因为 down_proj 压缩回 hidden_size，以保持残差与层间接口' },
      { label: 'B', text: '因为 Attention 会删除多余维度' },
      { label: 'C', text: '因为 RoPE 只支持 hidden_size' },
      { label: 'D', text: '因为 Cache 只能存一个维度' }
    ],
    correct: [0],
    explanation: '<strong>正确答案：A</strong><br>Block 必须保持形状，才能执行残差加法并重复堆叠。'
  }
]
</script>

<InteractiveQuiz :questions="quizData" quiz-id="transformer-block" />

## 综合题

### Q7：`post_attention_layernorm` 为什么仍属于 Pre-Norm？

<details>
<summary>查看参考答案</summary>

它位于 Attention 残差之后，但在 MLP 子层之前执行。Pre-Norm / Post-Norm 描述的是 Norm 相对子层的位置；对 MLP 而言，这个 Norm 仍然是 Pre-Norm。

</details>

### Q8：为什么最终还要执行一次 `self.norm(hidden_states)`？

<details>
<summary>查看参考答案</summary>

Pre-Norm Block 的直接路径可以跨层累计，Block 输出本身不会被强制归一化。最终 RMSNorm 为 `lm_head` 提供稳定尺度。

</details>

### Q9：如何测试 Decoder Block 是否泄露未来信息？

<details>
<summary>查看参考答案</summary>

复制同一个输入，只修改最后一个未来 token；分别运行 Block 后，比较最后位置之前的输出。如果因果掩码正确，过去位置输出差异应接近 0。

</details>
