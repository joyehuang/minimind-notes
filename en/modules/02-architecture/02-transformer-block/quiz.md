---
title: Transformer Block Quiz | MiniMind LLM Training
description: Test your understanding of Pre-Norm decoder blocks, double residuals, component order, causality, and KV Cache.
keywords: Transformer block quiz, decoder block quiz, KV Cache
---

# Transformer Block Quiz

## Interactive quiz

<script setup>
const quizData = [
  {
    question: 'What is the core MiniMindBlock forward order?',
    type: 'single',
    options: [
      { label: 'A', text: 'Attention → Norm → FFN → Norm' },
      { label: 'B', text: 'Norm → Attention → Residual → Norm → FFN → Residual' },
      { label: 'C', text: 'FFN → Attention → Softmax' },
      { label: 'D', text: 'Norm → FFN → Attention → Norm' }
    ],
    correct: [1],
    explanation: '<strong>Correct answer: B</strong><br>MiniMind uses Pre-Norm and one residual addition after each sub-layer.'
  },
  {
    question: 'Why does one block need two RMSNorm layers?',
    type: 'single',
    options: [
      { label: 'A', text: 'Each sub-layer needs controlled input scale' },
      { label: 'B', text: 'To change sequence length' },
      { label: 'C', text: 'To replace residuals' },
      { label: 'D', text: 'To cache Key and Value' }
    ],
    correct: [0],
    explanation: '<strong>Correct answer: A</strong><br>Attention and FFN are independent sub-layers with separate normalized inputs.'
  },
  {
    question: 'Which statement about Attention → FFN is most accurate?',
    type: 'single',
    options: [
      { label: 'A', text: 'It is the only mathematically valid order' },
      { label: 'B', text: 'Swapping the order is always equivalent' },
      { label: 'C', text: 'It is canonical, but other architectures can use different arrangements' },
      { label: 'D', text: 'Order only changes speed, not the function' }
    ],
    correct: [2],
    explanation: '<strong>Correct answer: C</strong><br>The operators generally do not commute, but the canonical order is not the only possible design.'
  },
  {
    question: 'Why does every layer need a separate KV Cache?',
    type: 'single',
    options: [
      { label: 'A', text: 'Each layer has different hidden states and K/V projections' },
      { label: 'B', text: 'Each layer has a different sequence length' },
      { label: 'C', text: 'RMSNorm deletes caches' },
      { label: 'D', text: 'FFN reads future tokens' }
    ],
    correct: [0],
    explanation: '<strong>Correct answer: A</strong><br>Different layer representations produce different Key and Value tensors.'
  },
  {
    question: 'What does decoder causality require?',
    type: 'single',
    options: [
      { label: 'A', text: 'Past positions can read future tokens' },
      { label: 'B', text: 'Changing a future token must not change past outputs' },
      { label: 'C', text: 'All position outputs must be identical' },
      { label: 'D', text: 'FFN must mix all tokens' }
    ],
    correct: [1],
    explanation: '<strong>Correct answer: B</strong><br>A causal mask prevents information from flowing backward from future tokens.'
  },
  {
    question: 'Why can FFN expand internally while the block still returns hidden_size?',
    type: 'single',
    options: [
      { label: 'A', text: 'down_proj returns to hidden_size for residual and stacking contracts' },
      { label: 'B', text: 'Attention deletes extra dimensions' },
      { label: 'C', text: 'RoPE only supports hidden_size' },
      { label: 'D', text: 'Cache stores only one dimension' }
    ],
    correct: [0],
    explanation: '<strong>Correct answer: A</strong><br>The block must preserve shape for residual addition and repeated stacking.'
  }
]
</script>

<InteractiveQuiz :questions="quizData" quiz-id="transformer-block-en" />

## Comprehensive questions

### Q7: Why is `post_attention_layernorm` still Pre-Norm?

<details>
<summary>Show reference answer</summary>

It is after the Attention residual, but before the MLP sub-layer. Pre-Norm/Post-Norm describes the Norm position relative to a sub-layer.

</details>

### Q8: How do you test decoder causality?

<details>
<summary>Show reference answer</summary>

Copy an input, change only a future token, run both versions, and compare outputs at earlier positions. With a correct causal mask, past outputs remain unchanged.

</details>
