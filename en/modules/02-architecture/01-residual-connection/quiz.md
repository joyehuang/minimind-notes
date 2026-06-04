---
title: Residual Connection Quiz | MiniMind LLM Training
description: Test your understanding of residual formulas, gradient flow, Pre-Norm, and the MiniMind implementation.
keywords: residual connection quiz, Transformer quiz, gradient flow
---

# Residual Connection Quiz

## Interactive quiz

<script setup>
const quizData = [
  {
    question: 'What is the standard residual form?',
    type: 'single',
    options: [
      { label: 'A', text: 'y = F(x)' },
      { label: 'B', text: 'y = x + F(x)' },
      { label: 'C', text: 'y = x × F(x)' },
      { label: 'D', text: 'y = Norm(F(x))' }
    ],
    correct: [1],
    explanation: '<strong>Correct answer: B</strong><br>The branch learns F(x), while the identity path preserves x.'
  },
  {
    question: 'For y = x + F(x), what is the local derivative with respect to x?',
    type: 'single',
    options: [
      { label: 'A', text: 'J_F' },
      { label: 'B', text: 'I + J_F' },
      { label: 'C', text: 'I × J_F' },
      { label: 'D', text: '0' }
    ],
    correct: [1],
    explanation: '<strong>Correct answer: B</strong><br>The identity path contributes I and the branch contributes J_F.'
  },
  {
    question: 'Why can a near-zero residual branch be a useful starting point?',
    type: 'single',
    options: [
      { label: 'A', text: 'The network no longer needs training' },
      { label: 'B', text: 'The layer still approximates the identity mapping' },
      { label: 'C', text: 'All gradients must also be zero' },
      { label: 'D', text: 'The input can be deleted' }
    ],
    correct: [1],
    explanation: '<strong>Correct answer: B</strong><br>If F(x)≈0, then x+F(x)≈x.'
  },
  {
    question: 'How many main residual paths are in one MiniMindBlock?',
    type: 'single',
    options: [
      { label: 'A', text: '0' },
      { label: 'B', text: '1' },
      { label: 'C', text: '2' },
      { label: 'D', text: '8' }
    ],
    correct: [2],
    explanation: '<strong>Correct answer: C</strong><br>Attention and the MLP each have their own residual path.'
  },
  {
    question: 'Why do Attention and the FFN both return hidden_size?',
    type: 'single',
    options: [
      { label: 'A', text: 'To apply Softmax' },
      { label: 'B', text: 'To support elementwise addition with the identity path' },
      { label: 'C', text: 'To reduce token count' },
      { label: 'D', text: 'To generate position encodings' }
    ],
    correct: [1],
    explanation: '<strong>Correct answer: B</strong><br>Residual addition requires matching shapes.'
  },
  {
    question: 'Which statement about residual connections is most accurate?',
    type: 'single',
    options: [
      { label: 'A', text: 'They guarantee gradients can never vanish or explode' },
      { label: 'B', text: 'They improve direct paths but still require normalization and controlled scale' },
      { label: 'C', text: 'They only work in CNNs' },
      { label: 'D', text: 'They force every layer to learn the same function' }
    ],
    correct: [1],
    explanation: '<strong>Correct answer: B</strong><br>Residuals are important, but they are not a universal stability guarantee.'
  }
]
</script>

<InteractiveQuiz :questions="quizData" quiz-id="residual-connection-en" />

## Comprehensive questions

### Q7: Why is there no explicit second `residual` variable in MiniMind?

<details>
<summary>Show reference answer</summary>

In `hidden_states = hidden_states + self.mlp(...)`, the left-hand `hidden_states` is already the direct path. It is equivalent to saving it in a separate variable before computing the MLP delta.

</details>

### Q8: When is a projected shortcut required?

<details>
<summary>Show reference answer</summary>

When the branch output shape differs from the input shape. The shortcut must project the input so `projection(x) + F(x)` has matching shapes.

</details>
