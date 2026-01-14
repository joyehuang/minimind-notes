---
title: Normalization Quiz | MiniMind LLM Training
description: Quiz for the Normalization module. Check understanding of gradient vanishing, RMSNorm, and Pre-LN vs Post-LN.
keywords: Normalization quiz, RMSNorm quiz, Pre-LN, Post-LN, Transformer quiz
---

# Normalization Quiz

> Check your understanding of core concepts in the Normalization module

---

## 🧠 Quiz (multiple choice)

<script setup>
const quizData = [
  {
    question: 'What is the main cause of gradient vanishing in deep networks?',
    type: 'single',
    options: [
      { label: 'A', text: 'Learning rate is too high' },
      { label: 'B', text: 'Activation scale shrinks across layers so signals vanish' },
      { label: 'C', text: 'The model has too many parameters' },
      { label: 'D', text: 'The dataset is too small' }
    ],
    correct: [1],
    explanation: `
      <strong>Correct answer: B</strong><br><br>
      Gradient vanishing happens when activation magnitudes shrink across layers, so gradients approach zero.
      <ul>
        <li>As activations decay, signals disappear and gradients vanish.</li>
        <li>This blocks learning in deep networks.</li>
        <li>Normalization stabilizes the activation scale and prevents this.</li>
      </ul>
    `
  },
  {
    question: 'What is the key difference between RMSNorm and LayerNorm?',
    type: 'single',
    options: [
      { label: 'A', text: 'RMSNorm uses more trainable parameters' },
      { label: 'B', text: 'RMSNorm does not subtract the mean; it normalizes by RMS only' },
      { label: 'C', text: 'RMSNorm is only used in Transformers' },
      { label: 'D', text: 'RMSNorm is slower than LayerNorm' }
    ],
    correct: [1],
    explanation: `
      <strong>Correct answer: B</strong><br><br>
      <ul>
        <li>LayerNorm: <code>(x - mean) / std</code></li>
        <li>RMSNorm: <code>x / RMS(x)</code> (no mean subtraction)</li>
        <li>RMSNorm is simpler and often faster (7–64% speedup).</li>
      </ul>
    `
  },
  {
    question: 'Why is Pre-LN more stable than Post-LN?',
    type: 'single',
    options: [
      { label: 'A', text: 'It has better GPU utilization' },
      { label: 'B', text: 'It reduces parameter count' },
      { label: 'C', text: 'It keeps a clean residual path and stabilizes gradients' },
      { label: 'D', text: 'It removes normalization completely' }
    ],
    correct: [2],
    explanation: `
      <strong>Correct answer: C</strong><br><br>
      <strong>Post-LN</strong>: <code>x = LayerNorm(x + Attention(x))</code>
      <ul>
        <li>Residual path is affected by normalization, gradients can be blocked.</li>
      </ul>
      <strong>Pre-LN</strong>: <code>x = x + Attention(Norm(x))</code>
      <ul>
        <li>Residual path stays clean, gradients flow more directly.</li>
        <li>More stable for deep stacks (&gt;12 layers).</li>
        <li>More tolerant to learning rate.</li>
      </ul>
    `
  },
  {
    question: 'In MiniMind, how many RMSNorm layers does each TransformerBlock contain?',
    type: 'single',
    options: [
      { label: 'A', text: '1' },
      { label: 'B', text: '2' },
      { label: 'C', text: '4' },
      { label: 'D', text: '8' }
    ],
    correct: [1],
    explanation: `
      <strong>Correct answer: B</strong><br><br>
      Each TransformerBlock has <strong>two RMSNorms</strong>:
      <ol>
        <li><strong>attention_norm</strong>: before Attention</li>
        <li><strong>ffn_norm</strong>: before FeedForward</li>
      </ol>
      Flow: <code>x → Norm #1 → Attention → + residual → Norm #2 → FFN → + residual → output</code><br><br>
      MiniMind-small has 8 blocks → 16 RMSNorms total.
    `
  },
  {
    question: 'Why does RMSNorm use .float() and .type_as(x) in forward?',
    type: 'single',
    options: [
      { label: 'A', text: 'To speed up the CPU' },
      { label: 'B', text: 'To reduce parameter count' },
      { label: 'C', text: 'To avoid FP16/BF16 underflow during normalization' },
      { label: 'D', text: 'To avoid PyTorch errors' }
    ],
    correct: [2],
    explanation: `
      <strong>Correct answer: C</strong><br><br>
      FP16/BF16 can underflow for small values during normalization.
      <ul>
        <li><code>.float()</code> converts to FP32 for stable computation</li>
        <li><code>.type_as(x)</code> casts back to the original dtype</li>
      </ul>
      Flow: input (BF16) → FP32 normalization → cast back to BF16.
    `
  }
]
</script>

<InteractiveQuiz :questions="quizData" quiz-id="normalization" />

## 💬 Open questions

### Q6: Debugging scenario

You train a 16-layer Transformer and the loss becomes NaN around step ~100. Based on what you learned, list possible causes and fixes.

<details>
<summary>Suggested answer</summary>

**Possible causes**:
1. Missing normalization → vanishing/exploding gradients
2. Using Post-LN → unstable in deep stacks
3. Too high learning rate
4. Poor initialization
5. FP16 numerical instability

**Fixes (top options)**:

1. **Switch to Pre-LN + RMSNorm**
   ```python
   class TransformerBlock(nn.Module):
       def forward(self, x):
           x = x + self.attn(self.norm1(x))  # Pre-Norm
           x = x + self.ffn(self.norm2(x))   # Pre-Norm
           return x
   ```

2. **Lower learning rate**
   - From 1e-3 down to 1e-4 or lower
   - Add warmup

3. **Check initialization**
   - Use Kaiming or Xavier initialization
   - Ensure activation std starts near 1.0

4. **Use gradient clipping**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

5. **Prefer BF16 over FP16**
   ```python
   with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
       output = model(input)
   ```

**Debug checklist**:
1. Log activation stats per layer (mean/std/max/min)
2. Monitor for NaNs in gradients
3. Compare with smaller configs

</details>

---

### Q7: Explain the intuition

Explain in your own words why normalization stabilizes training, using a simple analogy.

<details>
<summary>Suggested answer</summary>

**Example intuition**:

- **Without normalization**: each layer acts like a filter that can amplify or shrink signals, so the signal scale quickly drifts.
- **With normalization**: each layer first normalizes the input to a stable scale, so gradients flow reliably.

**Analogy**:
- A water system without pressure control: higher floors get almost no water.
- A system with pressure regulators: every floor receives stable pressure.

**Key takeaway**:
Normalization keeps signal scale stable, so gradients remain healthy and training converges.

</details>

---

## ✅ Completion checklist

- [ ] **Q1–Q5 correct**: core concepts confirmed
- [ ] **Q6: list 3+ causes and fixes**: practical debugging
- [ ] **Q7: clear analogy**: intuitive understanding

If you are not fully confident, review [teaching.md](./teaching.md) and rerun the experiments.

---

## 🎓 Next step

Continue to [02. Position Encoding](../../02-position-encoding).
