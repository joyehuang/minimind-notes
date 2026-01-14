---
title: Position Encoding Quiz | MiniMind LLM Training
description: Position Encoding quiz to test your understanding of RoPE, absolute vs relative positional encoding, and length extrapolation.
keywords: positional encoding quiz, RoPE quiz, Transformer positional encoding test, LLM training quiz
---

# Position Encoding Quiz

> Answer the following questions to check your understanding.

---

## 🎮 Interactive Quiz (Recommended)

<script setup>
const quizData = [
  {
    question: 'Why does Attention need positional encoding?',
    type: 'single',
    options: [
      { label: 'A', text: 'To speed up computation' },
      { label: 'B', text: 'To reduce parameters' },
      { label: 'C', text: 'Because Attention is permutation-invariant and cannot distinguish order' },
      { label: 'D', text: 'To support longer sequences' }
    ],
    correct: [2],
    explanation: `
      <strong>Correct answer: C</strong><br><br>
      <ul>
        <li>Self-Attention only cares about "who relates to whom"</li>
        <li>It does not encode "who comes before whom"</li>
        <li>For example, "I like you" and "You like I" can yield the same attention pattern</li>
        <li>Positional encoding allows the model to distinguish order</li>
      </ul>
    `
  },
  {
    question: 'What is the core idea of RoPE?',
    type: 'single',
    options: [
      { label: 'A', text: 'Assign a learnable vector to each position' },
      { label: 'B', text: 'Encode position by rotating vectors' },
      { label: 'C', text: 'Compute relative distance between tokens' },
      { label: 'D', text: 'Use sine functions to generate positional encodings' }
    ],
    correct: [1],
    explanation: `
      <strong>Correct answer: B</strong><br><br>
      <ul>
        <li>RoPE = Rotary Position Embedding</li>
        <li>Core idea: position m → rotate by mθ</li>
        <li>Relative position emerges from rotation angle differences</li>
        <li>Different from absolute embeddings (A) and the original sinusoidal encodings (D)</li>
      </ul>
    `
  },
  {
    question: 'Why does RoPE use multiple frequencies?',
    type: 'single',
    options: [
      { label: 'A', text: 'To speed up computation' },
      { label: 'B', text: 'To reduce memory usage' },
      { label: 'C', text: 'High frequency encodes local position, low frequency encodes global position' },
      { label: 'D', text: 'To be compatible with different head_dim values' }
    ],
    correct: [2],
    explanation: `
      <strong>Correct answer: C</strong><br><br>
      <strong>Clock analogy</strong>：<ul>
        <li>Second hand (high freq): fine-grained, local position</li>
        <li>Hour hand (low freq): coarse, global position</li>
        <li>Combined, they uniquely identify time</li>
      </ul>
      <strong>Single-frequency issues</strong>：<ul>
        <li>Only high freq: long sequences wrap too much → ambiguous positions</li>
        <li>Only low freq: short sequences rotate too slowly → poor discrimination</li>
      </ul>
      <strong>Multi-frequency combination</strong>：can uniquely encode million-level positions
    `
  },
  {
    question: 'Where is RoPE applied in Attention?',
    type: 'single',
    options: [
      { label: 'A', text: 'Query only' },
      { label: 'B', text: 'Key only' },
      { label: 'C', text: 'Query and Key' },
      { label: 'D', text: 'Query, Key, and Value' }
    ],
    correct: [2],
    explanation: `
      <strong>Correct answer: C</strong><br><br>
      <code>xq, xk = apply_rotary_emb(xq, xk, freqs_cis) # V not needed</code><br><br>
      <strong>Why V is not needed</strong><ul>
        <li>Q and K compute attention scores (need position info)</li>
        <li>V is the content being retrieved (position info already in Q·K)</li>
        <li>Applying RoPE to V is redundant</li>
      </ul>
    `
  },
  {
    question: 'What is the main advantage of RoPE over absolute positional embeddings?',
    type: 'single',
    options: [
      { label: 'A', text: 'Faster computation' },
      { label: 'B', text: 'Fewer parameters' },
      { label: 'C', text: 'Supports length extrapolation (train short, infer long)' },
      { label: 'D', text: 'Simpler implementation' }
    ],
    correct: [2],
    explanation: `
      <strong>Correct answer: C</strong><br><br>
      <strong>Absolute positional embedding</strong>：<ul>
        <li>One vector per position: <code>pos_embed[0], pos_embed[1], ..., pos_embed[511]</code></li>
        <li>Train up to 512 → inference at 600 has no embedding</li>
        <li>❌ Cannot extrapolate</li>
      </ul>
      <strong>RoPE</strong>：<ul>
        <li>Only rotation angles: position 600 → rotate 600θ</li>
        <li>Mathematically defined for any position</li>
        <li>✅ Supports extrapolation (even better with YaRN)</li>
      </ul>
    `
  }
]
</script>

<InteractiveQuiz :questions="quizData" quiz-id="position-encoding" />

---

## 🎯 Comprehensive Questions

### Q6: Practical scenario

Assume you trained a RoPE model with max_seq_len=512. Now you need to process sequences of length 2048. What issues might occur, and how can you fix them?

<details>
<summary>Show reference answer</summary>

**Possible issues**:

1. **Performance drop**:
   - Extrapolation is possible, but the model never saw such long sequences
   - Long-range attention patterns may be inaccurate

2. **Over-rotation**:
   - Position 2000 rotates 4× more than training length
   - High-frequency dimensions may wrap too many times and lose information

**Solutions**:

1. **Use YaRN** (recommended):
   ```python
   # enable in MiniMind
   config.inference_rope_scaling = True
   ```
   - rescale rotation frequencies
   - smooths long-sequence rotations

2. **Position Interpolation**:
   ```python
   # scale positions into training range
   pos_ids = pos_ids * (512 / 2048)
   ```
   - simple but effective
   - compresses the position range

3. **Continue training**:
   - fine-tune with long-sequence data
   - adapt the model to long-range attention

**Best practices**:
- use a larger rope_base in training (e.g., 1e6)
- enable YaRN in inference
- validate with task-specific tests

</details>

---

### Q7: Conceptual understanding

In your own words: why does the RoPE dot product depend only on relative position?

<details>
<summary>Show reference answer</summary>

**Math intuition**:

Assume:
- Query at position m: $q_m = R(m\theta) \cdot q$ (rotate by mθ)
- Key at position n: $k_n = R(n\theta) \cdot k$ (rotate by nθ)

Dot product:
$$q_m \cdot k_n = [R(m\theta) \cdot q] \cdot [R(n\theta) \cdot k]$$

**Key property**: the transpose of a rotation matrix is its inverse

$$= q \cdot R(-m\theta) \cdot R(n\theta) \cdot k$$
$$= q \cdot R((n-m)\theta) \cdot k$$

**Conclusion**:
- the dot product contains $(n-m)\theta$ only
- no absolute positions $m$ or $n$

**Intuitive analogy**:
- two people facing each other
- no matter where they stand in the room
- their relative angle stays the same

That’s why:
- positions (5, 8) and (100, 103) have the same attention score
- the model learns “distance = 3” patterns

</details>

---

## ✅ Completion check

After finishing all questions, check whether you can:

- [ ] **Get Q1–Q5 all correct**: solid basics
- [ ] **Provide 2+ solutions in Q6**: practical ability
- [ ] **Explain Q7 clearly**: deep conceptual understanding

If anything is unclear, return to [teaching.md](./teaching.md) or rerun the experiments.

---

## 🎓 Advanced challenge

Want to go deeper? Try:

1. **Modify experiment code**:
   - implement a simple absolute positional encoding
   - compare length extrapolation vs RoPE
   - test different rope_base values

2. **Read papers**:
   - [RoFormer original paper](https://arxiv.org/abs/2104.09864)
   - [YaRN length extrapolation](https://arxiv.org/abs/2309.00071)

3. **Implement variants**:
   - implement ALiBi (another positional encoding)
   - compare RoPE vs ALiBi

---

**Next**: go to [03. Attention](../../03-attention) to learn attention mechanisms.
