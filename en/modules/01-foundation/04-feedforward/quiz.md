---
title: FeedForward Quiz | MiniMind LLM Training
description: FeedForward quiz to test your understanding of expand-compress, SwiGLU activation, and FFN knowledge storage.
keywords: feedforward quiz, FFN quiz, Transformer feedforward test, LLM training quiz
---

# FeedForward Quiz

> Answer the following questions to check your understanding.

---

## 🎮 Interactive Quiz (Recommended)

<script setup>
const quizData = [
  {
    question: 'Why does FeedForward use "expand-compress"?',
    type: 'single',
    options: [
      { label: 'A', text: 'To reduce computation' },
      { label: 'B', text: 'To reduce parameters' },
      { label: 'C', text: 'To increase expressiveness in a higher-dimensional space' },
      { label: 'D', text: 'To speed up inference' }
    ],
    correct: [2],
    explanation: `
      <strong>Correct answer: C</strong><br><br>
      <strong>Expand-compress architecture</strong>：<ul>
        <li>Input: hidden_size (e.g., 512)</li>
        <li>Expand: intermediate_size (e.g., 2048, 4×)</li>
        <li>Compress: back to hidden_size (512)</li>
      </ul>
      <strong>Why expand?</strong><ul>
        <li>Higher-dimensional space gives more expressive capacity</li>
        <li>Allows more complex non-linear transformations</li>
        <li>Improves feature extraction</li>
      </ul>
      <strong>Analogy</strong>: like painting on a larger canvas
    `
  },
  {
    question: 'How many projection matrices does SwiGLU use?',
    type: 'single',
    options: [
      { label: 'A', text: '1' },
      { label: 'B', text: '2' },
      { label: 'C', text: '3' },
      { label: 'D', text: '4' }
    ],
    correct: [2],
    explanation: `
      <strong>Correct answer: C</strong><br><br>
      <strong>SwiGLU structure</strong>：<ul>
        <li><code>w1</code>: hidden_size → intermediate_size (gate path)</li>
        <li><code>w2</code>: intermediate_size → hidden_size (output projection)</li>
        <li><code>w3</code>: hidden_size → intermediate_size (value path)</li>
      </ul>
      <strong>Computation</strong>：<br>
      <code>output = w2(SiLU(w1(x)) * w3(x))</code><br><br>
      Compared to standard FFN, w3 adds gating
    `
  },
  {
    question: 'What is the formula for SiLU?',
    type: 'single',
    options: [
      { label: 'A', text: 'max(0, x)' },
      { label: 'B', text: 'x * sigmoid(x)' },
      { label: 'C', text: 'tanh(x)' },
      { label: 'D', text: '1 / (1 + e^(-x))' }
    ],
    correct: [1],
    explanation: `
      <strong>Correct answer: B</strong><br><br>
      <strong>SiLU (Swish) formula</strong>：<br>
      <code>SiLU(x) = x · σ(x) = x · (1 / (1 + e^(-x)))</code><br><br>
      <strong>Properties</strong>：<ul>
        <li>Smooth non-linear function</li>
        <li>Negative values are not completely zeroed (unlike ReLU)</li>
        <li>Lower bound around -0.28, no upper bound</li>
        <li>Smoother gradients, more stable training</li>
      </ul>
      <strong>Compare</strong>: ReLU = max(0, x); SiLU is softer
    `
  },
  {
    question: 'What does the gating mechanism do in SwiGLU?',
    type: 'single',
    options: [
      { label: 'A', text: 'Speed up computation' },
      { label: 'B', text: 'Reduce parameters' },
      { label: 'C', text: 'Selectively control information flow' },
      { label: 'D', text: 'Prevent vanishing gradients' }
    ],
    correct: [2],
    explanation: `
      <strong>Correct answer: C</strong><br><br>
      <strong>Gating mechanism</strong>：<br>
      <code>gate = SiLU(w1(x))</code><br>
      <code>value = w3(x)</code><br>
      <code>output = gate * value</code><br><br>
      <strong>Role</strong>：<ul>
        <li>gate decides "how much to pass"</li>
        <li>value decides "what content to pass"</li>
        <li>multiplication enables selective transfer</li>
        <li>similar idea to LSTM gates</li>
      </ul>
      The model learns to control information flow per position
    `
  },
  {
    question: 'What is the main difference between FeedForward and Attention?',
    type: 'single',
    options: [
      { label: 'A', text: 'FFN has more parameters' },
      { label: 'B', text: 'FFN processes each position independently; Attention mixes positions' },
      { label: 'C', text: 'FFN is faster' },
      { label: 'D', text: 'FFN performs better' }
    ],
    correct: [1],
    explanation: `
      <strong>Correct answer: B</strong><br><br>
      <strong>Attention</strong>：<ul>
        <li>Mixes information across positions</li>
        <li>Output at position i depends on all positions</li>
        <li>Responsible for "information exchange"</li>
      </ul>
      <strong>FeedForward</strong>：<ul>
        <li>Processes each position independently</li>
        <li>Output at position i depends only on position i</li>
        <li>Responsible for "feature transformation"</li>
      </ul>
      <strong>Analogy</strong>: Attention is discussion; FFN is individual thinking
    `
  },
  {
    question: 'Why is SwiGLU often better than ReLU?',
    type: 'single',
    options: [
      { label: 'A', text: 'Faster computation' },
      { label: 'B', text: 'Fewer parameters' },
      { label: 'C', text: 'Gating adds richer non-linearity and smoother gradients' },
      { label: 'D', text: 'Simpler implementation' }
    ],
    correct: [2],
    explanation: `
      <strong>Correct answer: C</strong><br><br>
      <strong>ReLU limitations</strong>：<ul>
        <li>Hard cutoff: negatives become 0</li>
        <li>Non-smooth gradient</li>
        <li>Dead ReLU problem (neurons can die)</li>
      </ul>
      <strong>SwiGLU advantages</strong>：<ul>
        <li>Smooth activation (SiLU)</li>
        <li>Gating increases expressiveness</li>
        <li>More stable gradients</li>
        <li>Empirically better for LLMs</li>
      </ul>
      Cost: slightly more compute (3 projections vs 2)
    `
  },
  {
    question: 'In MiniMind, intermediate_size is typically how many times hidden_size?',
    type: 'single',
    options: [
      { label: 'A', text: '2×' },
      { label: 'B', text: '4×' },
      { label: 'C', text: '8×' },
      { label: 'D', text: '16×' }
    ],
    correct: [1],
    explanation: `
      <strong>Correct answer: B</strong><br><br>
      <strong>Standard setup</strong>：<ul>
        <li>hidden_size = 512 → intermediate_size = 2048 (4×)</li>
        <li>hidden_size = 768 → intermediate_size = 3072 (4×)</li>
      </ul>
      <strong>Why 4×?</strong><ul>
        <li>Balances expressiveness and compute</li>
        <li>Mainstream models like Llama/GPT use 4×</li>
        <li>Empirically a good trade-off</li>
      </ul>
      <strong>Variants</strong>：<ul>
        <li>MoE models may use larger multiples</li>
        <li>Small models may use 2–3× to save parameters</li>
      </ul>
    `
  }
]
</script>

<InteractiveQuiz :questions="quizData" quiz-id="feedforward" />

---

## 🎯 Comprehensive Questions

### Q8: Practical scenario

If FeedForward outputs are always close to 0, what might be wrong and how would you debug it?

<details>
<summary>Show reference answer</summary>

**Possible causes**:

1. **Initialization issues**:
   - weights initialized too small
   - outputs become tiny

2. **Gate signal issues**:
   - gate_proj outputs mostly negative
   - SiLU(negative) ≈ 0
   - gate * up ≈ 0

3. **Vanishing gradients**:
   - deep network, gradients don’t flow
   - weights stop updating

4. **Precision issues**:
   - low precision (e.g., FP16)
   - small values underflow to 0

**Debug steps**:

```python
# 1. inspect intermediates
gate = self.gate_proj(x)
print(f"gate mean: {gate.mean()}, std: {gate.std()}")

silu_gate = F.silu(gate)
print(f"silu_gate mean: {silu_gate.mean()}, std: {silu_gate.std()}")

up = self.up_proj(x)
print(f"up mean: {up.mean()}, std: {up.std()}")

hidden = silu_gate * up
print(f"hidden mean: {hidden.mean()}, std: {hidden.std()}")

# 2. check weight initialization
for name, param in self.named_parameters():
    print(f"{name}: mean={param.mean():.6f}, std={param.std():.6f}")

# 3. check gradients
output.sum().backward()
for name, param in self.named_parameters():
    if param.grad is not None:
        print(f"{name} grad: {param.grad.abs().mean():.6f}")
```

**Solutions**:

1. **Adjust initialization**:
   ```python
   nn.init.xavier_uniform_(self.gate_proj.weight)
   ```

2. **Check RMSNorm**:
   - ensure inputs are normalized
   - avoid extreme ranges

3. **Mixed precision**:
   - use FP32 for critical ops
   - BF16 for the rest

</details>

---

### Q9: Conceptual understanding

Why does FeedForward process each position independently instead of global interaction like Attention?

<details>
<summary>Show reference answer</summary>

**Design rationale**:

1. **Clear division of labor**:
   - Attention already mixes information
   - FeedForward focuses on "deep processing"
   - avoids redundant functionality

2. **Compute efficiency**:
   - global interaction: O(n²d)
   - independent processing: O(nd²)
   - when n > d, independent is cheaper

3. **Parameter efficiency**:
   - global interaction needs position-dependent parameters
   - independent processing can reuse parameters

4. **Theory**:
   - Transformer design:
     - Attention = global routing
     - FFN = local feature transformation
   - similar to spatial conv + 1x1 conv in CNNs

**Analogy**:
- meeting (Attention): exchange info
- thinking (FFN): digest and organize
- alternating works best

**Empirical evidence**:
- papers show this division works well
- all-FFN or all-Attention designs perform worse

</details>

---

### Q10: Code understanding

Explain each step in the following code:

```python
def forward(self, x):
    return self.down_proj(
        F.silu(self.gate_proj(x)) * self.up_proj(x)
    )
```

<details>
<summary>Show reference answer</summary>

**Step-by-step**:

```python
def forward(self, x):
    # x: [batch, seq, hidden_dim]
    # e.g., [32, 512, 512]

    # Step 1: gate signal
    gate = self.gate_proj(x)
    # gate: [batch, seq, intermediate_dim]
    # e.g., [32, 512, 2048]

    # Step 2: SiLU activation
    gate_activated = F.silu(gate)
    # SiLU(x) = x * sigmoid(x)
    # smooth activation, preserves some negative info
    # gate_activated: [32, 512, 2048]

    # Step 3: value signal
    up = self.up_proj(x)
    # up: [32, 512, 2048]

    # Step 4: gating
    hidden = gate_activated * up
    # elementwise multiply
    # gate_activated controls the flow of up
    # hidden: [32, 512, 2048]

    # Step 5: compress back
    output = self.down_proj(hidden)
    # output: [32, 512, 512]

    return output
```

**Key points**:

1. **Two parallel paths**: gate_proj and up_proj
2. **Gating**: SiLU(gate) controls up
3. **Dimension flow**: 512 → 2048 → 512
4. **No bias**: all Linear layers use bias=False

**Why this style?**
- concise: one-line formula
- efficient: frameworks optimize this pattern
- direct mapping to SwiGLU formula

</details>

---

## ✅ Completion check

After finishing all questions, check whether you can:

- [ ] **Get Q1–Q7 all correct**: solid basics
- [ ] **Provide 2+ debugging steps in Q8**: troubleshooting ability
- [ ] **Explain design rationale in Q9**: architecture understanding
- [ ] **Explain code step-by-step in Q10**: implementation clarity

If anything is unclear, return to [teaching.md](./teaching.md) or rerun the experiments.

---

## 🎓 Advanced challenge

Want to go deeper? Try:

1. **Modify experiment code**:
   - compare ReLU, GELU, SiLU activation distributions
   - visualize gating patterns
   - test different intermediate_size values

2. **Read papers**:
   - [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - GLU family comparison
   - [Searching for Activation Functions](https://arxiv.org/abs/1710.05941) - Swish activation

3. **Implement variants**:
   - implement GeGLU (GELU gating)
   - implement standard FFN (compare results)
   - implement MoE FeedForward

---

**Next**: go to [05. Residual Connection](../../02-architecture/05-residual-connection) to learn residual connections.
