---
title: Attention Quiz | MiniMind LLM Training
description: Attention quiz to test your understanding of QKV, multi-head attention, scaled dot-product attention, and attention weights.
keywords: attention quiz, Attention test, QKV quiz, Transformer attention quiz, LLM training quiz
---

# Attention Quiz

> Answer the following questions to check your understanding.

---

## 🎮 Interactive Quiz (Recommended)

<script setup>
const quizData = [
  {
    question: 'Why divide by √d_k in Self-Attention?',
    type: 'single',
    options: [
      { label: 'A', text: 'To speed up computation' },
      { label: 'B', text: 'To reduce parameters' },
      { label: 'C', text: 'To prevent softmax saturation and stabilize gradients' },
      { label: 'D', text: 'To support longer sequences' }
    ],
    correct: [2],
    explanation: `
      <strong>Correct answer: C</strong><br><br>
      <ul>
        <li>Dot product variance grows with d_k</li>
        <li>Large values saturate softmax</li>
        <li>Saturated softmax → gradients near 0</li>
        <li>Dividing by √d_k keeps variance around 1</li>
      </ul>
      <strong>Example</strong> (d_k=64):<ul>
        <li>Unscaled: scores can reach ~64 (almost 0/1 softmax)</li>
        <li>Scaled: scores around ~8 (smoother distribution)</li>
      </ul>
    `
  },
  {
    question: 'What do Q, K, V represent?',
    type: 'single',
    options: [
      { label: 'A', text: 'Query=question, Key=key, Value=value' },
      { label: 'B', text: 'Query=problem, Key=keyword, Value=answer' },
      { label: 'C', text: 'Query=what I want, Key=what tags I have, Value=actual content' },
      { label: 'D', text: 'Query=input, Key=output, Value=intermediate' }
    ],
    correct: [2],
    explanation: `
      <strong>Correct answer: C</strong><br><br>
      <strong>Library analogy</strong>：<ul>
        <li><strong>Query</strong>: what book you want (search keywords)</li>
        <li><strong>Key</strong>: tags for each book</li>
        <li><strong>Value</strong>: the book content</li>
      </ul>
      <strong>Actual roles</strong>：<ul>
        <li>Q: what this token asks for</li>
        <li>K: how this token presents itself for matching</li>
        <li>V: what information this token provides</li>
      </ul>
      <strong>Why three projections?</strong><br>
      Let the model learn different views of the same token: Q/K for relevance, V for content
    `
  },
  {
    question: 'What is the main advantage of Multi-Head Attention?',
    type: 'single',
    options: [
      { label: 'A', text: 'Reduce computation' },
      { label: 'B', text: 'Reduce parameters' },
      { label: 'C', text: 'Learn multiple relationship patterns in parallel' },
      { label: 'D', text: 'Support longer sequences' }
    ],
    correct: [2],
    explanation: `
      <strong>Correct answer: C</strong><br><br>
      <strong>Single-head limitation</strong>：<ul>
        <li>Can only learn one "attention pattern"</li>
        <li>For example, only syntax but not semantics</li>
      </ul>
      <strong>Multi-head advantage</strong>：different heads learn different patterns<ul>
        <li>Head 1: syntax (subject-verb-object)</li>
        <li>Head 2: semantics (synonyms)</li>
        <li>Head 3: positional relations (nearby words)</li>
        <li>Head 4: coreference</li>
      </ul>
      Concatenation fuses multiple signals<br><br>
      <strong>Parameter note</strong>: total params are similar, but expressiveness is higher
    `
  },
  {
    question: 'What does GQA (Grouped Query Attention) do?',
    type: 'single',
    options: [
      { label: 'A', text: 'Improve accuracy' },
      { label: 'B', text: 'Reduce KV cache memory and speed up inference' },
      { label: 'C', text: 'Increase model capacity' },
      { label: 'D', text: 'Support more heads' }
    ],
    correct: [1],
    explanation: `
      <strong>Correct answer: B</strong><br><br>
      <strong>MHA issue</strong>：<ul>
        <li>Each head has its own K, V</li>
        <li>KV cache size = n_heads × seq_len × head_dim × 2</li>
        <li>Large memory and slower inference</li>
      </ul>
      <strong>GQA optimization</strong>：<ul>
        <li>Multiple Q heads share one KV group</li>
        <li>Example: 8 Q heads, only 2 KV heads</li>
        <li>KV cache reduced by 4×</li>
      </ul>
      Performance: minimal accuracy loss, significant memory and speed gains
    `
  },
  {
    question: 'What is the role of a causal mask?',
    type: 'single',
    options: [
      { label: 'A', text: 'Speed up computation' },
      { label: 'B', text: 'Prevent the model from seeing future tokens' },
      { label: 'C', text: 'Reduce parameters' },
      { label: 'D', text: 'Improve accuracy' }
    ],
    correct: [1],
    explanation: `
      <strong>Correct answer: B</strong><br><br>
      <strong>Scenario</strong>: autoregressive models generate tokens one by one<br><br>
      <strong>Causal mask</strong>：<ul>
        <li>Position i can attend only to positions ≤ i</li>
        <li>Future tokens are masked</li>
        <li>Ensures train/inference consistency</li>
      </ul>
      <strong>Implementation</strong>: set future scores to −∞ so softmax → 0
    `
  },
  {
    question: 'What does repeat_kv do in GQA?',
    type: 'single',
    options: [
      { label: 'A', text: 'Copy K and V to match the number of Q heads' },
      { label: 'B', text: 'Increase KV cache size' },
      { label: 'C', text: 'Improve numerical precision' },
      { label: 'D', text: 'Reduce memory usage' }
    ],
    correct: [0],
    explanation: `
      <strong>Correct answer: A</strong><br><br>
      <strong>GQA structure</strong>：<ul>
        <li>8 Q heads</li>
        <li>2 KV heads (each shared by 4 Q heads)</li>
      </ul>
      <strong>repeat_kv role</strong>：<ul>
        <li>Expand KV from (2, seq_len, head_dim) to (8, seq_len, head_dim)</li>
        <li>Each KV head is repeated 4 times</li>
        <li>So Q and KV shapes align for dot products</li>
      </ul>
      <strong>Key detail</strong>: repetition is a view expansion, not a full memory copy
    `
  },
  {
    question: 'What is the advantage of Flash Attention over the standard implementation?',
    type: 'single',
    options: [
      { label: 'A', text: 'Improves accuracy' },
      { label: 'B', text: 'Reduces parameters' },
      { label: 'C', text: 'Reduces GPU memory traffic and speeds up computation' },
      { label: 'D', text: 'Supports more attention heads' }
    ],
    correct: [2],
    explanation: `
      <strong>Correct answer: C</strong><br><br>
      <strong>Standard issue</strong>：<ul>
        <li>Materializes the full attention matrix (seq_len × seq_len)</li>
        <li>Frequent GPU HBM reads/writes (slow)</li>
      </ul>
      <strong>Flash Attention optimization</strong>：<ul>
        <li>Block computation in SRAM (fast)</li>
        <li>No full attention matrix materialization</li>
        <li>Less IO, 2–4× faster</li>
        <li>Longer sequences (memory O(N²) → O(N))</li>
      </ul>
      Mathematically equivalent, only implementation is optimized
    `
  }
]
</script>

<InteractiveQuiz :questions="quizData" quiz-id="attention" />

---

## 🎯 Comprehensive Questions

### Q8: Practical scenario

If you find that attention weights are almost uniform (each position is ~1/seq_len), what might be wrong and how can you fix it?

<details>
<summary>Show reference answer</summary>

**Possible causes**:

1. **Q/K initialization issues**:
   - projection matrices too small
   - Q·K scores near 0
   - softmax(0, 0, ..., 0) ≈ uniform

2. **Scaling factor issues**:
   - divided by too large a value
   - or forgot the square root (divide by d_k instead of √d_k)

3. **Head dim misconfigured**:
   - overly large head_dim makes variance huge
   - but this tends to cause extreme, not uniform, distributions

4. **Model didn’t learn meaningful patterns**:
   - data issues
   - insufficient capacity

**Diagnostics**:

```python
# check Q·K scores before softmax
scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(head_dim)
print(f"scores mean: {scores.mean()}, std: {scores.std()}")

# normal: mean ≈ 0, std ≈ 1
# problematic: std too small (near 0)
```

**Fixes**:

1. **Check initialization**:
   ```python
   # use Xavier or Kaiming initialization
   nn.init.xavier_uniform_(self.wq.weight)
   ```

2. **Verify scaling factor**:
   ```python
   # ensure sqrt(head_dim), not head_dim
   scale = math.sqrt(self.head_dim)
   ```

3. **Visualize attention**:
   ```python
   plt.imshow(attn_weights[0, 0].detach().numpy())
   plt.title("Attention weights")
   plt.colorbar()
   ```

</details>

---

### Q9: Conceptual understanding

Why do Q, K, V all come from the same input x, but still require three different projection matrices? Can we just use x directly?

<details>
<summary>Show reference answer</summary>

**Problem with using x directly**:

If Q = K = V = x, then:
```python
scores = x @ x.T  # self dot-products
```

This is essentially cosine similarity in the embedding space.

**Issues**:
1. **Symmetry**: attention from token_i to token_j equals attention from token_j to token_i
   - but language relations are often asymmetric
   - e.g., “cat eats fish”: cat should attend to fish, fish doesn’t need to attend to cat

2. **Limited expressiveness**:
   - only captures similarity
   - cannot represent relations like subject/object or modifier

**Why three projections matter**:

```python
Q = x @ W_Q  # "as a query, what should I focus on?"
K = x @ W_K  # "as a key, what should I reveal?"
V = x @ W_V  # "what content should I pass?"
```

**Advantages**:
1. **Asymmetry**: Q and K differ, enabling directional relations
2. **Role separation**: query view, key view, content view
3. **Expressiveness**: can learn arbitrary relation patterns

**Analogy**:
- Library search:
  - your question (Q): written in natural language
  - index labels (K): keywords/tags
  - content (V): the actual text
- different “languages,” matched to retrieve the right content

</details>

---

### Q10: Code understanding

Explain why `contiguous()` is necessary here:

```python
output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
```

<details>
<summary>Show reference answer</summary>

**Background**:

PyTorch tensors have two concepts:
1. **Storage**: physical memory layout
2. **View**: how you interpret that memory

**What transpose does**:

```python
# suppose output is [batch, n_heads, seq, head_dim]
# memory is laid out in that order

output = output.transpose(1, 2)
# now shape is [batch, seq, n_heads, head_dim]
# but memory layout is unchanged
```

**Problem**:

`view()` requires contiguous memory, but `transpose` makes it non-contiguous:

```python
# original memory order (simplified):
# [head0_pos0, head0_pos1, head1_pos0, head1_pos1, ...]

# logical order after transpose:
# [head0_pos0, head1_pos0, head0_pos1, head1_pos1, ...]

# non-contiguous → view fails
```

**What contiguous() does**:

```python
output = output.transpose(1, 2).contiguous()
# 1. reallocate memory
# 2. reorder data to match new view
# 3. now view is safe
```

**Performance note**:

- `contiguous()` copies memory (costly)
- but it’s required here
- alternative: `reshape()` handles this implicitly but is less explicit

**Best practice**:

```python
# when you know you need contiguous memory
output = output.transpose(1, 2).contiguous().view(...)

# or use reshape
output = output.transpose(1, 2).reshape(...)
```

</details>

---

## ✅ Completion check

After finishing all questions, check whether you can:

- [ ] **Get Q1–Q7 all correct**: solid basics
- [ ] **Provide 2+ diagnostics in Q8**: debugging ability
- [ ] **Explain projections in Q9**: design principles understood
- [ ] **Explain why contiguous() is needed in Q10**: PyTorch memory model

If anything is unclear, return to [teaching.md](./teaching.md) or rerun the experiments.

---

## 🎓 Advanced challenge

Want to go deeper? Try:

1. **Modify experiment code**:
   - implement Attention without scaling and observe softmax
   - implement MQA (Multi-Query Attention) and compare with GQA
   - visualize attention patterns from different heads

2. **Read papers**:
   - [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - original Transformer paper
   - [GQA Paper](https://arxiv.org/abs/2305.13245) - Grouped Query Attention
   - [Flash Attention](https://arxiv.org/abs/2205.14135) - efficient attention implementation

3. **Implement variants**:
   - implement Cross-Attention (Q and KV from different inputs)
   - implement Sliding Window Attention
   - implement Sparse Attention

---

**Next**: go to [04. FeedForward](../04-feedforward) to learn feed-forward networks.
