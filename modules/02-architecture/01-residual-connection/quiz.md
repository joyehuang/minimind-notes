---
title: Residual Connection 自测题 | minimind从零理解llm训练
description: 残差连接模块自测题，检验你对梯度消失、恒等映射、Pre-Norm 和双残差结构的理解程度。
keywords: 残差连接自测题, 梯度消失测试, 残差网络测试, Transformer残差测试
---

# Residual Connection 自测题

> 完成以下题目检验你的理解程度

---

## 🎮 交互式自测（推荐）

<script setup>
const quizData = [
  {
    question: '残差连接的核心公式是什么？',
    type: 'single',
    options: [
      { label: 'A', text: 'y = F(x) - x' },
      { label: 'B', text: 'y = F(x) * x' },
      { label: 'C', text: 'y = F(x) + x' },
      { label: 'D', text: 'y = F(x) / x' }
    ],
    correct: [2],
    explanation: `
      <strong>正确答案：C</strong><br><br>
      <strong>残差连接</strong>：<code>y = F(x) + x</code><br><br>
      <strong>含义</strong>：<ul>
        <li>F(x) 是子网络学到的残差（修正量）</li>
        <li>+x 是恒等映射（identity）路径</li>
        <li>网络不是从零学输出，而是学"需要调整多少"</li>
      </ul>
      <strong>类比</strong>：不是画一整幅画，而是在已有草稿上修改
    `
  },
  {
    question: '残差连接解决梯度消失的核心机制是什么？',
    type: 'single',
    options: [
      { label: 'A', text: '通过增加网络宽度' },
      { label: 'B', text: '通过导数公式中的 +1 项' },
      { label: 'C', text: '通过使用更大的学习率' },
      { label: 'D', text: '通过减少网络深度' }
    ],
    correct: [1],
    explanation: `
      <strong>正确答案：B</strong><br><br>
      <strong>梯度公式</strong>：<br>
      <code>∂L/∂x = ∂L/∂y · (∂F/∂x + 1)</code><br><br>
      <strong>关键</strong>：<ul>
        <li>括号里永远有 <strong>+1</strong> 这个常数</li>
        <li>即使子网络 ∂F/∂x ≈ 0，梯度仍然 ≈ ∂L/∂y</li>
        <li>不会像无残差那样连乘衰减！</li>
      </ul>
      <strong>无残差</strong>：<code>∂L/∂x = ∂L/∂y · ∂F₁/∂x · ∂F₂/∂x · (...) · ∂Fₙ/∂x</code><br>
      连乘 N 次，每一项 &lt; 1 时梯度 → 0
    `
  },
  {
    question: '"残差"（residual）这个词的含义是什么？',
    type: 'single',
    options: [
      { label: 'A', text: '网络输出的误差' },
      { label: 'B', text: '网络需要学习的变化量（y - x）' },
      { label: 'C', text: '网络深度的差值' },
      { label: 'D', text: '权重矩阵的差值' }
    ],
    correct: [1],
    explanation: `
      <strong>正确答案：B</strong><br><br>
      <strong>残差的含义</strong>：<ul>
        <li>如果<strong>没有</strong>子网络，输出就是 x（恒等映射）</li>
        <li>子网络 F(x) 学习的是 <strong>y - x</strong>，即"需要调整多少"</li>
        <li>这比直接从零学 y 容易得多</li>
      </ul>
      <strong>类比</strong>：<ul>
        <li>直接画肖像 → 很难（学完整的 y）</li>
        <li>在照片上修改瑕疵 → 容易（只学差值 F(x)）</li>
      </ul>
      <strong>极端情况</strong>：如果最优解就是什么都不做：<ul>
        <li>无残差：需要让 F(x) = x（精确拟合，很难）</li>
        <li>有残差：只需要让 F(x) = 0（权重全 0，很容易）</li>
      </ul>
    `
  },
  {
    question: 'Pre-Norm 和 Post-Norm 的主要区别是什么？',
    type: 'single',
    options: [
      { label: 'A', text: '归一化的数学公式不同' },
      { label: 'B', text: 'Pre-Norm 先归一化再做子层计算，Post-Norm 先计算再归一化' },
      { label: 'C', text: 'Pre-Norm 用 RMSNorm，Post-Norm 用 LayerNorm' },
      { label: 'D', text: 'Pre-Norm 更快但效果更差' }
    ],
    correct: [1],
    explanation: `
      <strong>正确答案：B</strong><br><br>
      <strong>Pre-Norm</strong>：<code>x = x + F(Norm(x))</code><ul>
        <li>先归一化输入，再做子层计算</li>
        <li>残差路径（+x）不过 Norm → 梯度流畅通</li>
        <li>训练更稳定，不需要 warmup</li>
        <li>Llama、GPT-3、MiniMind 的选择</li>
      </ul>
      <strong>Post-Norm</strong>：<code>x = Norm(x + F(x))</code><ul>
        <li>先计算子层，再归一化输出</li>
        <li>残差路径也过 Norm → 梯度可能被压缩</li>
        <li>需要学习率 warmup</li>
        <li>原始 Transformer 的选择</li>
      </ul>
    `
  },
  {
    question: 'Transformer Block 中有几个残差连接？',
    type: 'single',
    options: [
      { label: 'A', text: '1 个（只在 Attention 后）' },
      { label: 'B', text: '2 个（Attention 后和 FFN 后）' },
      { label: 'C', text: '3 个' },
      { label: 'D', text: '没有残差连接' }
    ],
    correct: [1],
    explanation: `
      <strong>正确答案：B</strong><br><br>
      <strong>双残差结构</strong>：<br>
      <code>x → Norm → Attention → [+x] → Norm → FFN → [+x] → 输出</code><br><br>
      <strong>两个残差各有分工</strong>：<ul>
        <li>残差 1（Attention 后）：让 Attention 的梯度直达输入</li>
        <li>残差 2（FFN 后）：让 FFN 的梯度也能跳过 Attention</li>
      </ul>
      <strong>为什么需要两个？</strong><ul>
        <li>两个子层（Attention 和 FFN）独立优化</li>
        <li>每个子层有各自的高速公路</li>
        <li>即使一个子层学坏了，另一个的梯度不受影响</li>
      </ul>
    `
  },
  {
    question: '什么是"退化问题"（degradation problem）？',
    type: 'single',
    options: [
      { label: 'A', text: '网络越深，参数量越大' },
      { label: 'B', text: '网络越深反而效果越差，比浅层网络还差' },
      { label: 'C', text: '网络越深，推理速度越慢' },
      { label: 'D', text: '网络越深，GPU 显存不够' }
    ],
    correct: [1],
    explanation: `
      <strong>正确答案：B</strong><br><br>
      <strong>退化问题</strong>（ResNet 论文提出）：<ul>
        <li>理论上：深层网络 <strong>至少</strong>不该比浅层差</li>
        <li>多出的层可以学成恒等映射（什么都不做）→ 效果应该持平</li>
        <li>实际上：深层网络反而 <strong>更差</strong>！</li>
      </ul>
      <strong>原因</strong>：梯度消失导致多出的层学不好，反而引入噪声<br><br>
      <strong>残差连接的解决</strong>：<ul>
        <li>让网络 <strong>真的</strong> 能学到恒等映射</li>
        <li>因为 F(x) = 0 是很自然的状态（权重初始化接近 0）</li>
        <li>深度增加至少不会变差</li>
      </ul>
    `
  },
  {
    question: '残差连接的计算开销有多大？',
    type: 'single',
    options: [
      { label: 'A', text: '很大，几乎翻倍' },
      { label: 'B', text: '几乎为零，只是逐元素加法' },
      { label: 'C', text: '中等，约增加 30%' },
      { label: 'D', text: '取决于网络深度' }
    ],
    correct: [1],
    explanation: `
      <strong>正确答案：B</strong><br><br>
      <strong>残差连接的计算</strong>：<ul>
        <li>只是逐元素加法：<code>y = a + b</code></li>
        <li>复杂度 O(batch × seq × dim)</li>
        <li>相比 Attention 的 O(seq² × dim) 和 FFN 的 O(dim²)，几乎可忽略</li>
      </ul>
      <strong>内存</strong>：需要保存原始输入用于反向传播，但用 checkpointing 可节省<br><br>
      <strong>总结</strong>：性价比极高——几乎零开销换来训练稳定性和深度能力
    `
  }
]
</script>

<InteractiveQuiz :questions="quizData" quiz-id="residual-connection" />

---

## 🎯 综合问答题

### Q8: 梯度推导

写出有残差和无残差两种情况下，对输入 x 的梯度公式，并解释关键区别。

<details>
<summary>点击查看参考答案</summary>

**无残差连接** （y = F(x)）：

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial F}{\partial x}$$

如果是 N 层串联：$F = F_N \circ F_{N-1} \circ ... \circ F_1$

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial F_N}{\partial x_{N-1}} \cdot ... \cdot \frac{\partial F_1}{\partial x}$$

每一层连乘 → 梯度指数衰减。

**有残差连接** （y = F(x) + x）：

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot (\frac{\partial F}{\partial x} + \mathbf{1})$$

多层时展开为：

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot (\frac{\partial F_N}{\partial x} + \mathbf{1})(\frac{\partial F_{N-1}}{\partial x} + \mathbf{1})...$$

展开后：连乘项 + 多个包含 +1 的路径 → 即使连乘项衰减，+1 路提供的梯度仍在。

**核心区别**：+1 保证了至少有一条路径的梯度恒为 1，不会衰减到 0。

</details>

---

### Q9: 实战问题

如果你发现自己实现的 Transformer Block 训练时深层梯度仍然消失，可能是什么问题？如何调试？

<details>
<summary>点击查看参考答案</summary>

**可能的原因**：

1. **残差连接没正确实现**：
   - 忘记保存 residual 变量
   - 子层输出维度不匹配
   - Pre-Norm 位置写反了

2. **初始化问题**：
   - 权重初始化太大导致激活爆炸
   - 后续被截断时梯度变 0

3. **子层内部问题**：
   - Attention 里的 softmax 可能导致极端分布
   - FFN 的激活函数负半轴截断过多

**调试方法**：

```python
# 在 MiniMindBlock.forward 中添加监控
residual = hidden_states

# 检查 Attention 输出
attn_out, _ = self.self_attn(self.input_layernorm(hidden_states), ...)
print(f"Attention output norm: {attn_out.norm():.4f}")

hidden_states = attn_out + residual
print(f"After residual1 norm: {hidden_states.norm():.4f}")

# 检查梯度
hidden_states.retain_grad()
residual.retain_grad()

# 反向传播后
print(f"residual grad: {residual.grad.norm():.4f}")
print(f"hidden_states grad: {hidden_states.grad.norm():.4f}")
```

**常见修复**：

1. 确保残差路径不经过 Norm（Pre-Norm）
2. 检查子层的输出维度 = 输入维度
3. 使用梯度裁剪 `torch.nn.utils.clip_grad_norm_`

</details>

---

### Q10: 概念理解

为什么 Transformer 中每个 Block 需要**两个**残差连接，而不是像 ResNet 那样一整个 Block 一个残差？

<details>
<summary>点击查看参考答案</summary>

**设计考虑**：

1. **粒度不同**：
   - ResNet：一个 Block 就是一个残差的单位
   - Transformer：一个 Block 包含两个**不同性质**的子层（Attention + FFN）

2. **功能差异**：
   - Attention：词与词交互，全局信息交换
   - FFN：逐位置独立处理，特征变换
   - 两个子层做的事完全不同，需要独立的高速公路

3. **容错性**：
   - 如果整个 Block 共用一个残差，Attention 学坏了会影响 FFN 的输入
   - 各用一个残差，互相独立，一个学坏了不影响另一个

4. **梯度流**：
   - Attention 的梯度可以通过残差直达输入，不经过 FFN
   - FFN 的梯度也可以通过残差直达 Attention 之前
   - 两个独立高速路比一个共享高速路更灵活

**代码体现**：

```python
# 两个独立的残差
x = x + attention(norm1(x))   # 残差 1，只跳 Attention
x = x + ffn(norm2(x))         # 残差 2，只跳 FFN

# 如果合并为一个（不推荐）：
x = x + ffn(attention(norm(x)))  # 两个子层共用一个残差
```

</details>

---

## ✅ 完成检查

完成所有题目后，检查你是否达到：

- [ ] **Q1-Q7 全对**：基础知识扎实
- [ ] **Q8 能正确写出梯度公式**：数学理解到位
- [ ] **Q9 能提出 2+ 调试方法**：具备调试能力
- [ ] **Q10 能解释双残差设计**：理解架构设计

如果还有不清楚的地方，回到 [teaching.md](./teaching.md) 复习，或重新运行实验代码。

---

## 🎓 进阶挑战

想要更深入理解？尝试：

1. **修改实验代码**：
   - 在 exp2 中添加 Post-Norm 的对比
   - 测量不同深度下两种 Norm 的梯度差异

2. **阅读论文**：
   - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
   - [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)

3. **代码挑战**：
   - 从零实现一个完整的 Pre-Norm Transformer Block
   - 堆叠 8 个 Block 并用实验 1 的数据验证能训练

---

**下一步**：前往 [02. Transformer Block](../02-transformer-block/) 学习完整的 Block 组装！
