# Normalization 自测题

> 完成以下题目检验你的理解程度

---

## 🎮 交互式自测（推荐）

<script setup>
const quizData = [
  {
    question: '梯度消失问题的根本原因是什么？',
    type: 'single',
    options: [
      { label: 'A', text: '学习率设置太小' },
      { label: 'B', text: '深层网络中激活值的分布随层数变化而失控' },
      { label: 'C', text: '优化器选择不当' },
      { label: 'D', text: '数据集太小' }
    ],
    correct: [1],
    explanation: `
      <strong>正确答案：B</strong><br><br>
      梯度消失的根本原因是深层网络中，激活值的标准差会随着层数增加而衰减。
      <ul>
        <li>当标准差变得非常小（接近 0）时，反向传播的梯度也会变小</li>
        <li>最终导致前面几层的权重几乎无法更新</li>
        <li>归一化通过控制每一层的激活分布来解决这个问题</li>
      </ul>
    `
  },
  {
    question: 'RMSNorm 和 LayerNorm 的主要区别是什么？',
    type: 'single',
    options: [
      { label: 'A', text: 'RMSNorm 有更多可学习参数' },
      { label: 'B', text: 'RMSNorm 不减均值，只除以 RMS' },
      { label: 'C', text: 'RMSNorm 只能用于 Transformer' },
      { label: 'D', text: 'RMSNorm 效果比 LayerNorm 差' }
    ],
    correct: [1],
    explanation: `
      <strong>正确答案：B</strong><br><br>
      <ul>
        <li>LayerNorm: <code>(x - μ) / σ</code>（减均值，除以标准差）</li>
        <li>RMSNorm: <code>x / RMS(x)</code>（只除以均方根，不减均值）</li>
        <li>RMSNorm 更简单、更快（减少一步计算）</li>
        <li>在 LLM 训练中效果相当，但 RMSNorm 速度提升 7-64%</li>
      </ul>
    `
  },
  {
    question: 'Pre-LN 相比 Post-LN 的优势是什么？',
    type: 'single',
    options: [
      { label: 'A', text: '计算速度更快' },
      { label: 'B', text: '使用更少参数' },
      { label: 'C', text: '残差路径更干净，梯度流更稳定' },
      { label: 'D', text: '不需要学习率调整' }
    ],
    correct: [2],
    explanation: `
      <strong>正确答案：C</strong><br><br>
      <strong>Post-LN</strong>（旧方案）：<code>x = LayerNorm(x + Attention(x))</code>
      <ul>
        <li>梯度需要经过 LayerNorm，可能被打断</li>
      </ul>
      <strong>Pre-LN</strong>（现代方案）：<code>x = x + Attention(Norm(x))</code>
      <ul>
        <li>残差路径上没有 Norm，梯度可以直接传播</li>
        <li>深层网络（>12 层）更稳定</li>
        <li>学习率容忍度更高</li>
      </ul>
    `
  },
  {
    question: '在 MiniMind 的一个 TransformerBlock 中，有几个 RMSNorm？',
    type: 'single',
    options: [
      { label: 'A', text: '1 个' },
      { label: 'B', text: '2 个' },
      { label: 'C', text: '4 个' },
      { label: 'D', text: '8 个' }
    ],
    correct: [1],
    explanation: `
      <strong>正确答案：B</strong><br><br>
      每个 TransformerBlock 有 <strong>2 个 RMSNorm</strong>：
      <ol>
        <li><strong>attention_norm</strong>：在 Attention 之前</li>
        <li><strong>ffn_norm</strong>：在 FeedForward 之前</li>
      </ol>
      数据流：<code>x → Norm #1 → Attention → + residual → Norm #2 → FFN → + residual → output</code><br><br>
      MiniMind-small 有 8 个 Block，所以总共 16 个 RMSNorm。
    `
  },
  {
    question: '为什么 RMSNorm 的 forward 方法中要用 .float() 和 .type_as(x)？',
    type: 'single',
    options: [
      { label: 'A', text: '为了节省内存' },
      { label: 'B', text: '为了提高计算速度' },
      { label: 'C', text: '为了避免 FP16/BF16 下的数值下溢' },
      { label: 'D', text: '为了兼容旧版 PyTorch' }
    ],
    correct: [2],
    explanation: `
      <strong>正确答案：C</strong><br><br>
      FP16/BF16 精度较低，归一化计算中的小数值容易下溢（变成 0）：
      <ul>
        <li><code>.float()</code>：转为 FP32，用高精度计算</li>
        <li><code>.type_as(x)</code>：转回原始数据类型，保持一致性</li>
      </ul>
      流程：输入 (BF16) → .float() (FP32) → 归一化 → .type_as(x) (BF16) → 输出
    `
  }
]
</script>

<InteractiveQuiz :questions="quizData" quiz-id="normalization" />

## 🎯 综合问答题

### Q6: 实战问题

假设你在训练一个 16 层的 Transformer，但 loss 在前 100 步就变成了 NaN。可能的原因和解决方案是什么？

<details>
<summary>点击查看参考答案</summary>

**可能原因**：
1. **没有使用归一化** → 梯度爆炸
2. **使用了 Post-LN** → 深层网络不稳定
3. **学习率太大** → 数值溢出
4. **权重初始化不当** → 初始激活值过大
5. **FP16 精度问题** → 数值下溢或溢出

**解决方案（按优先级）**：

1. **确保使用 Pre-LN + RMSNorm**：
   ```python
   class TransformerBlock(nn.Module):
       def forward(self, x):
           x = x + self.attn(self.norm1(x))  # Pre-Norm
           x = x + self.ffn(self.norm2(x))   # Pre-Norm
           return x
   ```

2. **降低学习率**：
   - 从 1e-3 降到 1e-4 或更小
   - 使用 warmup（前 N 步线性增加学习率）

3. **检查权重初始化**：
   - 使用 Kaiming 或 Xavier 初始化
   - 确保初始激活值在合理范围（std ≈ 1.0）

4. **使用梯度裁剪**：
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

5. **使用 BF16 而不是 FP16**：
   - BF16 数值范围更大，更稳定
   ```python
   with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
       output = model(input)
   ```

**调试步骤**：
1. 在每一层后打印激活值的统计量（均值、标准差、max/min）
2. 检查哪一层首先出现 NaN
3. 针对性地调整该层的配置

</details>

---

### Q7: 概念理解

用你自己的话解释："归一化就像给水龙头装水压稳定器"这个类比。

<details>
<summary>点击查看参考答案</summary>

**类比解释**：

**场景**：你家的水龙头

- **没有稳定器**：
  - 上游水压高 → 水龙头喷射（梯度爆炸）
  - 上游水压低 → 水龙头滴水（梯度消失）
  - 用水体验很差，无法控制

- **有稳定器**：
  - 无论上游水压如何变化
  - 稳定器调整后，输出水压始终稳定
  - 用水体验舒适，易于控制

**对应到神经网络**：

- **没有归一化**：
  - 前面层的激活值变化 → 后面层的输入不稳定
  - 数值可能爆炸（过大）或消失（过小）
  - 训练困难，模型难以收敛

- **有归一化（RMSNorm）**：
  - 每一层的输出都被"归一化"到标准范围
  - 无论输入如何变化，输出分布稳定（std ≈ 1.0）
  - 训练稳定，模型容易收敛

**核心思想**：
归一化不是改变信息内容，而是**控制信息的尺度**，让后续层更容易处理。

</details>

---

## ✅ 完成检查

完成所有题目后，检查你是否达到：

- [ ] **Q1-Q5 全对**：基础知识扎实
- [ ] **Q6 能提出 3+ 解决方案**：具备实战能力
- [ ] **Q7 能清晰解释类比**：深刻理解概念

如果还有不清楚的地方，回到 [teaching.md](./teaching.md) 复习，或重新运行实验代码。

---

## 🎓 进阶挑战

想要更深入理解？尝试：

1. **修改实验代码**：
   - 将 RMSNorm 替换为 LayerNorm，对比速度
   - 测试不同的 eps 值（1e-5 vs 1e-8）
   - 增加层数到 20 层，观察效果

2. **阅读论文**：
   - [RMSNorm 原始论文](https://arxiv.org/abs/1910.07467)
   - [Pre-LN vs Post-LN](https://arxiv.org/abs/2002.04745)

3. **实现变体**：
   - 实现 BatchNorm（在 batch 维度归一化）
   - 对比 BatchNorm vs LayerNorm vs RMSNorm

---

**下一步**：前往 [02. Position Encoding](../../02-position-encoding) 学习位置编码！
