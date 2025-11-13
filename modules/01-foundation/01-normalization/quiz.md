# Normalization 自测题

> 完成以下题目检验你的理解程度

---

## 📝 选择题

### Q1: 梯度消失问题的根本原因是什么？

A. 学习率设置太小
B. 深层网络中激活值的分布随层数变化而失控
C. 优化器选择不当
D. 数据集太小

<details>
<summary>点击查看答案</summary>

**答案：B**

**解析**：
- 梯度消失的根本原因是深层网络中，激活值的标准差会随着层数增加而衰减
- 当标准差变得非常小（接近 0）时，反向传播的梯度也会变小
- 最终导致前面几层的权重几乎无法更新
- 归一化通过控制每一层的激活分布来解决这个问题

</details>

---

### Q2: RMSNorm 和 LayerNorm 的主要区别是什么？

A. RMSNorm 有更多可学习参数
B. RMSNorm 不减均值，只除以 RMS
C. RMSNorm 只能用于 Transformer
D. RMSNorm 效果比 LayerNorm 差

<details>
<summary>点击查看答案</summary>

**答案：B**

**解析**：
- LayerNorm: $(x - \mu) / \sigma$（减均值，除以标准差）
- RMSNorm: $x / \text{RMS}(x)$（只除以均方根，不减均值）
- RMSNorm 更简单、更快（减少一步计算）
- 在 LLM 训练中效果相当，但 RMSNorm 速度提升 7-64%

</details>

---

### Q3: Pre-LN 相比 Post-LN 的优势是什么？

A. 计算速度更快
B. 使用更少参数
C. 残差路径更干净，梯度流更稳定
D. 不需要学习率调整

<details>
<summary>点击查看答案</summary>

**答案：C**

**解析**：

**Post-LN**（旧方案）：
```python
x = LayerNorm(x + Attention(x))  # 残差在 Norm 之前
```
- 梯度需要经过 LayerNorm，可能被打断

**Pre-LN**（现代方案）：
```python
x = x + Attention(Norm(x))  # Norm 在子层之前
```
- 残差路径上没有 Norm，梯度可以直接传播
- 深层网络（>12 层）更稳定
- 学习率容忍度更高

</details>

---

### Q4: 在 MiniMind 的一个 TransformerBlock 中，有几个 RMSNorm？

A. 1 个
B. 2 个
C. 4 个
D. 8 个

<details>
<summary>点击查看答案</summary>

**答案：B**

**解析**：
每个 TransformerBlock 有 **2 个 RMSNorm**：
1. **attention_norm**：在 Attention 之前
2. **ffn_norm**：在 FeedForward 之前

数据流：
```
x → Norm #1 → Attention → + residual
  → Norm #2 → FFN → + residual → output
```

MiniMind-small 有 8 个 Block，所以总共 16 个 RMSNorm。

</details>

---

### Q5: 为什么 RMSNorm 的 forward 方法中要用 `.float()` 和 `.type_as(x)`？

A. 为了节省内存
B. 为了提高计算速度
C. 为了避免 FP16/BF16 下的数值下溢
D. 为了兼容旧版 PyTorch

<details>
<summary>点击查看答案</summary>

**答案：C**

**解析**：
```python
def forward(self, x):
    output = self._norm(x.float()).type_as(x)  # ← 关键
    return output * self.weight
```

**原因**：
- FP16/BF16 精度较低，归一化计算中的小数值容易下溢（变成 0）
- `.float()`：转为 FP32，用高精度计算
- `.type_as(x)`：转回原始数据类型，保持一致性

**流程**：
```
输入 (BF16) → .float() (FP32) → 归一化 → .type_as(x) (BF16) → 输出
```

</details>

---

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
