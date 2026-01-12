---
title: Foundation（基础组件）模块 | minimind从零理解llm训练
description: 理解 Transformer 的基本构建块，包括归一化、位置编码、注意力机制和前馈网络。通过对照实验深入理解每个组件的设计原理。
keywords: Transformer基础组件, 归一化, 位置编码, 注意力机制, 前馈网络, LLM基础组件
---

# Tier 1: Foundation（基础组件）

> 理解 Transformer 的基本构建块

---

## 🎯 学习目标

完成本层后，你将能够：
- ✅ 解释每个基础组件的作用和数学原理
- ✅ 理解为什么现代 LLM 选择这些设计（而不是其他方案）
- ✅ 通过实验验证设计选择的合理性
- ✅ 从零实现这些组件

---

## 📚 模块列表

### [01. Normalization（归一化）](./01-normalization)

**核心问题**：
- 为什么深层网络需要归一化？
- RMSNorm vs LayerNorm：有什么区别？
- Pre-LN vs Post-LN：为什么 Pre-LN 更稳定？

**关键实验**：
- 实验 1：梯度消失可视化（无归一化 vs 有归一化）
- 实验 2：四种配置对比（NoNorm / Post-LN / Pre-LN / RMSNorm）
- 实验 3：精度影响（FP32 / FP16 / BF16）

**预计时长**：1 小时

---

### [02. Position Encoding（位置编码）](./02-position-encoding)

**核心问题**：
- 为什么 Attention 需要位置信息？
- RoPE 是如何工作的？
- 为什么 RoPE 能实现长度外推？

**关键实验**：
- 实验 1：无位置编码的排列不变性
- 实验 2：RoPE vs 绝对位置编码
- 实验 3：多频率机制可视化
- 实验 4：长度外推能力测试

**预计时长**：1.5 小时

---

### [03. Attention（注意力机制）](./03-attention)

**核心问题**：
- QKV 的直觉理解是什么？
- 为什么需要 Multi-Head Attention？
- GQA 如何提升效率？

**关键实验**：
- 实验 1：注意力权重可视化
- 实验 2：单头 vs 多头对比
- 实验 3：GQA 效率测试
- 实验 4：因果掩码的作用

**预计时长**：2 小时

---

### [04. FeedForward（前馈网络）](./04-feedforward)

**核心问题**：
- FFN 在 Transformer 中的作用是什么？
- 为什么需要扩张-压缩结构？
- SwiGLU vs ReLU：有什么区别？

**关键实验**：
- 实验 1：扩张比例的影响
- 实验 2：激活函数对比
- 实验 3：消融实验（去掉 FFN）

**预计时长**：1 小时

---

## 🚀 学习建议

### 推荐顺序

**必须按顺序学习**：
1. Normalization → 理解如何稳定训练
2. Position Encoding → 理解位置信息
3. Attention → 理解核心机制
4. FeedForward → 理解知识存储

原因：后面的模块会用到前面的概念。

### 学习方法

每个模块：
1. **先跑实验**（20 分钟）
   - 建立直觉："原来是这样的效果"
   - 不需要完全理解代码

2. **再看理论**（20 分钟）
   - 阅读 `teaching.md`
   - 理解数学公式和直觉类比

3. **读源码**（10 分钟）
   - 阅读 `code_guide.md`
   - 链接到 MiniMind 原始实现

4. **自测巩固**（10 分钟）
   - 完成 `quiz.md`
   - 检查理解程度

### 常见问题

**Q: 实验运行失败怎么办？**

A: 检查以下几点：
1. 是否激活了虚拟环境？ `source venv/bin/activate`
2. 是否下载了数据？ `cd modules/common && python datasets.py --download-all`
3. 是否在正确的目录？实验需要在 `experiments/` 目录下运行

**Q: 实验太慢怎么办？**

A: 使用快速模式：
```bash
python exp_xxx.py --quick
```
快速模式会减少训练步数（100 步），只验证概念。

**Q: 我想深入某个主题，有推荐资料吗？**

A: 每个模块的 `teaching.md` 最后都有"延伸阅读"部分，包含：
- 原始论文
- 相关博客
- 视频教程

---

## 📊 完成检查

完成本层后，你应该能够回答：

### 理论理解
- [ ] 为什么深层网络需要归一化？
- [ ] RoPE 如何编码相对位置信息？
- [ ] Attention 的 Q、K、V 分别代表什么？
- [ ] FFN 为什么要先扩张再压缩？

### 实践能力
- [ ] 能从零实现 RMSNorm
- [ ] 能从零实现 RoPE
- [ ] 能从零实现 Scaled Dot-Product Attention
- [ ] 能从零实现 SwiGLU FFN

### 设计直觉
- [ ] 能解释 Pre-LN 比 Post-LN 好在哪里
- [ ] 能解释 RoPE 比绝对位置编码好在哪里
- [ ] 能解释 Multi-Head 比 Single-Head 好在哪里
- [ ] 能解释为什么不能去掉 FFN

---

## 🎓 下一步

完成本层后，前往：
👉 [Tier 2: Architecture（架构组装）](../02-architecture)

学习如何将这些基础组件组装成完整的 Transformer Block。
