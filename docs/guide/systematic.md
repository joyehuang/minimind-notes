---
title: 系统学习 (6小时)
description: 完整掌握Transformer所有基础组件
---

# 📚 系统学习 (6小时)

> 完整掌握 Transformer 的所有基础组件

## 🎯 学习目标

6 小时后你将掌握:
- ✅ Transformer 的所有基础组件
- ✅ 每个设计选择的原因（通过对照实验）
- ✅ 从零实现一个简单的 Transformer

## 📋 学习路径

### 阶段 1: Foundation（基础组件）- 5.5 小时

按顺序学习 4 个核心模块:

#### 1. Normalization (1小时)

**学习内容**:
- 📖 阅读 [teaching.md](/modules/01-foundation/01-normalization/teaching)（30分钟）
- 🔬 运行所有实验（20分钟）
- 📝 完成 [quiz.md](/modules/01-foundation/01-normalization/quiz)（10分钟）

**完成标准**:
- [ ] 能解释梯度消失/爆炸问题
- [ ] 能从零实现 RMSNorm
- [ ] 理解 Pre-LN vs Post-LN 的区别

[开始学习 →](/modules/01-foundation/01-normalization/)

---

#### 2. Position Encoding (1.5小时)

**学习内容**:
- 📖 阅读 [teaching.md](/modules/01-foundation/02-position-encoding/teaching)（40分钟）
- 🔬 运行实验 1-3（40分钟）
- 📝 自测（10分钟）

**完成标准**:
- [ ] 理解 Attention 的排列不变性
- [ ] 能解释 RoPE 的旋转原理
- [ ] 理解多频率机制的作用

[开始学习 →](/modules/01-foundation/02-position-encoding/)

---

#### 3. Attention (2小时)

**学习内容**:
- 🔬 运行所有实验（1.5小时）
- 💻 阅读源码（30分钟）

**完成标准**:
- [ ] 理解 Q、K、V 的作用
- [ ] 理解 Multi-Head 的优势
- [ ] 理解 GQA（Grouped Query Attention）

[开始学习 →](/modules/01-foundation/03-attention/)

---

#### 4. FeedForward (1小时)

**学习内容**:
- 🔬 运行实验（40分钟）
- 💻 理解 SwiGLU 激活函数（20分钟）

**完成标准**:
- [ ] 理解 FFN 的扩张-压缩机制
- [ ] 理解 Attention vs FFN 的分工
- [ ] 能从零实现 SwiGLU

[开始学习 →](/modules/01-foundation/04-feedforward/)

---

### 阶段 2: Architecture（架构组装）- 0.5 小时

**学习内容**:
- 📖 阅读 [Architecture README](/modules/02-architecture/)（30分钟）
- 理解如何将组件组装成 Transformer Block

**完成标准**:
- [ ] 能画出 Pre-LN Transformer Block 的数据流图
- [ ] 理解残差连接的作用
- [ ] 能从零实现一个 Transformer Block

---

## 🎯 检查清单

完成系统学习后，确保你能做到:

### Foundation 模块
- [ ] ✅ 完成 Normalization 模块
- [ ] ✅ 完成 Position Encoding 模块
- [ ] ✅ 完成 Attention 模块
- [ ] ✅ 完成 FeedForward 模块

### 实践能力
- [ ] ✅ 能从零实现 Transformer Block
- [ ] ✅ 通过所有模块的自测题
- [ ] ✅ 理解每个设计选择的原因

---

## 📚 下一步

想继续深入？
- 🎓 [深度掌握 (30小时)](./mastery) - 从零训练一个完整的 LLM
- 📝 [记录笔记](/learning_log) - 记录你的学习过程
- 🗺️ [完整路线图](/ROADMAP) - 查看完整学习路径
