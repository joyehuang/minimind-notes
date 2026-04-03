---
title: 系统学习 (6小时) | minimind从零理解llm训练
description: 系统掌握 Transformer 所有核心组件。学习归一化、位置编码、注意力机制和前馈网络，深入理解 LLM 训练原理。
keywords: LLM系统学习, Transformer教程, LLM基础, 大模型训练教程
---

# 📚 系统学习（6 小时）

> 完整掌握 Transformer 所有核心组件

## 🎯 学习目标

完成 6 小时学习后，你将能够：
- ✅ 理解 Transformer 所有核心组件
- ✅ 通过对照实验解释每个设计选择
- ✅ 从零实现一个简单的 Transformer

## 📋 学习路径

### 第一阶段：基础组件（5.5 小时）

按顺序学习四个核心模块：

#### 1. 归一化（1 小时）

**学习内容**：
- 📖 阅读 [teaching.md](/modules/01-foundation/01-normalization/teaching)（30 分钟）
- 🔬 运行所有实验（20 分钟）
- 📝 完成 [quiz.md](/modules/01-foundation/01-normalization/quiz)（10 分钟）

**完成标准**：
- [ ] 能解释梯度消失/爆炸现象
- [ ] 能从零实现 RMSNorm
- [ ] 理解 Pre-LN vs Post-LN 的区别

[开始学习 →](/modules/01-foundation/01-normalization/)

---

#### 2. 位置编码（1.5 小时）

**学习内容**：
- 📖 阅读 [teaching.md](/modules/01-foundation/02-position-encoding/teaching)（40 分钟）
- 🔬 运行实验 1-3（40 分钟）
- 📝 自测复习（10 分钟）

**完成标准**：
- [ ] 理解 Attention 的排列不变性问题
- [ ] 能解释 RoPE 的旋转思想
- [ ] 理解多频率分量的作用

[开始学习 →](/modules/01-foundation/02-position-encoding/)

---

#### 3. 注意力机制（2 小时）

**学习内容**：
- 🔬 运行所有实验（1.5 小时）
- 💻 阅读源码实现（30 分钟）

**完成标准**：
- [ ] 理解 Q、K、V 各自的角色
- [ ] 理解多头注意力的优势
- [ ] 理解 GQA（分组查询注意力）

[开始学习 →](/modules/01-foundation/03-attention/)

---

#### 4. 前馈网络（1 小时）

**学习内容**：
- 🔬 运行实验（40 分钟）
- 💻 理解 SwiGLU 激活函数（20 分钟）

**完成标准**：
- [ ] 理解 FFN 的"扩张-压缩"模式
- [ ] 理解 Attention 和 FFN 的分工
- [ ] 能从零实现 SwiGLU

[开始学习 →](/modules/01-foundation/04-feedforward/)

---

### 第二阶段：架构组装（0.5 小时）

**学习内容**：
- 📖 阅读[架构总览](/modules/02-architecture/)（30 分钟）
- 理解基础组件如何组装成 Transformer Block

**完成标准**：
- [ ] 能画出 Pre-LN Transformer Block 的数据流
- [ ] 理解残差连接的作用
- [ ] 能从零实现一个 Transformer Block

---

## 🎯 总检查清单

完成系统学习后，确认以下内容：

### 基础模块
- [ ] ✅ 完成归一化模块
- [ ] ✅ 完成位置编码模块
- [ ] ✅ 完成注意力机制模块
- [ ] ✅ 完成前馈网络模块

### 实践能力
- [ ] ✅ 能从零实现 Transformer Block
- [ ] ✅ 通过所有模块自测题
- [ ] ✅ 能解释每个设计选择的原因

---

## 📚 下一步

想要更深入？
- 🎓 [深度掌握（30+ 小时）](/docs/guide/mastery) — 从零训练完整 LLM
- 📝 [记录学习笔记](/learning_log) — 跟踪学习进度
- 🗺️ [完整路线图](/ROADMAP) — 查看完整学习路径
