---
title: 快速体验 (30分钟) | minimind从零理解llm训练
description: 通过3个关键实验快速理解LLM核心设计选择。适合想要快速了解大模型训练原理的同学，特别是准备大模型岗位面试的同学。
keywords: LLM快速入门, 大模型训练入门, Transformer快速学习, LLM面试准备, 30分钟学习LLM
---

# ⚡ 快速体验 (30分钟)

> 通过 3 个关键实验，理解现代 LLM 的核心设计选择

## 🎯 学习目标

30 分钟后你将理解:
- ✅ 为什么现代 LLM 用 Pre-LN + RMSNorm
- ✅ 为什么 RoPE 比绝对位置编码好
- ✅ Attention 的数学原理和直觉

## 🚀 准备环境 (5分钟)

```bash
# 1. 克隆仓库
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 下载实验数据（可选，部分实验不需要）
cd modules/common
python datasets.py --download-all
cd ../..
```

## 🔬 实验 1: 为什么需要归一化？ (10分钟)

**问题**: 深层网络为什么难训练？

```bash
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py
```

**你会看到**:
- ❌ 无归一化：激活标准差从 1.0 衰减到 0.016（梯度消失）
- ✅ 有 RMSNorm：标准差保持稳定在 1.0 附近

**关键发现**: 归一化是深层网络训练的"稳定器"

[深入了解 →](/modules/01-foundation/01-normalization/)

---

## 🔬 实验 2: 为什么用 RoPE 位置编码？ (10分钟)

**问题**: Transformer 如何知道词的顺序？

```bash
cd ../../02-position-encoding/experiments
python exp1_rope_basics.py
```

**你会看到**:
- Attention 本身是"排列不变"的（无法区分顺序）
- RoPE 通过旋转向量编码位置信息
- 自动产生相对位置关系

**关键发现**: RoPE 既有绝对位置，又有相对位置，还支持长度外推

[深入了解 →](/modules/01-foundation/02-position-encoding/)

---

## 🔬 实验 3: Attention 如何工作？ (10分钟)

**问题**: Attention 的 Q、K、V 是什么意思？

```bash
cd ../../03-attention/experiments
python exp1_attention_basics.py
```

**你会看到**:
- Query（查询）：我想找什么信息？
- Key（键）：我提供什么信息？
- Value（值）：找到后传递什么内容？
- 注意力权重可视化

**关键发现**: Attention 让模型自动学习"哪些词相关"

[深入了解 →](/modules/01-foundation/03-attention/)

---

## 🎯 下一步

想深入学习？继续:
- 📚 [系统学习 (6小时)](./systematic) - 完整掌握 Transformer 所有基础组件
- 🎓 [深度掌握 (30小时)](./mastery) - 从零训练一个完整的 LLM
- 🗺️ [完整路线图](/ROADMAP) - 查看完整学习路径
