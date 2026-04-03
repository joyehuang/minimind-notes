---
title: 快速体验 (30分钟) | minimind从零理解llm训练
description: 通过三个实验，30 分钟快速理解 LLM 训练的核心设计选择。
keywords: LLM快速体验, 归一化, RoPE, 注意力机制, 对照实验
---

# 快速体验（30 分钟）

通过三个简短实验，理解 LLM 训练中最重要的设计选择。

## 环境准备（5 分钟）

```bash
# 1. 克隆仓库
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 下载实验数据集（可选）
cd modules/common
python data_sources.py --download-all
cd ../..
```

## 实验 1：归一化（10 分钟）

观察梯度消失现象，理解为什么 Pre-LN + RMSNorm 能保持训练稳定。

```bash
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py
```

深入学习：[归一化模块](/modules/01-foundation/01-normalization/)

---

## 实验 2：RoPE 位置编码（10 分钟）

对比绝对位置编码，理解 RoPE 为什么能更好地进行长度外推。

```bash
cd ../../02-position-encoding/experiments
python exp1_rope_basics.py
```

深入学习：[位置编码模块](/modules/01-foundation/02-position-encoding/)

---

## 实验 3：注意力机制（10 分钟）

理解 Q/K/V 和注意力权重的工作原理。

```bash
cd ../../03-attention/experiments
python exp1_attention_basics.py
```

深入学习：[注意力机制模块](/modules/01-foundation/03-attention/)

---

## 接下来

- [系统学习（6 小时）](/docs/guide/systematic) — 完整掌握 Transformer 基础组件
- [深度掌握（30+ 小时）](/docs/guide/mastery) — 从零训练完整 LLM
- [学习路线图](/ROADMAP) — 查看完整路线
