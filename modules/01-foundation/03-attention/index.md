---
title: Attention（注意力机制）模块 | minimind从零理解llm训练
description: 深入理解 Self-Attention 如何工作，以及 Q、K、V 的直觉。通过对照实验理解多头注意力、缩放点积注意力的原理和实现。
keywords: 注意力机制, Attention, Self-Attention, QKV, 多头注意力, Transformer注意力, LLM训练
---

# 03. Attention（注意力机制）

> Self-Attention 如何工作？Q、K、V 的直觉是什么？

---

## 🎯 学习目标

完成本模块后，你将能够：
- ✅ 理解 Self-Attention 的数学原理
- ✅ 理解 Q、K、V 的直觉意义
- ✅ 理解 Multi-Head Attention 的优势
- ✅ 理解 GQA（Grouped Query Attention）
- ✅ 从零实现 Scaled Dot-Product Attention

---

## 📚 学习路径

### 1️⃣ 快速体验（15 分钟）

```bash
cd experiments

# 实验 1：Attention 基础
python exp1_attention_basics.py

# 实验 2：Q、K、V 详解
python exp2_qkv_explained.py
```

---

## 🔬 实验列表

| 实验 | 目的 | 时间 |
|------|------|------|
| exp1_attention_basics.py | Attention 的排列不变性和基础计算 | 5分钟 |
| exp2_qkv_explained.py | Q、K、V 的直觉理解 | 5分钟 |
| exp3_multihead_attention.py | Multi-Head 机制详解 | 10分钟 |

---

## 💡 关键要点

### 1. Self-Attention 的核心公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**直觉**：
- $QK^T$：计算"相关性分数"（谁和谁相关）
- $\sqrt{d_k}$：缩放因子（防止 softmax 饱和）
- softmax：归一化为概率分布
- $\times V$：加权求和（提取相关信息）

---

### 2. Q、K、V 的直觉

| 角色 | 全称 | 问题 | 类比 |
|------|------|------|------|
| **Q** | Query | 我想找什么？ | 图书馆查询 |
| **K** | Key | 我提供什么标签？ | 书的关键词 |
| **V** | Value | 找到后给什么？ | 书的内容 |

**例子**：处理句子 "The cat sat on the mat"
- "cat" 的 Query："我想找与动物相关的信息"
- "sat" 的 Key："我是一个动作词"
- 如果 Q·K 匹配 → 取出 "sat" 的 Value

---

### 3. 为什么需要 Multi-Head？

**问题**：单头注意力只能学习一种"关系模式"

**解决**：多头并行，每个头学习不同模式
- Head 1：语法关系（主谓宾）
- Head 2：语义关系（同义词）
- Head 3：位置关系（相邻词）
- ...

**公式**：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

---

### 4. GQA（Grouped Query Attention）

**MHA**（Multi-Head Attention）：每个头有独立的 Q、K、V
- 参数：3 × n_heads × head_dim

**GQA**：多个 Q 头共享一组 K、V
- 参数：(n_heads + 2 × n_kv_heads) × head_dim
- 减少 KV Cache，加速推理

**MiniMind**：`n_heads=8, n_kv_heads=2`
- 8 个 Q 头，2 个 KV 头
- 每 4 个 Q 头共享一组 KV

---

## 📖 文档

- 📘 [teaching.md](./teaching.md) - 完整的概念讲解
- 💻 [code_guide.md](./code_guide.md) - MiniMind 源码导读
- 📝 [quiz.md](./quiz.md) - 自测题

---

## ✅ 完成检查

学完本模块后，你应该能够：

### 理论
- [ ] 写出 Attention 的数学公式
- [ ] 解释 Q、K、V 的作用
- [ ] 解释缩放因子 $\sqrt{d_k}$ 的作用
- [ ] 解释 Multi-Head 的优势

### 实践
- [ ] 从零实现 Scaled Dot-Product Attention
- [ ] 可视化注意力权重
- [ ] 理解因果掩码（Causal Mask）

---

## 🔗 相关资源

### 论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245)

### 代码实现
- MiniMind: `model/model_minimind.py:250-330`

---

## 🎓 下一步

完成本模块后，前往：
👉 [04. FeedForward（前馈网络）](../04-feedforward)
