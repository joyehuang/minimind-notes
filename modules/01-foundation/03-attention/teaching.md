---
title: Attention（注意力机制）教学文档 | minimind从零理解llm训练
description: 深入理解 Self-Attention 如何让模型"关注"相关信息。通过对照实验理解 QKV 机制、多头注意力、缩放点积注意力的原理和实现。
keywords: 注意力机制, Attention, Self-Attention, QKV, 多头注意力, Multi-Head Attention, Transformer注意力, LLM注意力机制
---

# Attention（注意力机制）教学文档

> 理解 Self-Attention 如何让模型"关注"相关信息

---

## 🤔 1. 为什么（Why）

### 问题场景：理解词与词的关系

**例子**：
```
句子: "小明很喜欢他的猫，它总是在窗边睡觉"

问题：这里的"它"指的是什么？
```

人类可以轻松理解"它"指的是"猫"，而不是"小明"或"窗边"。但模型怎么知道呢？

**需要一种机制**：让模型学会"关注"句子中相关的部分。

---

### 直觉理解：图书馆查询

📚 **类比**：Self-Attention 就像在图书馆查书

1. **Query（查询）**：你想找什么？
   - "我想找关于猫的信息"

2. **Key（索引）**：每本书的关键词
   - 《小明传》→ 关键词：人物
   - 《猫的一生》→ 关键词：动物、猫
   - 《窗户设计》→ 关键词：建筑

3. **Value（内容）**：书的实际内容
   - 匹配后，你拿到的是书的内容

4. **Attention**：根据查询和索引的匹配度，决定从哪本书获取信息

---

### 数学本质

Attention 做三件事：

1. **计算相关性**：Query 和 Key 的点积
2. **归一化**：softmax 转为概率分布
3. **加权求和**：用概率分布对 Value 求和

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

---

## 📐 2. 是什么（What）

### Self-Attention 公式详解

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**分步解释**：

#### Step 1: 计算相关性分数 $QK^T$

```
Q: [batch, seq_len, d_k]
K: [batch, seq_len, d_k]

QK^T: [batch, seq_len, seq_len]
      ↑
    每两个 token 之间的相关性
```

**结果**：seq_len × seq_len 的矩阵，第 (i, j) 个元素表示 token_i 和 token_j 的相关性。

---

#### Step 2: 缩放 $/ \sqrt{d_k}$

**为什么要缩放？**
- 点积的方差随 $d_k$ 增大而增大
- 大数值会让 softmax 饱和（梯度接近 0）
- 除以 $\sqrt{d_k}$ 稳定方差

**例子**（d_k=64）：
- 未缩放：分数可能是 64（太大）
- 缩放后：分数约 8（合理）

---

#### Step 3: Softmax 归一化

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

- 将分数转为概率分布
- 所有位置的权重加起来 = 1
- 高分数 → 高权重

---

#### Step 4: 加权求和 $\times V$

```
attention_weights: [batch, seq_len, seq_len]
V: [batch, seq_len, d_v]

output: [batch, seq_len, d_v]
```

**效果**：每个 token 的输出是所有 Value 的加权平均，权重由相关性决定。

---

### Q、K、V 的生成

在 Self-Attention 中，Q、K、V 都来自同一个输入：

```python
# 输入 x: [batch, seq_len, hidden_dim]

Q = x @ W_Q  # [batch, seq_len, d_k]
K = x @ W_K  # [batch, seq_len, d_k]
V = x @ W_V  # [batch, seq_len, d_v]
```

**三个投影矩阵** $W_Q, W_K, W_V$ 是可学习参数。

**为什么不直接用 x 做 Q、K、V？**
- 投影矩阵让模型学习"从什么角度看待这个 token"
- Q：作为查询时应该强调什么
- K：作为被查询时应该展示什么
- V：实际应该传递什么内容

---

### Multi-Head Attention

**问题**：单头只能学习一种关系模式

**解决**：多个头并行，每个学不同模式

```python
# 8 个头
heads = []
for i in range(8):
    head_i = Attention(Q_i, K_i, V_i)
    heads.append(head_i)

# 拼接 + 投影
output = Concat(heads) @ W_O
```

**不同头学到的模式**：
- Head 1：语法关系（主谓宾）
- Head 2：语义相似（同义词）
- Head 3：位置关系（相邻词）
- Head 4：代词指代关系
- ...

---

### GQA（Grouped Query Attention）

**MHA 的问题**：KV Cache 太大

```
MHA: 每个头独立的 K、V
     8 heads × 512 seq × 64 dim = 262,144 参数/token
```

**GQA 的解决**：多个 Q 头共享 KV

```
GQA: 8 个 Q 头，2 个 KV 头
     每 4 个 Q 共享一组 KV
     内存减少 75%
```

**MiniMind 配置**：
```python
n_heads = 8        # Q 头数
n_kv_heads = 2     # KV 头数
# 4 个 Q 共享 1 个 KV
```

---

### 因果掩码（Causal Mask）

**问题**：语言模型只能看到"过去"的词

```
生成 "The cat sat" 时：
- "cat" 可以看到 "The"
- "sat" 可以看到 "The", "cat"
- 但 "The" 不能看到 "cat"（还没生成）
```

**解决**：用掩码遮盖"未来"的位置

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
# 上三角为 1（表示遮盖）

scores = scores.masked_fill(mask == 1, float('-inf'))
# softmax(−∞) = 0，相当于完全忽略
```

---

## 🔬 3. 怎么验证（How to Verify）

### 实验 1：Attention 基础

**目的**：理解 Attention 的计算过程

**运行**：
```bash
python experiments/exp1_attention_basics.py
```

**预期输出**：
- 展示 Attention 的排列不变性
- 可视化注意力权重矩阵

---

### 实验 2：Q、K、V 详解

**目的**：直观理解 Q、K、V 的作用

**运行**：
```bash
python experiments/exp2_qkv_explained.py
```

**预期输出**：
- 展示 Q、K、V 的生成过程
- 对比不同投影矩阵的效果

---

### 实验 3：Multi-Head Attention

**目的**：理解多头机制的优势

**运行**：
```bash
python experiments/exp3_multihead_attention.py
```

**预期输出**：
- 对比单头 vs 多头的效果
- 可视化不同头学到的模式

---

## 💡 4. 关键要点总结

### 核心公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 核心概念

| 概念 | 作用 | 类比 |
|------|------|------|
| Q (Query) | 我想查什么？ | 搜索关键词 |
| K (Key) | 我有什么标签？ | 索引标签 |
| V (Value) | 我的实际内容 | 文档内容 |
| $\sqrt{d_k}$ | 缩放因子 | 防止 softmax 饱和 |
| Multi-Head | 多种关系模式 | 多个角度看问题 |
| Causal Mask | 只看过去 | 语言模型的约束 |

### 设计原则

```python
# MiniMind 的 Attention 配置
n_heads = 8           # 8 个注意力头
n_kv_heads = 2        # GQA：2 个 KV 头
head_dim = 64         # 每个头 64 维
# hidden_size = 8 × 64 = 512
```

---

## 📚 5. 延伸阅读

### 必读论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [GQA Paper](https://arxiv.org/abs/2305.13245) - Grouped Query Attention

### 推荐博客
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)

### 代码实现
- MiniMind: `model/model_minimind.py:250-330` - Attention 实现
- MiniMind: `model/model_minimind.py:180-210` - GQA 实现

### 自测题
- 📝 [quiz.md](./quiz.md) - 完成自测题巩固理解

---

## 🎯 完成检查清单

学完本文档后，检查你是否能够：

- [ ] 写出 Attention 的完整公式
- [ ] 解释 Q、K、V 的作用和生成方式
- [ ] 解释缩放因子的必要性
- [ ] 解释 Multi-Head 的优势
- [ ] 解释 GQA 如何减少内存
- [ ] 解释因果掩码的作用
- [ ] 从零实现 Scaled Dot-Product Attention

如果还有不清楚的地方，回到实验代码，动手验证！
