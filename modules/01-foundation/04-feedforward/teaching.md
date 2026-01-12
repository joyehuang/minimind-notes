---
title: FeedForward（前馈网络）教学文档 | minimind从零理解llm训练
description: 深入理解前馈网络的"扩张-压缩"结构和 SwiGLU 激活函数。通过对照实验理解 FFN 如何存储知识，以及为什么需要扩张因子。
keywords: 前馈网络, FeedForward, FFN, SwiGLU, 激活函数, Transformer前馈网络, LLM训练
---

# FeedForward（前馈网络）教学文档

> 理解"扩张-压缩"结构和 SwiGLU 激活函数

---

## 🤔 1. 为什么（Why）

### 问题场景：表达能力不足

**Attention 的局限**：
- Attention 负责"信息交换"
- 但只是加权平均，都是线性操作
- 无法表达复杂的非线性变换

**例子**：
```
输入：[0.5, 1.0, 0.8]  → 某个词的向量
目标：学习"这个词是动词还是名词"

需要的是复杂的非线性决策边界，而不是简单的线性组合
```

---

### 直觉理解：厨房加工

🍳 **类比**：FeedForward 就像厨房的加工过程

1. **输入**：生食材（768 维向量）
2. **扩张**：切碎、展开（768 → 2048 维）
   - 更细粒度的处理
   - 更多的"工作空间"
3. **激活**：加热、烹饪（非线性变换）
   - 产生化学反应
   - 不可逆的变化
4. **压缩**：装盘（2048 → 768 维）
   - 回到原来的维度
   - 但内容已经改变

**关键**：虽然输入输出维度相同，但内容已经经过"加工"。

---

### 数学本质

FeedForward 是一个**通用函数逼近器**：

1. **扩张**：映射到高维空间
2. **非线性激活**：创造非线性决策边界
3. **压缩**：保留有用信息，回到原维度

**理论支撑**（通用逼近定理）：
- 带有一个隐藏层的神经网络可以逼近任意连续函数
- 隐藏层越宽，逼近能力越强

---

## 📐 2. 是什么（What）

### 标准 FeedForward 结构

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2$$

**组成**：
- $W_1$：hidden_size → intermediate_size（扩张）
- ReLU：激活函数
- $W_2$：intermediate_size → hidden_size（压缩）

**典型配置**：
- intermediate_size = 4 × hidden_size
- MiniMind：512 → 2048 → 512

---

### 为什么要"扩张-压缩"？

**实验对比**：

```python
# 方案 A：直接变换 768 → 768
output_A = W_direct(x)  # 一个线性层

# 方案 B：扩张-压缩 768 → 2048 → 768
h = ReLU(W1(x))  # 768 → 2048
output_B = W2(h)  # 2048 → 768
```

**方案 A 的问题**：
- 只能表达线性变换
- 决策边界是超平面
- 无法分离复杂模式

**方案 B 的优势**：
- 高维空间中线性可分的概率更高
- 非线性激活创造复杂边界
- 压缩时保留判别性特征

**直觉**：
- 在 2D 空间中，一条直线无法分开环形分布的点
- 映射到 3D 空间后，一个平面就可以分开

---

### SwiGLU 激活函数

MiniMind 使用 SwiGLU，而不是简单的 ReLU：

$$\text{SwiGLU}(x) = W_{\text{down}} \cdot (\text{SiLU}(W_{\text{gate}} \cdot x) \odot W_{\text{up}} \cdot x)$$

**三个投影**：
1. `gate_proj`：$W_{\text{gate}} \cdot x$（计算门控信号）
2. `up_proj`：$W_{\text{up}} \cdot x$（计算值信号）
3. `down_proj`：压缩回原维度

**门控机制**：
```python
gate = SiLU(gate_proj(x))  # [batch, seq, intermediate]
up = up_proj(x)            # [batch, seq, intermediate]
hidden = gate * up         # 逐元素相乘（门控）
output = down_proj(hidden)
```

**为什么用门控？**
- 动态控制信息流
- gate 决定"让多少信息通过"
- 比单纯的激活函数更灵活

---

### SiLU（Swish）激活函数

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**特点**：
- 平滑：处处可导
- 非单调：小于 0 的部分不完全为 0
- 自门控：输入乘以自己的 sigmoid

**对比 ReLU**：
```
ReLU(x)  = max(0, x)
SiLU(x)  = x * sigmoid(x)

x = -1:
  ReLU(-1) = 0
  SiLU(-1) ≈ -0.27  # 保留部分负数信息
```

**为什么 SiLU 更好？**
- 梯度更平滑，训练更稳定
- 不完全"杀死"负数
- 在 LLM 中实验效果更好

---

### GLU 变体对比

| 变体 | 公式 | 特点 |
|------|------|------|
| **GLU** | $\sigma(W_1 x) \odot W_2 x$ | sigmoid 门控 |
| **ReGLU** | $\text{ReLU}(W_1 x) \odot W_2 x$ | ReLU 门控 |
| **GeGLU** | $\text{GELU}(W_1 x) \odot W_2 x$ | GELU 门控 |
| **SwiGLU** | $\text{SiLU}(W_1 x) \odot W_2 x$ | SiLU 门控 |

**LLM 最佳实践**：SwiGLU（Llama、MiniMind）或 GeGLU（GPT-J）

---

### FeedForward 与 Attention 的分工

**Transformer Block 结构**：
```
x → Norm → Attention → + → Norm → FeedForward → +
         (信息交换)   残差        (独立思考)   残差
```

**分工明确**：

| 组件 | 操作 | 类比 |
|------|------|------|
| **Attention** | 全局信息交换 | 开会讨论，听取他人意见 |
| **FeedForward** | 局部特征变换 | 各自思考，消化信息 |

**关键区别**：
- Attention：有 seq × seq 的交互矩阵
- FeedForward：每个位置完全独立处理

**为什么要独立处理？**
- Attention 已经做了信息融合
- FeedForward 负责"深度思考"
- 分开处理减少计算复杂度

---

### 参数量分析

**标准 FFN**：
```
W1: hidden_size × intermediate_size
W2: intermediate_size × hidden_size

total = 2 × hidden_size × intermediate_size
     = 2 × 512 × 2048 = 2M 参数
```

**SwiGLU**（三个投影）：
```
gate_proj: hidden_size × intermediate_size
up_proj:   hidden_size × intermediate_size
down_proj: intermediate_size × hidden_size

total = 3 × hidden_size × intermediate_size
     = 3 × 512 × (2048 × 2/3)  # 通常缩小 intermediate
     ≈ 2M 参数
```

**注意**：SwiGLU 为了保持参数量，通常将 intermediate_size 缩小到 2/3。

---

## 🔬 3. 怎么验证（How to Verify）

### 实验 1：FeedForward 基础

**目的**：理解扩张-压缩结构和 SwiGLU

**运行**：
```bash
python experiments/exp1_feedforward.py
```

**预期输出**：
- 展示维度变化过程
- 对比不同激活函数
- 可视化门控机制

---

## 💡 4. 关键要点总结

### 核心公式

**标准 FFN**：
$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x)$$

**SwiGLU**：
$$\text{SwiGLU}(x) = W_{\text{down}} \cdot (\text{SiLU}(W_{\text{gate}} \cdot x) \odot W_{\text{up}} \cdot x)$$

### 核心概念

| 概念 | 作用 | 关键点 |
|------|------|--------|
| 扩张 | 映射到高维 | 增加表达空间 |
| 激活 | 非线性变换 | 创造复杂边界 |
| 压缩 | 回到原维度 | 保留有用信息 |
| 门控 | 动态控制 | 选择性传递 |

### 设计原则

```python
# MiniMind 的 FeedForward 配置
hidden_size = 512
intermediate_size = 2048  # 4x 扩张（或调整以匹配 SwiGLU）

# SwiGLU 实现
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

---

## 📚 5. 延伸阅读

### 必读论文
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - GLU 系列详细对比
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始 Transformer FFN

### 推荐博客
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 可视化理解 FFN

### 代码实现
- MiniMind: `model/model_minimind.py:330-380` - FeedForward 实现

### 自测题
- 📝 [quiz.md](./quiz.md) - 完成自测题巩固理解

---

## 🎯 完成检查清单

学完本文档后，检查你是否能够：

- [ ] 解释为什么需要"扩张-压缩"结构
- [ ] 解释 SwiGLU 的三个投影的作用
- [ ] 解释 SiLU 与 ReLU 的区别
- [ ] 解释门控机制的作用
- [ ] 解释 FeedForward 与 Attention 的分工
- [ ] 从零实现 SwiGLU FeedForward

如果还有不清楚的地方，回到实验代码，动手验证！
