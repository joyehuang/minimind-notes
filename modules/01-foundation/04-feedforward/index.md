---
title: FeedForward（前馈网络）模块 | minimind从零理解llm训练
description: 深入理解为什么需要"扩张-压缩"结构，以及 SwiGLU 激活函数有什么优势。通过对照实验理解 FFN 如何存储知识。
keywords: 前馈网络, FeedForward, FFN, SwiGLU, 激活函数, Transformer前馈网络, LLM训练
---

# 04. FeedForward（前馈网络）

> 为什么需要"扩张-压缩"？SwiGLU 有什么优势？

---

## 🎯 学习目标

完成本模块后，你将能够：
- ✅ 理解 FeedForward 的"扩张-压缩"结构
- ✅ 理解为什么需要高维中间层
- ✅ 理解 SwiGLU 激活函数的优势
- ✅ 理解 FeedForward 与 Attention 的分工
- ✅ 从零实现 SwiGLU FeedForward

---

## 📚 学习路径

### 1️⃣ 快速体验（10 分钟）

```bash
cd experiments

# 实验 1：FeedForward 基础
python exp1_feedforward.py
```

---

## 🔬 实验列表

| 实验 | 目的 | 时间 |
|------|------|------|
| exp1_feedforward.py | 理解扩张-压缩结构和 SwiGLU | 10分钟 |

---

## 💡 关键要点

### 1. FeedForward 的核心结构

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 \cdot x)$$

**标准 FFN**：
- $W_1$：hidden_size → intermediate_size（扩张）
- $\sigma$：激活函数（ReLU、GELU 等）
- $W_2$：intermediate_size → hidden_size（压缩）

**MiniMind SwiGLU**：
$$\text{SwiGLU}(x) = W_{\text{down}} \cdot (\text{SiLU}(W_{\text{gate}} \cdot x) \odot W_{\text{up}} \cdot x)$$

---

### 2. 为什么要"扩张-压缩"？

**直接变换的问题**：
- 768 → 768：只是线性变换，表达能力有限
- 无法拟合复杂的非线性函数

**扩张-压缩的优势**：
- 768 → 2048 → 768：经过高维空间
- 高维空间中更容易分离不同的模式
- 压缩回来时保留了有用信息

**类比**：
- 做菜：食材 → 切碎加工 → 装盘（维度相同但已加工）
- 照片：像素 → 特征提取 → 优化像素（质量提升）

---

### 3. SwiGLU 激活函数

**为什么不用 ReLU？**
- ReLU 简单但效果一般
- GLU 系列在 LLM 中表现更好

**SwiGLU 公式**：
```python
hidden = SiLU(gate) * up  # 门控机制
output = down_proj(hidden)
```

**三个投影**：
- `gate_proj`：计算门控信号
- `up_proj`：计算值信号
- `down_proj`：压缩回原维度

**SiLU（Swish）激活**：
$$\text{SiLU}(x) = x \cdot \sigma(x)$$

---

### 4. FeedForward 与 Attention 的分工

| 组件 | 职责 | 类比 |
|------|------|------|
| **Attention** | 词与词的信息交换 | 开会讨论 |
| **FeedForward** | 每个词的独立处理 | 各自思考 |

**关键区别**：
- Attention：有 seq_len × seq_len 的交互
- FeedForward：完全独立，每个位置分别处理

**Transformer Block 流程**：
```
x → RMSNorm → Attention → + → RMSNorm → FeedForward → +
              (信息交换)              (独立思考)
```

---

## 📖 文档

- 📘 [teaching.md](./teaching.md) - 完整的概念讲解
- 💻 [code_guide.md](./code_guide.md) - MiniMind 源码导读
- 📝 [quiz.md](./quiz.md) - 自测题

---

## ✅ 完成检查

学完本模块后，你应该能够：

### 理论
- [ ] 解释为什么需要"扩张-压缩"结构
- [ ] 解释 SwiGLU 的三个投影的作用
- [ ] 解释 SiLU 激活函数的特点
- [ ] 解释 FeedForward 与 Attention 的分工

### 实践
- [ ] 从零实现标准 FFN
- [ ] 从零实现 SwiGLU
- [ ] 对比不同激活函数的效果

---

## 🔗 相关资源

### 论文
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - GLU 系列激活函数
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始 FFN

### 代码实现
- MiniMind: `model/model_minimind.py:330-380`

---

## 🎓 下一步

完成本模块后，前往：
👉 [05. Residual Connection（残差连接）](../../02-architecture/05-residual-connection)
