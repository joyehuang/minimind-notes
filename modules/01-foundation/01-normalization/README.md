---
title: Normalization（归一化）模块导航 | minimind从零理解llm训练
description: Normalization（归一化）模块学习导航，包含教学文档、代码导读和自测题。理解为什么深层网络需要归一化，以及 RMSNorm 如何稳定训练。
keywords: 归一化, Normalization, RMSNorm, LayerNorm, Pre-LN, Post-LN, LLM训练教程
---

# 01. Normalization（归一化）

> 为什么深层网络需要归一化？RMSNorm 如何稳定训练？

---

## 🎯 学习目标

完成本模块后，你将能够：
- ✅ 理解梯度消失/爆炸问题的本质
- ✅ 理解归一化如何稳定激活分布
- ✅ 理解 RMSNorm vs LayerNorm 的区别
- ✅ 理解 Pre-LN vs Post-LN 的优劣
- ✅ 从零实现 RMSNorm

---

## 📚 学习路径

### 1️⃣ 快速体验（10 分钟）

先运行实验，建立直觉：

```bash
cd experiments

# 实验 1：看看没有归一化会发生什么
python exp1_gradient_vanishing.py

# 实验 2：对比四种配置
python exp2_norm_comparison.py --quick
```

**你会看到**：
- 无归一化：激活值标准差迅速衰减到 0（梯度消失）
- Post-LN：训练初期 loss 震荡
- Pre-LN + RMSNorm：稳定收敛 ✅

---

### 2️⃣ 理论学习（30 分钟）

阅读教学文档：
- 📖 [teaching.md](./teaching.md) - 完整的概念讲解

**核心内容**：
- **Why**: 为什么需要归一化？
- **What**: RMSNorm 是什么？
- **How**: 如何验证？

---

### 3️⃣ 代码实现（15 分钟）

查看真实代码：
- 💻 [code_guide.md](./code_guide.md) - MiniMind 源码导读

**关键文件**：
- `model/model_minimind.py:95-105` - RMSNorm 实现
- `model/model_minimind.py:178-194` - MiniMindBlock 中的使用

---

### 4️⃣ 自测巩固（5 分钟）

完成自测题：
- 📝 [quiz.md](./quiz.md) - 5 道选择题

---

## 🔬 实验列表

| 实验 | 目的 | 时间 | 数据 |
|------|------|------|------|
| exp1_gradient_vanishing.py | 证明归一化的必要性 | 10秒 | 合成数据 |
| exp2_norm_comparison.py | 对比四种配置 | 5分钟 | TinyShakespeare |
| exp3_prenorm_vs_postnorm.py | Pre-LN vs Post-LN | 8分钟 | TinyShakespeare |

### 运行所有实验

```bash
cd experiments
bash run_all.sh
```

---

## 💡 关键要点

### 1. 为什么需要归一化？

```
无归一化的深层网络：
Layer 1: std = 1.00
Layer 2: std = 0.85
Layer 3: std = 0.62
Layer 4: std = 0.38
...
Layer 8: std = 0.016  ← 梯度几乎消失！
```

**直觉**：像水龙头装"水压稳定器"，无论输入如何变化，输出都保持稳定。

---

### 2. RMSNorm vs LayerNorm

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 计算步骤 | 减均值 + 除标准差 | 除 RMS |
| 参数量 | 2d (γ, β) | d (γ) |
| 速度 | 慢 | 快 7-64% |
| 半精度稳定性 | 较差 | 更好 |

**结论**：RMSNorm 更简单、更快、更稳定。

---

### 3. Pre-LN vs Post-LN

```python
# Post-LN（旧方案）
x = x + Attention(x)          # 先计算
x = LayerNorm(x)              # 后归一化

# Pre-LN（现代方案）
x = x + Attention(Norm(x))    # 先归一化
x = x + FFN(Norm(x))          # 再计算
```

**优势**：
- ✅ 残差路径更"干净"（梯度直达）
- ✅ 深层网络（>12 层）更稳定
- ✅ 学习率容忍度更高

---

## 📊 预期结果

完成实验后，你会在 `experiments/results/` 看到：

1. **gradient_vanishing.png**
   - 无归一化：标准差衰减曲线
   - 有归一化：标准差保持稳定

2. **norm_comparison.png**
   - 四条 loss 曲线对比
   - NoNorm 在 500 步后 NaN
   - Pre-LN 最稳定

3. **prenorm_vs_postnorm.png**
   - Pre-LN vs Post-LN 的收敛速度对比

---

## ✅ 完成检查

学完本模块后，你应该能够：

### 理论
- [ ] 用自己的话解释梯度消失问题
- [ ] 说出 RMSNorm 的数学公式
- [ ] 解释 Pre-LN 为什么比 Post-LN 好

### 实践
- [ ] 从零实现 RMSNorm
- [ ] 能画出 Pre-LN Transformer Block 的数据流图
- [ ] 能调试归一化相关的训练问题

### 直觉
- [ ] 能用类比解释归一化的作用
- [ ] 能预测去掉归一化会发生什么
- [ ] 能解释为什么现代 LLM 选择 Pre-LN + RMSNorm

---

## 🔗 相关资源

### 论文
- [RMSNorm](https://arxiv.org/abs/1910.07467) - Root Mean Square Layer Normalization
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - Pre-LN vs Post-LN

### 博客
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Understanding Layer Normalization](https://leimao.github.io/blog/Layer-Normalization/)

### 视频
- [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

## 🎓 下一步

完成本模块后，前往：
👉 [02. Position Encoding（位置编码）](../02-position-encoding)

学习 Transformer 如何处理位置信息，理解 RoPE 的原理。
