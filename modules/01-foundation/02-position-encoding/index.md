# 02. Position Encoding（位置编码）

> 为什么 Transformer 需要位置信息？RoPE 如何工作？

---

## 🎯 学习目标

完成本模块后，你将能够：
- ✅ 理解 Attention 的排列不变性问题
- ✅ 理解 RoPE 的核心思想（旋转编码）
- ✅ 理解多频率机制的必要性
- ✅ 理解 RoPE 如何实现长度外推
- ✅ 从零实现 RoPE

---

## 📚 学习路径

### 1️⃣ 快速体验（10 分钟）

```bash
cd experiments

# 实验 1：证明 Attention 的排列不变性
python exp1_why_position.py

# 实验 2：RoPE 基础原理
python exp2_rope_basics.py --quick
```

---

## 🔬 实验列表

| 实验 | 目的 | 时间 | 数据 |
|------|------|------|------|
| exp1_why_position.py | 证明位置编码的必要性 | 30秒 | 合成 |
| exp2_rope_basics.py | RoPE 核心原理 | 2分钟 | 合成 |
| exp3_multi_frequency.py | 多频率机制 | 2分钟 | 合成 |
| exp4_rope_explained.py | 完整实现（可选） | 5分钟 | 合成 |

---

## 💡 关键要点

### 1. 为什么需要位置编码？

**问题**：Attention 是排列不变的
```python
# 两个句子包含相同的词
句子1: "我 喜欢 你" → Attention 无法区分
句子2: "你 喜欢 我" → 顺序不同，意思不同！
```

**解决**：给每个词标记位置信息

---

### 2. RoPE 的核心思想

**旋转编码**：用旋转矩阵编码位置
- 位置 0：旋转 0°
- 位置 1：旋转 θ°
- 位置 2：旋转 2θ°
- ...

**优势**：
- ✅ 相对位置信息自动产生（通过旋转角度差）
- ✅ 支持长度外推（训练 256，推理 512+）
- ✅ 计算高效（直接应用旋转）

---

### 3. 多频率机制

**为什么需要多个频率？**
- 低频（慢旋转）：编码远距离位置（如段落级别）
- 高频（快旋转）：编码近距离位置（如词级别）

**类比**：
- 钟表：秒针（高频）+ 分针（中频）+ 时针（低频）
- 共同作用才能准确表示时间

---

## 📖 文档

- 📘 [teaching.md](./teaching.md) - 完整的概念讲解
- 💻 [code_guide.md](./code_guide.md) - MiniMind 源码导读
- 📝 [quiz.md](./quiz.md) - 自测题

---

## ✅ 完成检查

学完本模块后，你应该能够：

### 理论
- [ ] 解释为什么 Attention 需要位置信息
- [ ] 说出 RoPE 的数学表达式
- [ ] 解释多频率机制的作用

### 实践
- [ ] 从零实现 RoPE
- [ ] 能画出 RoPE 的旋转示意图
- [ ] 能解释 RoPE 如何支持长度外推

---

## 🔗 相关资源

### 论文
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

### 代码实现
- MiniMind: `model/model_minimind.py:108-200`

---

## 🎓 下一步

完成本模块后，前往：
👉 [03. Attention（注意力机制）](../03-attention)
