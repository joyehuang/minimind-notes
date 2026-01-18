---
title: 知识库 | minimind从零理解llm训练
description: 系统化整理的技术知识、概念解释和问答记录。包含归一化机制、位置编码、注意力机制等 LLM 核心概念的详细解释。
keywords: LLM知识库, Transformer知识库, 大模型知识库, 深度学习知识库, LLM概念解释
---

# MiniMind 知识库

> 系统化整理的知识点、概念解释和问答记录

---

## 📑 目录

1. [归一化机制](#1-归一化机制)
   - RMSNorm 原理
   - LayerNorm vs RMSNorm
2. [位置编码](#2-位置编码)
   - RoPE 基本原理
   - 多频率机制
3. [注意力机制](#3-注意力机制)
   - Self-Attention 原理
   - Q、K、V 详解
   - Multi-Head Attention
   - RoPE 在 Attention 中的应用
4. [前馈网络](#4-前馈网络feedforward)
   - FeedForward 原理
   - 扩张-压缩机制
   - SwiGLU 实现
   - Attention vs FeedForward 对比
5. [Transformer 架构](#5-transformer-架构)
6. [问答记录](#问答记录)

---

## 1. 归一化机制

### 1.1 为什么需要归一化？

在没有归一化的深层神经网络中，会出现**梯度消失或梯度爆炸**问题：

**梯度消失**：
- 现象：数值越来越小，最终接近 0
- 实验证据：8 层网络，标准差从 1.04 → 0.016
- 后果：梯度接近 0，权重几乎不更新，模型学不到东西

**梯度爆炸**：
- 现象：数值越来越大，最终溢出
- 后果：数值变成 NaN（Not a Number），训练崩溃

**类比**：
归一化就像给水龙头装"水压稳定器"，无论输入水压多大或多小，输出都保持稳定。

---

### 1.2 RMSNorm 原理

**全称**：Root Mean Square Normalization（均方根归一化）

**数学公式**：
```
x_norm = x / sqrt(mean(x²) + eps) * weight
```

其中：
- `x`：输入向量
- `mean(x²)`：向量元素平方的平均值
- `eps`：防止除零的小常数（1e-5）
- `weight`：可学习的缩放参数

**关键特点**：
1. **只做缩放，不减均值**（比 LayerNorm 简单）
2. **保持向量方向不变**，只调整大小
3. **信息不丢失**，只是控制数值规模
4. **计算高效**，参数更少

**代码实现**（`model/model_minimind.py:95-105`）：
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
```

---

### 1.3 LayerNorm vs RMSNorm 对比

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| **公式** | `(x - mean) / std` | `x / sqrt(mean(x²))` |
| **计算步骤** | 1. 计算均值<br>2. 减均值<br>3. 除以标准差 | 1. 除以 RMS |
| **均值** | 强制为 0 | 不强制（可以不是 0）|
| **速度** | 较慢 | **快 7.7 倍**（实测）|
| **参数** | weight + bias | 只有 weight |
| **效果** | 很好 | 在 LLM 上相当或更好 |
| **使用模型** | BERT, GPT-2 | Llama, MiniMind, GPT-3+ |

**为什么 RMSNorm 能省略减均值？**
- 在深度网络中，激活值的分布通常已经接近零均值
- 只控制标准差就足够稳定训练了
- 省略减均值步骤 → 计算更快

---

### 1.4 RMSNorm 在 Transformer 中的位置

RMSNorm **不是单独一层**，而是 Transformer Block 内部的组件：

```
Transformer Block（MiniMind 有 8 个）
├─ RMSNorm #1 ← 在 Attention 之前
├─ Self-Attention
├─ 残差连接（x + Attention(x)）
├─ RMSNorm #2 ← 在 FeedForward 之前
├─ FeedForward
└─ 残差连接（x + FeedForward(x)）
```

**统计**：
- 每个 Block 有 **2 个 RMSNorm**
- MiniMind 有 8 个 Block
- 总共有 **16 个 RMSNorm**

---

## 2. 位置编码

### 2.1 为什么需要位置编码？

**问题**：Attention 机制本身是"排列不变"的（permutation invariant）

**例子**：
- 句子1："我喜欢你"
- 句子2："你喜欢我"
- 包含的词相同：{我, 喜欢, 你}
- 如果没有位置编码，Attention 无法区分这两个句子！

**解决**：需要给每个词"标记位置"

---

### 2.2 位置编码的三代演进

**第1代：绝对位置编码**（BERT）
- 做法：给每个位置（0, 1, 2, ...）分配一个固定的向量
- 问题：无法外推到训练时未见过的长度

**第2代：相对位置编码**（T5）
- 做法：记录两个词之间的相对距离
- 问题：计算复杂，难以优化

**第3代：RoPE（旋转位置编码）**（Llama/MiniMind）⭐️
- 做法：通过旋转向量来编码位置
- 优点：
  - ✅ 自然包含相对位置信息
  - ✅ 计算高效
  - ✅ 可以外推到更长的序列（YaRN）

---

### 2.3 RoPE 核心原理

**基本思想**：用旋转角度编码位置

```
位置 0 → 旋转 0°
位置 1 → 旋转 θ°
位置 2 → 旋转 2θ°
位置 3 → 旋转 3θ°
...
```

**关键性质**：相对位置 = 相对旋转角度

**数学证明**（简化版）：
- 词A在位置5：旋转 `5θ`
- 词B在位置8：旋转 `8θ`
- 点积：`rotate(A, 5θ) · rotate(B, 8θ)`
- 根据三角函数性质：`= A · B · cos((8-5)θ) = A · B · cos(3θ)`
- **只依赖相对距离3！**

**优势**：
1. **有绝对位置信息**：每个词被旋转到特定角度
2. **有相对位置信息**：Attention 分数只依赖相对距离
3. **两全其美**！

---

### 2.4 RoPE 的多频率机制

**问题**：旋转360度不是回到原点了吗？

**解决**：使用多个不同频率的旋转！

**类比钟表**：
- 秒针：转得快（1分钟转360度）→ 精确到秒
- 分针：转得慢（1小时转360度）→ 精确到分
- 时针：转得超慢（12小时转360度）→ 精确到时

**MiniMind 的实际参数**（head_dim=64，使用32个频率）：
```
频率0（高频）：每 6.3 个token转一圈      → 编码局部位置
频率15（中频）：每 1,000 个token转一圈   → 编码中等距离
频率31（低频）：每 6,283,185 个token转一圈 → 编码全局位置
```

**频率计算公式**：
```python
freqs[i] = 1 / (rope_base ^ (2i / dim))
```

其中：
- `rope_base = 1000000.0`（MiniMind 的值）
- `dim = head_dim = 64`
- `i = 0, 1, 2, ..., 31`

**效果**：32个频率组合，可以唯一标识百万级别的位置！

---

### 2.5 RoPE 的实现细节

**预计算旋转频率**（`model/model_minimind.py:108-128`）：
```python
def precompute_freqs_cis(dim, end, rope_base=1e6, rope_scaling=None):
    # 计算频率
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))

    # YaRN 长度外推（可选）
    if rope_scaling is not None and end > original_max:
        # 应用 YaRN 缩放
        ...

    # 为每个位置生成 cos 和 sin
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

    return freqs_cos, freqs_sin
```

**应用旋转**（`model/model_minimind.py:131-137`）：
```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    def rotate_half(x):
        # 将向量分成两半并交换
        return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)

    # 旋转公式
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
```

**为什么只旋转 Q 和 K，不旋转 V？**
- 位置信息只需要影响"匹配度"（Q·K）
- 不需要影响"内容"（V）

---

## 3. 注意力机制

### 3.1 什么是 Attention？

**核心思想**：让模型理解词与词之间的关系

**例子**：
```
句子: "小明很喜欢他的猫，它总是在窗边睡觉"

当理解"它"时：
  "它" 和 "小明" 的相关性: 0.1  （关系不大）
  "它" 和 "喜欢" 的相关性: 0.05 （关系很小）
  "它" 和 "猫" 的相关性:   0.8  （关系很大！）✅
  "它" 和 "窗边" 的相关性: 0.05 （关系很小）

最终理解 = 0.1×小明 + 0.05×喜欢 + 0.8×猫 + 0.05×窗边
         ≈ 主要吸收"猫"的信息
```

**Attention 做什么**：
1. 计算每两个 token 之间的相关性分数
2. 归一化成概率分布（加起来 = 1）
3. 用这个分布加权求和，得到融合了上下文的新表示

---

### 3.2 Self-Attention vs Cross-Attention

**Self-Attention**（自注意力）：
- 句子关注"自己内部"的词
- 例如："我爱编程" 内部计算 我←→爱←→编程 的关系
- **MiniMind 使用的就是 Self-Attention**

**Cross-Attention**（交叉注意力）：
- 句子 A 关注句子 B
- 例如：翻译时，中文句子关注对应的英文句子
- 用于 Encoder-Decoder 架构（如 T5）

**为什么叫 "Self"**？
- 因为计算的是**同一个句子内部**词与词的关系
- 不是因为"token 与自身"（虽然也会计算，但不是重点）

---

### 3.3 Q、K、V 详解

**核心类比**：数据库查询

```sql
SELECT value
FROM memory_bank
WHERE key MATCHES query
```

**在 Attention 中**：

| 名称 | 全称 | 作用 | 类比 |
|------|------|------|------|
| **Q** | Query | "我想查询什么信息？" | 搜索条件 |
| **K** | Key | "我这里有什么信息？" | 索引标签 |
| **V** | Value | "我的实际内容" | 数据值 |

**例子**：

```python
句子: "我 爱 编程"

# 当理解"爱"这个词时：
Query("爱"):  "我想知道我在表达什么动作？"

Keys:
  - Key("我"):    "我是主语"
  - Key("爱"):    "我是情感动词"
  - Key("编程"):  "我是宾语"

# 匹配过程
"爱"的Query 与 "我"的Key    → 相似度 0.6 (需要知道主语)
"爱"的Query 与 "爱"的Key    → 相似度 0.3 (自己)
"爱"的Query 与 "编程"的Key  → 相似度 0.8 (需要知道宾语) ✅

# 归一化成权重
权重分布: [0.25, 0.15, 0.60]  # 最关注"编程"

# 加权求和 Value
"爱"的新表示 = 0.25×Value("我") + 0.15×Value("爱") + 0.60×Value("编程")
```

---

### 3.4 Q、K、V 怎么得到？

**关键**：Q、K、V 都是从同一个输入 X 通过不同的权重矩阵变换得到！

```python
# 输入（词嵌入）
X = [
    [我的向量],     # 768 维
    [爱的向量],     # 768 维
    [编程的向量]    # 768 维
]
X.shape: [3, 768]  # 3个词，每个768维

# 三个可学习的权重矩阵（模型参数）
W_Q: [768, 768]  # Query 权重
W_K: [768, 768]  # Key 权重
W_V: [768, 768]  # Value 权重

# 矩阵乘法生成 Q、K、V
Q = X @ W_Q  # [3, 768]
K = X @ W_K  # [3, 768]
V = X @ W_V  # [3, 768]
```

**权重矩阵的本质**：
- **是什么**：神经网络的可学习参数
- **怎么得到**：通过训练数据反向传播学习
- **存在哪里**：保存在模型文件里（.pth, .safetensors）
- **作用**：把输入变换成三个不同的"视角"

**代码位置**（`model/model_minimind.py:159-161`）：
```python
self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # W_Q
self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # W_K
self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # W_V
```

---

### 3.5 Attention 计算流程

**完整公式**：
```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

**分步骤**：

```python
# 步骤 1: 计算相似度分数
scores = Q @ K.T  # [3, 3] 矩阵，每个词和每个词的相似度
#        我    爱   编程
# 我   [1.1, 0.6, 0.4]
# 爱   [0.6, 1.0, 0.8]
# 编程 [0.4, 0.8, 1.2]

# 步骤 2: 缩放（防止梯度消失/爆炸）
scaled_scores = scores / sqrt(head_dim)  # head_dim = 96

# 步骤 3: Softmax 归一化成概率分布
attn_weights = softmax(scaled_scores, dim=-1)
# 每一行加起来 = 1
#        我     爱    编程
# 我   [0.35, 0.32, 0.33]  # 我关注其他词的权重
# 爱   [0.25, 0.15, 0.60]  # 爱关注其他词的权重
# 编程 [0.20, 0.30, 0.50]  # 编程关注其他词的权重

# 步骤 4: 加权求和 Value
output = attn_weights @ V
# output[0] = 0.35×V("我") + 0.32×V("爱") + 0.33×V("编程")
# output[1] = 0.25×V("我") + 0.15×V("爱") + 0.60×V("编程")
# output[2] = 0.20×V("我") + 0.30×V("爱") + 0.50×V("编程")
```

**代码位置**（`model/model_minimind.py:205-218`）：
```python
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
output = scores @ xv
```

---

### 3.6 Multi-Head Attention

**为什么需要多头**？

单头 Attention 只能关注一个方面，多头可以同时关注多个方面：

```
句子: "我爱编程"

Head 1: 关注语法结构（主谓宾关系）
Head 2: 关注语义相关性（动词+宾语）
Head 3: 关注情感倾向（正向/负向）
...
Head 8: 关注长距离依赖
```

**多面性的体现**：就像用多副不同的"眼镜"看同一句话！

---

### 3.7 Multi-Head 的实现

**核心思想**：拆分 → 并行计算 → 合并

```python
# MiniMind2 配置
hidden_size = 768
num_heads = 8
head_dim = hidden_size / num_heads = 96

# 流程
输入 X: [batch, seq_len, 768]
  ↓
生成 Q、K、V: [batch, seq_len, 768]
  ↓
拆分成 8 个头: [batch, 8, seq_len, 96]
  ↓
每个头独立计算 Attention
  ↓
输出: [batch, 8, seq_len, 96]
  ↓
合并（拼接）: [batch, seq_len, 768]
```

**关键代码**（`model/model_minimind.py:177-220`）：

```python
# 1. 生成 Q、K、V
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

# 2. 拆分成多头
xq = xq.view(bsz, seq_len, num_heads, head_dim)
xk = xk.view(bsz, seq_len, num_heads, head_dim)
xv = xv.view(bsz, seq_len, num_heads, head_dim)

# 3. 调整维度顺序（方便并行计算）
xq = xq.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]

# 4. 应用 RoPE（只对 Q、K）
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

# 5. 计算 Attention（8个头并行）
scores = xq @ xk.transpose(-2, -1) / sqrt(head_dim)
attn_weights = softmax(scores, dim=-1)
output = attn_weights @ xv  # [batch, 8, seq_len, 96]

# 6. 合并多头
output = output.transpose(1, 2)  # [batch, seq_len, 8, 96]
output = output.reshape(bsz, seq_len, 768)  # 拼接成 [batch, seq_len, 768]
```

**维度计算**：
- 每个头的维度：`head_dim = hidden_size / num_heads = 768 / 8 = 96`
- 合并后的维度：`8 × 96 = 768`（恢复原始维度）

**关键不变量**：
```
输入维度 = 输出维度 = 768
heads × head_dim = 768（永远成立）
```

---

### 3.8 RoPE 在 Attention 中的应用

**位置**：在生成 Q、K 之后，计算 Attention 之前

```python
流程：
1. X → Q, K, V（通过权重矩阵）
2. 拆分成多头
3. ⭐ 对 Q、K 施加 RoPE（加入位置信息）
4. 计算 Attention
5. 合并多头
```

**为什么只对 Q、K 使用 RoPE，不对 V？**

```
位置信息的作用：
  Q @ K^T  → 计算相似度
            → 需要知道"距离"信息
            → RoPE 让相近的词相似度更高

  attn @ V → 加权求和内容
            → 不需要位置信息
            → V 保持原始语义
```

**类比**：
- Q 和 K 是"地图上的位置" → 需要坐标（RoPE）
- V 是"实际的宝藏" → 不需要坐标，就是内容本身

**代码位置**（`model/model_minimind.py:182`）：
```python
# 应用 RoPE
cos, sin = position_embeddings
xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
```

---

### 3.9 Attention 完整流程总结

```python
输入: x = [我, 爱, 编程]  shape: [1, 3, 768]

┌────────────────────────────────────────────┐
│ 1. 生成 Q、K、V                             │
│    Q = x @ W_Q  [1, 3, 768]                │
│    K = x @ W_K  [1, 3, 768]                │
│    V = x @ W_V  [1, 3, 768]                │
└────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────┐
│ 2. 拆分成 8 个头                            │
│    Q: [1, 8, 3, 96]                        │
│    K: [1, 8, 3, 96]                        │
│    V: [1, 8, 3, 96]                        │
└────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────┐
│ 3. 对 Q、K 施加 RoPE（位置编码）⭐          │
│    Q_rot = RoPE(Q, cos, sin)               │
│    K_rot = RoPE(K, cos, sin)               │
└────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────┐
│ 4. 计算 Attention                           │
│    scores = Q_rot @ K_rot^T / √96          │
│    scores = softmax(scores)                │
│    output = scores @ V                     │
│    output: [1, 8, 3, 96]                   │
└────────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────────┐
│ 5. 合并 8 个头                              │
│    output.transpose(1,2).reshape(1,3,768)  │
│    output: [1, 3, 768]                     │
└────────────────────────────────────────────┘

输出: [1, 3, 768]  # 每个词融合了上下文信息！
```

---

### 3.10 Softmax 归一化详解

**Softmax 和 RMSNorm 的区别**：

| 特性 | Softmax（Attention 中） | RMSNorm（Transformer Block 中） |
|------|------------------------|-------------------------------|
| **用在哪里** | Attention 计算**内部** | Attention **之前**和 FeedForward **之前** |
| **归一化什么** | Attention 权重矩阵（每一行） | 词向量（每个向量的大小） |
| **目的** | 把分数变成概率分布 | 稳定数值，防止梯度爆炸/消失 |
| **输入** | 相似度分数（任意大小） | 词向量（768 维） |
| **输出** | 概率（0-1之间，和为1） | 归一化后的向量（保持方向） |
| **公式** | `exp(x_i) / Σexp(x_j)` | `x / sqrt(mean(x²))` |

**Softmax 在 Attention 中的作用**：

```python
# 步骤 1: 计算相似度分数
scores = Q @ K.T / sqrt(96)
scores = [
  [5.2, 3.1, 2.8],  # "我" 和其他词的分数
  [3.0, 4.5, 6.2],  # "爱" 和其他词的分数
  [2.1, 5.8, 7.3],  # "编程" 和其他词的分数
]

# 步骤 2: Softmax 归一化
weights = softmax(scores, dim=-1)
weights = [
  [0.59, 0.24, 0.17],  # 和 = 1.0 ✅
  [0.12, 0.24, 0.64],  # 和 = 1.0 ✅
  [0.05, 0.27, 0.68],  # 和 = 1.0 ✅
]

# 步骤 3: 加权求和
output = weights @ V
```

**为什么要用 Softmax？**

1. **转换成概率分布**：
   - 所有权重 ≥ 0
   - 所有权重加起来 = 1
   - 可以解释为"关注度"

2. **放大差异**：
   - 使用 exp 函数，大的分数变得更大
   - 小的分数变得更小
   - 让模型更关注相关的词

**Softmax 公式**：
```
softmax(x_i) = exp(x_i) / Σexp(x_j)
```

**位置关系**：
```python
输入 X
  ↓
RMSNorm #1 ← 归一化词向量
  ↓
Attention:
  Q, K, V = X @ W_Q, W_K, W_V
  scores = Q @ K^T
  weights = Softmax(scores) ← Softmax 在这里！
  output = weights @ V
  ↓
残差连接
  ↓
RMSNorm #2 ← 归一化词向量
  ↓
FeedForward
```

**总结**：
- **Softmax**：在 Attention **内部**，把分数变成概率
- **RMSNorm**：在 Attention **外部**，稳定词向量大小
- **完全不同的归一化，用途和位置都不一样！**

---

## 4. 前馈网络（FeedForward）

### 4.1 FeedForward 是什么？

**核心思想**：对每个词的向量进行非线性变换，"深度思考"

**典型结构**：扩张 → 激活 → 压缩
```
768 维 → 2048 维 → 768 维
```

**关键特点**：
- ✅ 每个词**独立处理**（没有词与词的交互）
- ✅ 输入维度 = 输出维度（768 维）
- ✅ 但内容完全改变（经过非线性变换）
- ✅ 分工明确：Attention 负责信息交换，FeedForward 负责深度处理

---

### 4.2 为什么要"扩张-压缩"？

**问题**：为什么不直接 768 → 768？

**答案**：直接 768 → 768 只是线性变换，表达能力有限

**对比**：

| 方案 | 计算 | 表达能力 |
|------|------|---------|
| 直接变换 | `768 → 768` | 只是线性组合，简单 |
| 扩张-压缩 | `768 → 2048 → 768` | 经过高维空间，能表达复杂函数 ✅ |

**数学本质**：
- 在高维空间中，向量有更多"自由度"
- 可以进行更复杂的非线性变换
- 压缩回来时，已经包含了更丰富的信息

**类比**：
1. **做菜**：
   - 输入：食材（原始维度）
   - 扩张：切碎、加工（高维空间）
   - 压缩：装盘（回到原始维度）
   - 结果：维度没变，但"熟"了

2. **照片处理**：
   - 输入：像素（原始维度）
   - 扩张：提取特征（高维空间）
   - 压缩：优化像素（回到原始维度）
   - 结果：分辨率没变，但"质量提升"了

---

### 4.3 普通 FFN vs SwiGLU

**普通 FeedForward（早期 Transformer）**：
```python
class SimpleFeedForward(nn.Module):
    def forward(self, x):
        h = self.w1(x)      # 扩张：768 → 2048
        h = F.relu(h)       # 激活
        output = self.w2(h) # 压缩：2048 → 768
        return output
```

**SwiGLU（MiniMind 使用）**：
```python
class SwiGLU(nn.Module):
    def forward(self, x):
        gate = self.gate_proj(x)  # 门控分支：768 → 2048
        up = self.up_proj(x)      # 上投影分支：768 → 2048

        # SiLU 激活 + 门控机制（逐元素相乘）
        hidden = F.silu(gate) * up

        output = self.down_proj(hidden)  # 压缩：2048 → 768
        return output
```

**区别**：

| 特性 | 普通 FFN | SwiGLU |
|------|----------|--------|
| **分支数** | 1 个 | 2 个（gate + up）|
| **激活函数** | ReLU | SiLU (Swish) |
| **门控机制** | 无 | 有（gate × up）|
| **性能** | 较好 | **更好**（实验证明）|
| **参数量** | 较少 | 约 1.5 倍 |
| **使用模型** | GPT-2, BERT | Llama, MiniMind, PaLM |

**SwiGLU 的优势**：
- **门控机制**：`gate` 分支控制 `up` 分支哪些信息通过
- **SiLU 激活**：比 ReLU 更平滑，梯度更好
- **表达能力**：两个分支提供更丰富的特征

---

### 4.4 SwiGLU 实现细节

**代码位置**（`model/model_minimind.py:225-238`）：
```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # SwiGLU: gate 和 up 两个分支
        gate = self.gate_proj(x)      # [batch, seq, hidden_dim]
        up = self.up_proj(x)          # [batch, seq, hidden_dim]

        # SiLU 激活 + 门控
        hidden = F.silu(gate) * up    # 逐元素相乘

        # 压缩回原维度
        return self.down_proj(hidden)
```

**维度变化**（以 MiniMind2 为例）：
```
输入 x:     [batch, seq_len, 768]
  ↓
gate:       [batch, seq_len, 2048]  # 扩张
up:         [batch, seq_len, 2048]  # 扩张
  ↓
hidden:     [batch, seq_len, 2048]  # gate × up
  ↓
输出:       [batch, seq_len, 768]   # 压缩
```

**SiLU 激活函数**：
```
SiLU(x) = x * sigmoid(x)
```

特点：
- 平滑（处处可导）
- 非单调（有负值）
- 梯度更稳定

---

### 4.5 FeedForward 的作用演示

```python
# 输入：3个词的向量
x = [
    [我的向量],     # [1.0, 2.0, 1.0, 0.5]
    [爱的向量],     # [0.5, 1.5, 2.0, 1.0]
    [编程的向量],   # [2.0, 0.5, 1.0, 1.5]
]

# 经过 FeedForward
output = FeedForward(x)

# 关键观察：
# 1. 每个词独立处理（没有词与词的交互）
# 2. 输入输出维度相同（4维 → 4维）
# 3. 但内容完全不同（经过了非线性变换）
```

**类比理解**：
```
Attention:  开会讨论（词与词交换信息）
FeedForward: 各自思考（每个词独立深化理解）

完整流程：
1. Attention: "我" 听取 "爱" 和 "编程" 的信息
2. FeedForward: "我" 基于收集到的信息深度思考
3. 输出：融合了上下文并深化理解的 "我"
```

---

### 4.6 Attention vs FeedForward 对比

| 特性 | Attention | FeedForward |
|------|-----------|-------------|
| **处理方式** | 词与词交互 | 每个词独立 |
| **作用** | 信息交换（开会） | 深度思考（各自消化）|
| **输入维度** | [seq_len, 768] | [seq_len, 768] |
| **中间维度** | [seq_len, seq_len]（分数矩阵）| [seq_len, 2048]（扩张）|
| **输出维度** | [seq_len, 768] | [seq_len, 768] |
| **位置编码** | 需要（RoPE）| 不需要 |
| **核心操作** | Q @ K^T, softmax, @ V | Linear, SiLU, Linear |
| **类比** | 词汇表查询 | 函数变换 |

**为什么需要两者配合**？
- ✅ **Attention**：让模型知道"哪些词相关"
- ✅ **FeedForward**：让模型知道"如何处理这些信息"
- ✅ **分工明确**：交互 + 处理，缺一不可

---

### 4.7 在 Transformer Block 中的位置

```python
Transformer Block:
  输入 x: [batch, seq_len, 768]
    ↓
  RMSNorm #1 ← 归一化
    ↓
  Attention ← 词与词交互
    ↓
  残差连接：x = x + Attention(x)
    ↓
  RMSNorm #2 ← 归一化
    ↓
  FeedForward ← 独立深度处理 ⭐
    ↓
  残差连接：x = x + FeedForward(x)
    ↓
  输出 x: [batch, seq_len, 768]
```

**数据流**：
```
词向量 → 归一化 → 多头注意力 → 残差 → 归一化 → 前馈网络 → 残差 → 下一层
         ↑                                      ↑
      Attention 阶段                      FeedForward 阶段
```

---

## 5. Transformer 架构

### 5.1 整体结构

MiniMind 是 **Decoder-Only Transformer**，类似 GPT。

```
MiniMindForCausalLM
├─ lm_head: 输出层 (hidden_size → vocab_size)
└─ MiniMindModel
    ├─ embed_tokens: 词嵌入层 (vocab_size → hidden_size)
    ├─ layers: N 个 MiniMindBlock (默认 8 层)
    │   └─ MiniMindBlock
    │       ├─ input_layernorm: RMSNorm
    │       ├─ self_attn: Attention
    │       ├─ 残差连接
    │       ├─ post_attention_layernorm: RMSNorm
    │       ├─ mlp: FeedForward (SwiGLU)
    │       └─ 残差连接
    └─ norm: 最终的 RMSNorm
```

### 5.2 关键配置参数

```python
MiniMindConfig(
    hidden_size=512,           # 隐藏层维度
    num_hidden_layers=8,       # Transformer Block 层数
    num_attention_heads=8,     # 注意力头数
    num_key_value_heads=2,     # GQA: KV 头数
    vocab_size=6400,           # 词汇表大小
    rope_theta=1000000.0,      # RoPE 基础频率
    max_position_embeddings=32768,  # 最大序列长度
    rms_norm_eps=1e-5,         # RMSNorm 的 epsilon
)
```

---

### 5.3 三层架构详解

MiniMind 采用**三层架构设计**：

```
第 1 层：MiniMindForCausalLM（接口层）
  ├─ 接收用户输入
  ├─ 调用核心模型
  └─ 返回预测结果
    ↓
第 2 层：MiniMindModel（核心层）
  ├─ 词嵌入
  ├─ Transformer Blocks（8层）
  └─ 最终归一化
    ↓
第 3 层：lm_head（输出层）
  └─ 投影到词汇表
```

---

### 5.4 第 1 层：MiniMindForCausalLM

**代码位置**：`model/model_minimind.py:439-471`

**作用**：对外接口，负责输入输出

```python
class MiniMindForCausalLM:
    def __init__(self, config):
        self.model = MiniMindModel(config)      # 核心模型
        self.lm_head = nn.Linear(768, 6400)      # 输出层
        # 权重共享（重要！）
        self.model.embed_tokens.weight = self.lm_head.weight
```

**数据流**：
```
输入：token IDs [245, 1023, 562]
  ↓
MiniMindModel: [batch, seq_len, 768]
  ↓
lm_head: [batch, seq_len, 6400]
  ↓
输出：logits（每个词的分数）
```

---

### 5.5 第 2 层：MiniMindModel

**代码位置**：`model/model_minimind.py:383-436`

MiniMindModel 包含 **4 个核心组件**：

#### **组件 1：词嵌入层（embed_tokens）**

**作用**：token ID → 向量

```python
self.embed_tokens = nn.Embedding(6400, 768)
# 权重矩阵：[6400, 768]
# 6400 行，每行是一个词的 768 维向量
```

**为什么需要？**
- Token ID 只是整数（245, 1023, 562）
- 整数没有语义信息（245 和 246 在数值上相近，但词义可能完全不同）
- 映射到高维向量后，才能表达"相似度"、"语义关系"

**例子**：
```python
token_id = 245  # "我"
vector = embed_tokens.weight[245]  # [0.12, 0.56, ..., 0.89]
# 相似的词有相似的向量
# "我" 和 "你" 的向量可能很接近
# "我" 和 "苹果" 的向量距离较远
```

---

#### **组件 2：8 个 Transformer Block**

**代码位置**：`model/model_minimind.py:390`

```python
self.layers = nn.ModuleList([
    MiniMindBlock(0, config),
    MiniMindBlock(1, config),
    ...
    MiniMindBlock(7, config)
])
```

**数据流**：
```
输入（嵌入后）: [batch, seq_len, 768]
  ↓
Block 0: Attention + FFN → [batch, seq_len, 768]
  ↓
Block 1: Attention + FFN → [batch, seq_len, 768]
  ↓
...
  ↓
Block 7: Attention + FFN → [batch, seq_len, 768]
  ↓
输出: [batch, seq_len, 768]
```

**每层的作用**（逐步细化）：
- 第 0-1 层：基础词关系
- 第 2-3 层：语法和句法
- 第 4-5 层：语义理解
- 第 6-7 层：高层推理

**类比**：像修改文档 8 次
- 第 1 遍：改错别字
- 第 2 遍：改语法
- 第 3 遍：改表达
- 第 8 遍：最终润色

---

#### **组件 3：最终 RMSNorm**

**代码位置**：`model/model_minimind.py:391, 428`

```python
self.norm = RMSNorm(768)
# 在 forward 中
hidden_states = self.norm(hidden_states)
```

**为什么需要？**
- 8 个 Block 累积后，数值可能偏离正常范围
- 最终 RMSNorm 确保输出到 `lm_head` 的数值稳定
- 类比：做完 8 道工序，最后质检一次

---

#### **组件 4：RoPE 预计算**

**代码位置**：`model/model_minimind.py:393-397`

```python
freqs_cos, freqs_sin = precompute_freqs_cis(
    dim=96,              # head_dim
    end=32768,           # max_position_embeddings
    rope_base=1000000.0
)
self.register_buffer("freqs_cos", freqs_cos)
self.register_buffer("freqs_sin", freqs_sin)
```

**为什么预计算？**
- RoPE 频率是**固定的**（对所有句子都一样）
- 计算一次，到处复用 → 效率大幅提升
- 每个 Block 都会用到，预计算避免重复计算

---

### 5.6 第 3 层：输出层（lm_head）

**代码位置**：`model/model_minimind.py:446, 465`

**作用**：隐藏状态 → 词汇表分数

```python
self.lm_head = nn.Linear(768, 6400)
# 权重矩阵：[6400, 768]
```

**计算过程**：
```python
hidden_state = [0.12, 0.56, ..., 0.89]  # 768维
  ↓
logits = lm_head(hidden_state)
  ↓
logits = [
    -2.3,   # token 0 的分数
    5.8,    # token 1 的分数
    ...
    8.4,    # token 562 的分数 ← 最高分
]  # 6400 个分数
```

**下一步**：
- Softmax → 概率分布
- 采样或 argmax → 预测下一个词

---

### 5.7 完整数据流

从输入到输出的完整流程：

```python
用户输入: "我爱编程"
  ↓
分词器: [245, 1023, 562]
  ↓
─────────────────────────────────────────
│ MiniMindForCausalLM                   │
│   ↓                                   │
│ ┌───────────────────────────────────┐ │
│ │ MiniMindModel                     │ │
│ │   ↓                               │ │
│ │ embed_tokens                      │ │
│ │   [245,1023,562] → [3, 768]       │ │
│ │   ↓                               │ │
│ │ Block 0:                          │ │
│ │   RMSNorm → Attention → 残差     │ │
│ │   RMSNorm → FFN → 残差           │ │
│ │   ↓                               │ │
│ │ Block 1-7: （相同结构）           │ │
│ │   ↓                               │ │
│ │ 最终 RMSNorm                      │ │
│ │   ↓                               │ │
│ │ Output: [3, 768]                  │ │
│ └───────────────────────────────────┘ │
│   ↓                                   │
│ lm_head                               │
│   [3, 768] → [3, 6400]                │
│   ↓                                   │
─────────────────────────────────────────
  ↓
Logits: [
  [...],  # "我" 之后的预测
  [...],  # "爱" 之后的预测
  [...]   # "编程" 之后的预测
]
  ↓
取最后位置: logits[-1]
  ↓
Softmax + 采样
  ↓
预测: "吗"
  ↓
输出: "我爱编程吗"
```

---

### 5.8 权重共享（Weight Tying）⭐

**代码位置**：`model/model_minimind.py:447`

```python
self.model.embed_tokens.weight = self.lm_head.weight
```

**什么是权重共享？**

让 `embed_tokens` 和 `lm_head` 使用**同一个权重矩阵**。

**矩阵对比**：

```python
# 不共享（传统做法）
embed_tokens.weight: [6400, 768]  # 矩阵 A
lm_head.weight:      [6400, 768]  # 矩阵 B（独立）
# 总参数：6400×768×2 = 9,830,400

# 共享（MiniMind 做法）
shared_weight:       [6400, 768]  # 只有一个矩阵
embed_tokens.weight = shared_weight
lm_head.weight      = shared_weight  # 指向同一个对象
# 总参数：6400×768 = 4,915,200
# 省了：4,915,200 个参数（一半！）
```

**为什么合理？**

1. **语义相关**：
   - 输入嵌入：token ID → 向量（编码）
   - 输出投影：向量 → token ID（解码）
   - 编码和解码应该用同一套"词典"

2. **节省内存**：
   - 原本需要 9.8M 参数
   - 现在只需 4.9M 参数
   - **省一半内存**！

3. **训练效果更好**：
   - 两个层共享知识
   - 输入输出保持一致性

**类比**：
- 翻译时用同一本双语字典
- 不需要两本独立的字典

**数学解释**：

```python
# 输出预测 token 245 ("我") 的分数
hidden = [...]  # 某个 768 维向量

# 不共享
score = lm_head.weight[245] @ hidden

# 共享（用嵌入向量）
score = embed_tokens.weight[245] @ hidden

# 含义：hidden 和 "我" 的嵌入向量的相似度
# 相似度高 → score 高 → 预测 "我"
```

---

### 5.9 Pre-Norm vs Post-Norm

MiniMind 使用 **Pre-Norm**（归一化在子层之前）：

**Pre-Norm（MiniMind）**：
```python
# Attention 部分
residual = x
x = self.self_attn(self.input_layernorm(x))  # 先归一化
x = x + residual  # 残差连接

# FeedForward 部分
residual = x
x = self.mlp(self.post_attention_layernorm(x))  # 先归一化
x = x + residual  # 残差连接
```

**Post-Norm（早期 Transformer）**：
```python
# Attention 部分
x = self.layernorm(x + self.self_attn(x))  # 先残差，后归一化

# FeedForward 部分
x = self.layernorm(x + self.mlp(x))  # 先残差，后归一化
```

**Pre-Norm 的优势**：
- ✅ 对**深层网络**（>12层）更稳定
- ✅ 训练更容易（梯度流动更好）
- ✅ 所有现代 LLM 都用这个（GPT-3、Llama、Claude）
- ✅ 残差路径更"干净"（不被 Norm 打断）

---

## 问答记录

### 关于归一化

**Q1: 如果网络有 100 层，标准差会变成多少？**

A: 会变成 0.0000001 甚至更小，几乎接近 0！这就是梯度消失问题。

---

**Q2: 如果所有数值都接近 0，梯度会怎样？**

A: 梯度也会接近 0，权重几乎不更新，模型学不到东西！

---

**Q3: RMSNorm 是干什么的？**

A: 是 Transformer 中的一个组件，用来保持多层网络中数值规模稳定，防止梯度消失/爆炸，同时不丢失向量的信息。

---

**Q4: 不同归一化方法会有很大区别吗？**

A: 会有区别！RMSNorm 比 LayerNorm 快 7.7 倍，参数更少，但效果相当。这就是为什么现代 LLM（Llama、MiniMind）都用 RMSNorm。

---

**Q5: 为什么训练会让标准差变化？**

A: 不一定总是变小，也可能变大，取决于：
- 权重初始化（太大会爆炸）
- 激活函数（ReLU 会让负数变 0，导致标准差变小）
- 网络结构（残差连接可以缓解这个问题）

在实验中标准差变小，是因为使用了 ReLU 且没有残差连接。

---

### 关于 RoPE

**Q6: Attention 为什么是"排列不变"的？**

A: 因为 Attention 的计算是 `Q @ K^T`，只看词向量本身，不看词的位置。如果两个句子包含的词相同（只是顺序不同），矩阵乘法的结果也相同（行列顺序被打乱）。

例如：
- "我喜欢你" 包含 {我, 喜欢, 你}
- "你喜欢我" 包含 {我, 喜欢, 你}
- 没有位置信息的话，Attention 无法区分！

---

**Q7: 为什么相对距离相同，注意力分数就相同？**

A: 这是 RoPE 的数学性质。根据三角函数的加法定理：

```
rotate(A, pos_i) · rotate(B, pos_j)
= A · B · cos((pos_j - pos_i) × θ)
= A · B · cos(相对距离 × θ)
```

只要相对距离相同，`cos(相对距离 × θ)` 就相同，所以注意力分数也相同！

---

**Q8: 旋转720度不是回到原点了吗？**

A: 很好的问题！RoPE 用了巧妙的设计避免这个问题：**多频率机制**

- 不是只用一个频率，而是用 32 个不同频率
- 高频率（频率0）：每 6.3 个token转一圈 → 编码局部位置
- 低频率（频率31）：每 6,283,185 个token转一圈 → 编码全局位置
- 多个频率组合 → 每个位置有唯一的"指纹"

就像钟表：单独看时针，12点和24点无法区分；但结合时针+分针+秒针，就能精确表示任意时刻！

---

**Q9: RoPE 是不是丢失了绝对位置信息？**

A: 没有！RoPE 同时包含绝对和相对位置信息：

1. **绝对位置**：每个词的向量被旋转到特定角度
   - 位置5的向量 ≠ 位置10的向量
   - 模型知道"这个词在位置5"

2. **相对位置**：Attention 分数只依赖相对距离
   - 位置5看位置8 = 位置0看位置3（相对距离都是3）
   - 模型知道"这两个词相距3个位置"

两全其美！

---

**Q10: 为什么 RoPE 只用顺时针旋转，不用逆时针？**

A: 因为目标是"区分不同位置"，只要每个位置的角度不同就行了。

- 顺时针：位置0→0°, 位置1→+30°, 位置2→+60°
- 逆时针：位置0→0°, 位置1→-30°, 位置2→-60°

两种方式都能唯一标识位置，所以选一个方向就够了！RoPE 实际使用的是顺时针（正角度）。

---

**Q11: 为什么需要 32 个频率？只用一个超低频率覆盖所有位置不行吗？** ⭐️

A: 这是个非常深刻的问题！理论上可以，但**实践中不行**，原因是**浮点数精度限制**。

**理论分析**：
- 如果使用频率 `θ = 2π/1000000`（每 100 万个 token 转一圈）
- 理论上可以覆盖 100 万个位置
- 每个位置的角度都不同

**实际问题 - 浮点数精度**：

1. **float32 的精度限制**：
   - float32 有效数字约 6-7 位
   - 精度下限约 `10^-7`

2. **超低频率的相邻位置差**：
   ```
   位置0的角度: 0.000000000000...
   位置1的角度: 0.000006283185...
   角度差: 6.28e-6

   cos(位置0) = 1.000000000000000
   cos(位置1) = 0.999999999980261
   差值 = 1.97e-11  ← 比 float32 精度还小！

   在 float32 下:
   cos(位置0) = 1.0
   cos(位置1) = 1.0  ← 完全相同！
   ```

3. **结果**：在计算机中，位置 0 和位置 1 无法区分！

**多频率的解决方案**：

使用 32 个不同频率（MiniMind，head_dim=64）：
- **高频率**（频率0）：每 6.3 个 token 转一圈
  - 相邻位置角度差：57.3°
  - 远远超过浮点数精度，可以清晰区分！

- **低频率**（频率31）：每 6,283,185 个 token 转一圈
  - 覆盖超长距离

- **组合效果**：
  ```
  位置0: (0°,  0°,  0°,  ..., 0°)   ← 32个频率
  位置1: (57°, 37°, 24°, ..., 0°)  ← 前面几个频率差别明显
  位置10: (213°, 12°, 242°, ..., 0°)
  ```

**类比**：
- **望远镜**（低频率）：能看远处，但看不清细节
- **显微镜**（高频率）：能看清细节，但视野有限
- **多焦距组合**（多频率）：既看清细节，又覆盖远距离

**数学本质**：
```
信噪比问题：
- 超低频率的相邻位置差 ≈ θ² (微弧度级别)
- 这个差值 < float32 精度下限
- 导致 Attention 分数几乎相同
- 模型失去位置区分能力
```

**总结**：
- ✅ 理论上：一个超低频率能覆盖所有位置（数学正确）
- ❌ 实践中：浮点数精度不够，无法区分相邻位置（工程限制）
- ✅ 解决方案：多频率组合是**数学理论 + 计算机硬件约束**的完美平衡

参考代码：`learning_materials/rope_why_multi_frequency.py`

---

### 关于 Transformer 架构

**Q12: Transformer 是什么结构？**

A: Transformer 是一种神经网络架构，由多个 Transformer Block 堆叠而成。每个 Block 包含：
- 归一化层（RMSNorm）
- 注意力机制（Attention）
- 前馈网络（FeedForward）
- 残差连接（Residual Connection）

MiniMind 是 **Decoder-Only Transformer**，类似 GPT，只有解码器部分。

---

**Q13: RMSNorm 是 Transformer 的一层吗？**

A: 不是单独一层，而是 Transformer Block **内部的组件**。每个 Block 里有 2 个 RMSNorm：
- 第1个：在 Attention 之前
- 第2个：在 FeedForward 之前

MiniMind 有 8 个 Block，所以总共有 16 个 RMSNorm。

---

**Q14: RoPE 在哪里应用？**

A: 在 Attention 计算之前，RoPE 被应用到 Query 和 Key 向量上：

```python
# Attention.forward() 中
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

# 应用 RoPE（只旋转 Q 和 K）
cos, sin = position_embeddings
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

# 然后计算 Attention
scores = xq @ xk.transpose(-2, -1)
```

---

**Q15: MiniMindModel 和 MiniMindForCausalLM 有什么区别？**

A: 两者分工明确：

- **MiniMindForCausalLM**（接口层）：
  - 对外接口，用户直接加载使用
  - 包含 `MiniMindModel`（核心）+ `lm_head`（输出层）
  - 负责：接收输入 → 调用核心 → 投影到词汇表 → 返回预测

- **MiniMindModel**（核心层）：
  - Transformer 核心实现
  - 包含：词嵌入 + 8个Block + 最终RMSNorm
  - 负责：token ID → 嵌入 → 逐层处理 → 输出隐藏状态

**类比**：
- ForCausalLM = 餐厅前台（接待）
- MiniMindModel = 后厨（核心工作）

---

**Q16: 为什么词嵌入不能直接用 token ID？**

A: 因为 token ID 只是整数，没有语义信息：

**问题**：
- Token ID 245 ("我") 和 246 ("你") 在数值上相近
- 但语义可能完全不同
- 整数之间只有大小关系，没有"相似度"

**解决**：
- 映射到 768 维向量空间
- 相似的词有相似的向量
- 可以计算距离、角度等语义关系

**例子**：
```
"我" → [0.12, 0.56, ..., 0.89]
"你" → [0.15, 0.53, ..., 0.92]  # 向量接近
"苹果" → [0.87, 0.23, ..., 0.14]  # 向量远离
```

向量空间中才有"东西"让模型学习！

---

**Q17: 权重共享（Weight Tying）是什么？为什么能省内存？** ⭐

A: 让 `embed_tokens` 和 `lm_head` 使用**同一个权重矩阵**。

**不共享**：
```python
embed_tokens.weight: [6400, 768]  # 4.9M 参数
lm_head.weight:      [6400, 768]  # 4.9M 参数
总计：9.8M 参数
```

**共享**：
```python
shared_weight: [6400, 768]  # 只有一个矩阵
embed_tokens.weight = shared_weight
lm_head.weight      = shared_weight  # 指向同一个对象
总计：4.9M 参数（省一半！）
```

**为什么合理？**
- 输入嵌入：token ID → 向量（编码）
- 输出投影：向量 → token ID（解码）
- 编码和解码用同一本"字典"很合理！

**数学意义**：
```python
# 预测 "我" 的分数
score = embed_weight[245] @ hidden
# 就是计算 hidden 和 "我" 的嵌入向量的相似度
# 相似度高 → 预测 "我"
```

---

**Q18: 为什么需要最终的 RMSNorm？**

A: 8 个 Transformer Block 累积后，数值可能偏离正常范围。

**作用**：
- 稳定最终输出
- 确保投影到 `lm_head` 的数值在合理范围
- 类比：做完 8 道工序，最后质检一次

**位置**：
```python
Block 0 → Block 1 → ... → Block 7
  ↓
最终 RMSNorm  ← 在这里！
  ↓
lm_head
```

---

**Q19: Pre-Norm 和 Post-Norm 有什么区别？**

A: 归一化的**位置不同**：

**Pre-Norm**（MiniMind 使用）：
```python
x = x + Attention(RMSNorm(x))  # 先归一化，再Attention
x = x + FFN(RMSNorm(x))        # 先归一化，再FFN
```

**Post-Norm**（早期 Transformer）：
```python
x = RMSNorm(x + Attention(x))  # 先Attention，再归一化
x = RMSNorm(x + FFN(x))        # 先FFN，再归一化
```

**Pre-Norm 的优势**：
- ✅ 对深层网络（>12层）更稳定
- ✅ 训练更容易（梯度流动更好）
- ✅ 所有现代 LLM 都用（GPT-3、Llama、Claude）

---

## 📊 重要公式汇总

### RMSNorm
```
x_norm = x / sqrt(mean(x²) + eps) * weight
```

### RoPE 频率计算
```
freqs[i] = 1 / (rope_base ^ (2i / dim))
```

### RoPE 旋转
```
q_rotated = q * cos(pos × freqs) + rotate_half(q) * sin(pos × freqs)
k_rotated = k * cos(pos × freqs) + rotate_half(k) * sin(pos × freqs)
```

### Attention 分数
```
scores = (Q @ K^T) / sqrt(head_dim)
```

---

**最后更新**：2026-01-18
