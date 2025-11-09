"""
Attention 机制中的 Q、K、V 详解
用数据库查询的类比理解 Query、Key、Value

作者: joyehuang
日期: 2025-11-09
"""

import torch
import torch.nn.functional as F

print("=" * 60)
print("Attention 中的 Q、K、V 类比理解")
print("=" * 60)

# ============================================================
# 1. 数据库类比示例
# ============================================================
print("\n【1. 数据库类比】")
print("-" * 60)

# 图书馆藏书（Key-Value 存储）
library = {
    "编程": "《Python 入门》内容...",
    "数学": "《线性代数》内容...",
    "历史": "《世界史》内容..."
}

# 查询
query = "我想学编程"

print(f"查询 (Query): {query}")
print(f"\n图书馆藏书 (Key-Value):")
for key, value in library.items():
    print(f"  Key: {key:4s} → Value: {value}")

# 模拟匹配过程
print(f"\n匹配过程:")
print(f"  '{query}' 和 '编程' 匹配度高 ✅ → 返回《Python 入门》")
print(f"  '{query}' 和 '数学' 匹配度低    → 权重很小")
print(f"  '{query}' 和 '历史' 匹配度低    → 权重很小")

# ============================================================
# 2. Attention 中的实际例子（简化版）
# ============================================================
print("\n\n【2. Attention 实际计算】")
print("-" * 60)

# 句子: "小明 很 喜欢 猫"
words = ["小明", "很", "喜欢", "猫"]
print(f"句子: {' '.join(words)}")

# 为简化，我们用 2 维向量表示（实际是 512/768 维）
# 这里的数值是手工设计的，让你看到效果

# Query: 每个词"想要查询什么信息"
Q = torch.tensor([
    [1.0, 0.5],  # 小明的Query: 关注主语相关
    [0.3, 0.8],  # 很的Query: 关注程度相关
    [0.6, 0.9],  # 喜欢的Query: 关注动作相关
    [0.4, 0.2],  # 猫的Query: 关注对象相关
])

# Key: 每个词"提供什么信息的索引"
K = torch.tensor([
    [0.9, 0.4],  # 小明的Key: 我是主语
    [0.2, 0.7],  # 很的Key: 我是程度词
    [0.5, 0.8],  # 喜欢的Key: 我是动词
    [0.3, 0.1],  # 猫的Key: 我是宾语
])

# Value: 每个词"实际的语义内容"
V = torch.tensor([
    [1.0, 1.0],  # 小明的语义信息
    [2.0, 2.0],  # 很的语义信息
    [3.0, 3.0],  # 喜欢的语义信息
    [4.0, 4.0],  # 猫的语义信息
])

print(f"\nQ (Query) shape: {Q.shape}  # [4个词, 2维向量]")
print(f"K (Key) shape:   {K.shape}")
print(f"V (Value) shape: {V.shape}")

# ============================================================
# 3. 计算 Attention
# ============================================================
print("\n\n【3. 计算过程】")
print("-" * 60)

# 步骤 1: 计算相似度分数 (Q @ K^T)
# 每个 Query 和所有 Key 计算点积
scores = Q @ K.T  # [4, 4] 矩阵
print(f"\n步骤 1: 计算相似度分数 (Q @ K^T)")
print(f"scores shape: {scores.shape}  # [4个Query词, 4个Key词]")
print(f"\n相似度矩阵:")
print(f"{'':6s}", end="")
for w in words:
    print(f"{w:>8s}", end="")
print()
for i, word in enumerate(words):
    print(f"{word:6s}", end="")
    for j in range(len(words)):
        print(f"{scores[i,j]:8.2f}", end="")
    print()

# 解释第一行
print(f"\n解释: '小明'的Query 和所有Key的匹配度:")
for i, word in enumerate(words):
    print(f"  与'{word}'的Key匹配度: {scores[0, i]:.2f}")

# 步骤 2: 缩放 (除以 sqrt(d_k))
d_k = Q.shape[-1]  # 向量维度
scaled_scores = scores / (d_k ** 0.5)
print(f"\n步骤 2: 缩放 (除以 sqrt({d_k}) = {d_k**0.5:.2f})")
print(f"防止梯度消失/爆炸")

# 步骤 3: Softmax 归一化
attention_weights = F.softmax(scaled_scores, dim=-1)
print(f"\n步骤 3: Softmax 归一化成概率分布")
print(f"attention_weights shape: {attention_weights.shape}")
print(f"\n注意力权重矩阵 (每行和=1):")
print(f"{'':6s}", end="")
for w in words:
    print(f"{w:>8s}", end="")
print(f"  {'Sum':>6s}")
for i, word in enumerate(words):
    print(f"{word:6s}", end="")
    row_sum = 0
    for j in range(len(words)):
        val = attention_weights[i, j].item()
        print(f"{val:8.2%}", end="")
        row_sum += val
    print(f"  {row_sum:6.0%}")

# 解释一行
print(f"\n解释: '喜欢'这个词关注其他词的权重分布:")
for i, word in enumerate(words):
    weight = attention_weights[2, i].item()
    print(f"  关注'{word}': {weight:6.2%}")

# 步骤 4: 加权求和 Value
output = attention_weights @ V
print(f"\n步骤 4: 用权重加权求和 Value")
print(f"output = attention_weights @ V")
print(f"output shape: {output.shape}  # [4个词, 2维新表示]")
print(f"\n每个词的新表示:")
for i, word in enumerate(words):
    print(f"{word}: {output[i].tolist()}")

# ============================================================
# 4. 详细展示一个词的计算
# ============================================================
print("\n\n【4. 详细展示: '喜欢'这个词的计算】")
print("-" * 60)

word_idx = 2  # "喜欢"
word = words[word_idx]

print(f"\n'{word}'的 Query 想要查询什么:")
print(f"  Q[{word_idx}] = {Q[word_idx].tolist()}")

print(f"\n其他词的 Key（索引标签）:")
for i, w in enumerate(words):
    print(f"  K[{i}] ('{w}'的Key) = {K[i].tolist()}")

print(f"\n'{word}'的Query 和每个Key的匹配度:")
for i, w in enumerate(words):
    score = scores[word_idx, i].item()
    print(f"  Q[{word_idx}] · K[{i}] ('{w}') = {score:.2f}")

print(f"\nSoftmax 后的注意力权重:")
for i, w in enumerate(words):
    weight = attention_weights[word_idx, i].item()
    bar = "█" * int(weight * 50)
    print(f"  '{w}': {weight:6.2%} {bar}")

print(f"\n加权求和计算 '{word}' 的新表示:")
print(f"  新表示 = ", end="")
for i, w in enumerate(words):
    weight = attention_weights[word_idx, i].item()
    if i > 0:
        print(f"         + ", end="")
    print(f"{weight:.2%} × V[{i}] ('{w}'的Value)")

final = output[word_idx]
print(f"  结果 = {final.tolist()}")

# ============================================================
# 5. 总结
# ============================================================
print("\n\n【5. 总结：Q、K、V 的作用】")
print("=" * 60)
print("""
类比数据库查询:
  SELECT value
  FROM memory_bank
  WHERE key MATCHES query

在 Attention 中:
  ┌─────────────┬────────────────────────────────┐
  │ Query (Q)   │ "我想查询什么信息？"           │
  │             │ 每个词想要从其他词获取的信息   │
  ├─────────────┼────────────────────────────────┤
  │ Key (K)     │ "我这里有什么信息？"           │
  │             │ 每个词提供的索引标签           │
  ├─────────────┼────────────────────────────────┤
  │ Value (V)   │ "我的实际内容是什么？"         │
  │             │ 每个词的实际语义信息           │
  └─────────────┴────────────────────────────────┘

计算流程:
  1. Q @ K^T      → 计算相似度分数
  2. / sqrt(d_k)  → 缩放（稳定梯度）
  3. Softmax      → 归一化成概率分布
  4. @ V          → 加权求和得到新表示

关键点:
  • Q、K、V 都来自同一个输入（Self-Attention）
  • Q、K、V 通过不同的权重矩阵变换得到
  • 最终每个词的新表示融合了其他词的信息
""")

print("=" * 60)
print("运行完成！")
print("=" * 60)
