"""
🎯 Attention 机制从零理解
===========================

用最简单的例子帮你理解：
1. Q、K、V 是什么
2. Attention 如何计算
3. 为什么需要 Attention
"""

import torch
import torch.nn.functional as F


# ============================================================
# 实验 1: 理解 Q、K、V 的概念
# ============================================================
def explain_qkv_concept():
    print("="*70)
    print("📚 实验 1: 理解 Query、Key、Value")
    print("="*70)

    print("""
🔍 类比：在图书馆找书

场景：你想找一本关于"机器学习"的书

1. Query (查询):
   - 你的需求："我想找机器学习的书"
   - 在 Attention 中：当前词想要什么信息

2. Key (键):
   - 每本书的标签："Python编程"、"机器学习"、"小说"...
   - 在 Attention 中：每个词提供的信息类型

3. Value (值):
   - 书的实际内容
   - 在 Attention 中：每个词的具体信息

匹配过程:
   你的查询 vs 每本书的标签 → 计算相似度
   相似度高的书 → 仔细阅读（高权重）
   相似度低的书 → 忽略（低权重）
    """)


# ============================================================
# 实验 2: 最简单的 Attention 计算
# ============================================================
def simple_attention_example():
    print("\n" + "="*70)
    print("🧮 实验 2: 最简单的 Attention 计算")
    print("="*70)

    # 假设有3个词的句子："I love cats"
    # 用简化的3维向量表示
    print("\n句子: \"I love cats\"")
    print("用简化的3维向量表示每个词:\n")

    # 词向量（为了演示，手动设置简单的值）
    words = torch.tensor([
        [1.0, 0.0, 0.0],  # "I"
        [0.0, 1.0, 0.0],  # "love"
        [0.0, 0.0, 1.0],  # "cats"
    ])

    print("词向量:")
    print("  I:    [1.0, 0.0, 0.0]")
    print("  love: [0.0, 1.0, 0.0]")
    print("  cats: [0.0, 0.0, 1.0]")

    # 为了简化，让 Q = K = V = words（实际中是不同的投影）
    Q = words  # Query: 每个词想要什么信息
    K = words  # Key: 每个词提供什么信息
    V = words  # Value: 每个词的实际内容

    print("\n" + "-"*70)
    print("步骤 1: 计算注意力分数 (Q @ K^T)")
    print("-"*70)

    # 计算注意力分数
    scores = Q @ K.T  # [3, 3]

    print("\n注意力分数矩阵:")
    print("       I    love  cats")
    for i, word in enumerate(["I", "love", "cats"]):
        print(f"  {word:4s} ", end="")
        for j in range(3):
            print(f"{scores[i, j].item():5.1f} ", end="")
        print()

    print("\n💡 理解:")
    print("  - 对角线都是 1.0: 每个词和自己最相似")
    print("  - 其他位置都是 0.0: 这些词向量互相正交（特意设计的）")

    print("\n" + "-"*70)
    print("步骤 2: Softmax 归一化（得到注意力权重）")
    print("-"*70)

    # Softmax
    attention_weights = F.softmax(scores, dim=-1)

    print("\n注意力权重矩阵:")
    print("       I     love   cats")
    for i, word in enumerate(["I", "love", "cats"]):
        print(f"  {word:4s} ", end="")
        for j in range(3):
            print(f"{attention_weights[i, j].item():.3f} ", end="")
        print(f"  (和={attention_weights[i].sum().item():.3f})")

    print("\n💡 理解:")
    print("  - 每行的权重加起来 = 1.0")
    print("  - 权重表示'关注程度'")

    print("\n" + "-"*70)
    print("步骤 3: 加权求和 (Attention × V)")
    print("-"*70)

    # 加权求和
    output = attention_weights @ V

    print("\n输出向量:")
    for i, word in enumerate(["I", "love", "cats"]):
        print(f"  {word}: {output[i].numpy()}")

    print("\n💡 理解:")
    print("  - 每个词的输出是所有词 Value 的加权平均")
    print("  - 权重由注意力分数决定")


# ============================================================
# 实验 3: 真实场景 - "it" 指代消歧
# ============================================================
def realistic_attention_example():
    print("\n\n" + "="*70)
    print("🌟 实验 3: 真实场景 - 代词消歧")
    print("="*70)

    print("""
句子: "The animal didn't cross the street because it was too tired."

问题: "it" 指代什么？
  选项 A: animal  ← 正确！
  选项 B: street

Attention 如何解决这个问题？
    """)

    # 简化示例：只关注 "animal", "street", "it" 三个词
    # 用 5 维向量表示（实际中是 512 维）

    print("简化表示（只看关键词）:")
    print()

    # 手工设计的向量（模拟真实的词嵌入）
    # "animal" 和 "it" 的向量比较相似
    animal = torch.tensor([0.8, 0.6, 0.2, 0.1, 0.9])
    street = torch.tensor([0.2, 0.1, 0.9, 0.8, 0.1])
    it = torch.tensor([0.7, 0.5, 0.3, 0.2, 0.8])  # 和 animal 更相似

    # 组合成矩阵
    words = torch.stack([animal, street, it])
    word_names = ["animal", "street", "it"]

    print("词向量（简化的5维）:")
    for name, vec in zip(word_names, words):
        print(f"  {name:6s}: {vec.numpy()}")

    # 计算注意力
    Q = words  # Query
    K = words  # Key
    V = words  # Value

    scores = Q @ K.T  # [3, 3]
    attention_weights = F.softmax(scores, dim=-1)

    print("\n" + "="*70)
    print("观察:")
    print("="*70)
    print(f"  - it 对 animal 的注意力: {attention_weights[2, 0].item():.3f}")
    print(f"  - it 对 street 的注意力: {attention_weights[2, 1].item():.3f}")
    print(f"  - it 对 自己 的注意力:   {attention_weights[2, 2].item():.3f}")

    print(f"\n✅ 'it' 最关注 '{word_names[attention_weights[2].argmax().item()]}'!")
    print("   这就是 Attention 如何帮助模型理解代词指代的！")


# ============================================================
# 总结
# ============================================================
def summary():
    print("\n\n" + "="*70)
    print("📚 总结：Attention 机制的本质")
    print("="*70)

    print("""
1️⃣  三个角色:
    Query (Q): 我想找什么信息？
    Key   (K): 我有什么信息？
    Value (V): 具体信息是什么？

2️⃣  计算流程:
    ① 相似度: Q @ K^T  (查询和每个键的相似度)
    ② 归一化: Softmax   (转换为权重)
    ③ 加权和: Attention @ V  (按权重组合信息)

3️⃣  核心思想:
    - 让模型能"查看"所有上下文
    - 自动学会关注重要的信息
    - 忽略不相关的信息

4️⃣  为什么叫 "Self-Attention"？
    - "Self" = 自己看自己
    - 句子中的词互相看（而不是看外部信息）
    - Q、K、V 都来自同一个句子

5️⃣  在 MiniMind 中:
    - 每个 Transformer Block 都有一个 Attention 层
    - 使用 8 个注意力头（Multi-Head Attention）
    - 使用 GQA（Grouped Query Attention）优化
    """)

    print("="*70)


# ============================================================
# 运行所有实验
# ============================================================
if __name__ == "__main__":
    print("\n🎯 Attention 机制从零理解\n")

    # 实验 1: 概念解释
    explain_qkv_concept()

    # 实验 2: 最简单的计算
    simple_attention_example()

    # 实验 3: 真实场景
    realistic_attention_example()

    # 总结
    summary()

    print("\n💭 思考题:")
    print("="*70)
    print("""
1. 为什么需要 Softmax？
   → 确保权重加起来=1，变成概率分布

2. 如果去掉 Attention，只用 FeedForward 行吗？
   → 不行！模型无法看到上下文，只能逐词处理

3. Q、K、V 为什么要分开？
   → 灵活性！可以学习"我想要什么"和"我有什么"的不同表示

4. Attention 分数很大会怎样？
   → Softmax 后会变成接近 [0, 0, ..., 1]（只关注一个词）
      所以要除以 sqrt(head_dim) 来缩放
    """)
    print("="*70)
