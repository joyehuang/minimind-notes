"""
🌀 RoPE 从零理解 - 第一部分：为什么需要位置编码
==================================================

这个程序帮你理解：
1. Attention 为什么是"排列不变"的
2. 为什么需要位置编码
3. RoPE 的核心思想
"""

import torch
import torch.nn.functional as F


# ============================================================
# 实验 1: Attention 的"排列不变性"问题
# ============================================================
def experiment_permutation_invariance():
    print("="*70)
    print("🔴 实验 1: Attention 的排列不变性问题")
    print("="*70)

    # 模拟三个词的嵌入向量
    # 假设这是 ["我", "喜欢", "你"] 的词向量
    sentence1 = torch.tensor([
        [1.0, 0.0, 0.0],  # "我"
        [0.0, 1.0, 0.0],  # "喜欢"
        [0.0, 0.0, 1.0],  # "你"
    ]).unsqueeze(0)  # [batch_size=1, seq_len=3, dim=3]

    # 打乱顺序：["你", "喜欢", "我"]
    sentence2 = torch.tensor([
        [0.0, 0.0, 1.0],  # "你"
        [0.0, 1.0, 0.0],  # "喜欢"
        [1.0, 0.0, 0.0],  # "我"
    ]).unsqueeze(0)

    print("\n句子 1: [我, 喜欢, 你]")
    print(sentence1.squeeze(0))

    print("\n句子 2: [你, 喜欢, 我]")
    print(sentence2.squeeze(0))

    # 简化的 Attention 计算（只用 Q·K^T）
    # 注意：这里为了演示，Q=K=V=输入本身
    def simple_attention_scores(x):
        # x: [1, seq_len, dim]
        # 计算注意力分数: Q @ K^T
        scores = x @ x.transpose(-2, -1)  # [1, 3, 3]
        return scores.squeeze(0)  # [3, 3]

    scores1 = simple_attention_scores(sentence1)
    scores2 = simple_attention_scores(sentence2)

    print("\n句子 1 的注意力分数矩阵:")
    print(scores1)

    print("\n句子 2 的注意力分数矩阵:")
    print(scores2)

    print("\n" + "="*70)
    print("❌ 问题：虽然词序不同，但注意力模式（矩阵的数值）")
    print("   被打乱了！模型无法区分 '我喜欢你' 和 '你喜欢我'")
    print("="*70)


# ============================================================
# 实验 2: 添加简单的位置编码
# ============================================================
def experiment_with_position():
    print("\n\n")
    print("="*70)
    print("🟢 实验 2: 添加位置信息后")
    print("="*70)

    # 同样的词向量
    sentence1 = torch.tensor([
        [1.0, 0.0, 0.0],  # "我"
        [0.0, 1.0, 0.0],  # "喜欢"
        [0.0, 0.0, 1.0],  # "你"
    ])

    sentence2 = torch.tensor([
        [0.0, 0.0, 1.0],  # "你"
        [0.0, 1.0, 0.0],  # "喜欢"
        [1.0, 0.0, 0.0],  # "我"
    ])

    # 添加简单的位置编码（位置0, 1, 2）
    # 这里用最简单的方法：把位置编号乘以一个小数加到向量上
    position_ids = torch.tensor([0.1, 0.2, 0.3]).unsqueeze(1)  # [3, 1]

    # 加上位置信息
    sentence1_with_pos = sentence1 + position_ids
    sentence2_with_pos = sentence2 + position_ids

    print("\n句子 1 + 位置编码: [我(位置0), 喜欢(位置1), 你(位置2)]")
    print(sentence1_with_pos)

    print("\n句子 2 + 位置编码: [你(位置0), 喜欢(位置1), 我(位置2)]")
    print(sentence2_with_pos)

    def simple_attention_scores(x):
        scores = x @ x.T
        return scores

    scores1 = simple_attention_scores(sentence1_with_pos)
    scores2 = simple_attention_scores(sentence2_with_pos)

    print("\n句子 1 的注意力分数矩阵:")
    print(scores1)

    print("\n句子 2 的注意力分数矩阵:")
    print(scores2)

    print("\n" + "="*70)
    print("✅ 现在矩阵不同了！模型可以区分不同的词序")
    print("="*70)


# ============================================================
# 实验 3: RoPE 的核心思想 - 旋转
# ============================================================
def experiment_rope_intuition():
    print("\n\n")
    print("="*70)
    print("🌀 实验 3: RoPE 的核心思想 - 旋转向量")
    print("="*70)

    print("\n想象一个 2D 向量在平面上旋转：")
    print("""
        y
        |
        |  ● (x, y) 原始向量
        | /
        |/_____ x

    旋转 θ 角度后：
        y
        |
        |      ● (x', y') 旋转后的向量
        |    /
        |  /
        |/_____ x
    """)

    # 创建一个简单的 2D 向量
    vector = torch.tensor([1.0, 0.0])  # 指向右边的单位向量

    print(f"\n原始向量: {vector}")
    print(f"  这个向量指向 →（东）方向")

    # 旋转矩阵公式：
    # [cos(θ)  -sin(θ)] [x]
    # [sin(θ)   cos(θ)] [y]

    def rotate_vector(v, angle_degrees):
        """旋转一个 2D 向量"""
        angle = torch.tensor(angle_degrees * 3.14159 / 180)  # 转换为弧度
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        rotation_matrix = torch.tensor([
            [cos, -sin],
            [sin,  cos]
        ])

        return rotation_matrix @ v

    # 旋转不同角度
    for angle in [0, 45, 90, 180]:
        rotated = rotate_vector(vector, angle)
        print(f"\n旋转 {angle:3d}°: {rotated}")

        if angle == 0:
            print("  → 方向（东）")
        elif angle == 45:
            print("  ↗ 方向（东北）")
        elif angle == 90:
            print("  ↑ 方向（北）")
        elif angle == 180:
            print("  ← 方向（西）")

    print("\n" + "="*70)
    print("💡 RoPE 的关键思想:")
    print("  - 位置 0 → 旋转 0°")
    print("  - 位置 1 → 旋转 θ°")
    print("  - 位置 2 → 旋转 2θ°")
    print("  - 位置 3 → 旋转 3θ°")
    print("  ...")
    print("\n  这样，相对位置 = 相对旋转角度！")
    print("="*70)


# ============================================================
# 实验 4: 相对位置的魔法
# ============================================================
def experiment_relative_position():
    print("\n\n")
    print("="*70)
    print("✨ 实验 4: RoPE 如何自动编码相对位置")
    print("="*70)

    print("\n假设我们有两个词：")
    print("  词 A 在位置 5")
    print("  词 B 在位置 8")
    print("  相对距离 = 8 - 5 = 3")

    # 创建两个简单向量
    word_a = torch.tensor([1.0, 0.0])
    word_b = torch.tensor([0.8, 0.6])  # 不同的向量

    # 假设每个位置旋转 30 度
    theta = 30

    def rotate_vector(v, angle_degrees):
        angle = torch.tensor(angle_degrees * 3.14159 / 180)
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        rotation_matrix = torch.tensor([[cos, -sin], [sin, cos]])
        return rotation_matrix @ v

    # 应用 RoPE：旋转到各自的位置
    word_a_pos5 = rotate_vector(word_a, 5 * theta)  # 位置 5
    word_b_pos8 = rotate_vector(word_b, 8 * theta)  # 位置 8

    print(f"\n词 A 在位置 5（旋转 {5*theta}°）: {word_a_pos5}")
    print(f"词 B 在位置 8（旋转 {8*theta}°）: {word_b_pos8}")

    # 计算注意力分数（点积）
    attention_score = (word_a_pos5 @ word_b_pos8).item()
    print(f"\n注意力分数（词A·词B）: {attention_score:.4f}")

    print("\n" + "-"*70)
    print("🔍 关键观察：")
    print("-"*70)

    # 现在把两个词都向前移动（比如都减去位置2）
    # 相对距离还是 3！
    print("\n如果两个词都往前移动 2 个位置：")
    print("  词 A: 位置 5 → 位置 3")
    print("  词 B: 位置 8 → 位置 6")
    print("  相对距离还是 = 6 - 3 = 3")

    word_a_pos3 = rotate_vector(word_a, 3 * theta)  # 位置 3
    word_b_pos6 = rotate_vector(word_b, 6 * theta)  # 位置 6

    attention_score2 = (word_a_pos3 @ word_b_pos6).item()
    print(f"\n新的注意力分数: {attention_score2:.4f}")
    print(f"之前的注意力分数: {attention_score:.4f}")
    print(f"\n✅ 分数几乎相同！因为相对位置（相对旋转角度）没变！")

    print("\n" + "="*70)
    print("🎯 这就是 RoPE 的魔法：")
    print("  通过旋转，自动实现了相对位置编码！")
    print("="*70)


# ============================================================
# 总结
# ============================================================
def summary():
    print("\n\n")
    print("="*70)
    print("📚 总结：RoPE 核心思想")
    print("="*70)

    print("""
1️⃣  问题：Attention 是排列不变的
    → 无法区分 "我喜欢你" 和 "你喜欢我"

2️⃣  解决：需要位置编码
    → 告诉模型每个词在哪个位置

3️⃣  RoPE 的方法：旋转向量
    → 位置 i 的向量旋转 i×θ 度
    → 相对位置 = 相对旋转角度（自动！）

4️⃣  优点：
    ✅ 自然包含相对位置信息
    ✅ 计算高效（预计算 cos/sin）
    ✅ 支持长度外推（YaRN）

5️⃣  在 MiniMind 中：
    - 每个 Attention 计算前都会应用 RoPE
    - 旋转频率预先计算好（freqs_cos, freqs_sin）
    - precompute_freqs_cis: model/model_minimind.py:62-78
    - apply_rotary_pos_emb: model/model_minimind.py:80-93
    - MiniMindBlock 中调用: model/model_minimind.py:119, 204-207
    """)

    print("="*70)


# ============================================================
# 运行所有实验
# ============================================================
if __name__ == "__main__":
    print("\n🌀 RoPE 从零理解 - 基础概念\n")

    # 实验 1: 展示问题
    experiment_permutation_invariance()

    # 实验 2: 简单的解决方案
    experiment_with_position()

    # 实验 3: RoPE 的旋转思想
    experiment_rope_intuition()

    # 实验 4: 相对位置的魔法
    experiment_relative_position()

    # 总结
    summary()

    print("\n\n💭 思考题:")
    print("="*70)
    print("""
1. 如果不用位置编码，Transformer 能工作吗？
   → 能工作，但无法理解词序，效果很差

2. 为什么 RoPE 能外推到更长的序列？
   → 因为它用的是旋转角度，只要角度算法合理（YaRN），
      可以推广到训练时没见过的长度

3. RoPE 只旋转 Q 和 K，为什么不旋转 V？
   → 因为位置信息只需要影响"匹配度"（Q·K），
      不需要影响"内容"（V）
    """)
    print("="*70)
