"""
🌀 RoPE 多频率机制详解
======================

回答问题：
1. 旋转720度不是回到原点了吗？
2. 为什么 RoPE 不会"循环"？
3. 绝对位置信息丢失了吗？
"""

import torch
import math


def demonstrate_multi_frequency():
    print("="*70)
    print("🌀 RoPE 的多频率机制")
    print("="*70)

    print("\n问题：如果只用一个频率，旋转720度就回到原点了！")
    print("解决：使用多个不同频率的旋转！\n")

    # MiniMind 的实际参数
    head_dim = 64  # 每个注意力头的维度
    rope_base = 1000000.0  # θ 的基础值

    # 计算频率（这是 RoPE 的核心公式）
    # freqs[i] = 1 / (rope_base ^ (2i / dim))
    freqs = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    print("📊 MiniMind 使用的频率（前10个）:")
    print("-"*70)
    for i in range(10):
        freq = freqs[i].item()
        # 计算这个频率需要多少个 token 才转360度
        tokens_per_circle = 2 * math.pi / freq
        print(f"  频率 {i:2d}: {freq:.2e}  "
              f"→ 每 {tokens_per_circle:>10.1f} 个token转一圈")

    print("\n💡 观察:")
    print("  - 高频率（频率0）：每 6.3 个token转一圈 → 编码局部位置")
    print("  - 低频率（频率31）：每 6,283,185.3 个token转一圈 → 编码全局位置")
    print("  - 组合起来：可以精确编码百万级别的位置！")

    print("\n" + "="*70)
    print("🔍 实际例子：位置编码向量")
    print("="*70)

    # 为几个位置生成 RoPE 编码
    positions = [0, 1, 10, 100, 1000]

    print("\n每个位置的旋转角度（前4个维度）:\n")
    print("位置".ljust(8), end="")
    for i in range(4):
        print(f"  维度{i}的角度".ljust(15), end="")
    print()
    print("-"*70)

    for pos in positions:
        angles = pos * freqs[:4]  # 只看前4个维度
        print(f"{pos:6d}".ljust(8), end="")
        for angle in angles:
            degree = (angle.item() * 180 / math.pi) % 360  # 转换为度数
            print(f"{degree:>12.1f}°".ljust(15), end="")
        print()

    print("\n💡 观察（理解这个表格的两个维度）:")
    print("  【纵向看 - 沿 token 维度】：")
    print("    - 维度0（高频）：随位置增加，角度变化快")
    print("      例如：从位置0到位置10，角度变化约100度")
    print("    - 维度3（低频）：随位置增加，角度变化相对较慢")
    print("      例如：频率约为维度0的1/4，转一圈需要约23个token（维度0只需6.3个）")
    print()
    print("  【横向看 - 沿嵌入维度】：")
    print("    - 从维度0到维度3：频率递减，转动速度逐渐变慢")
    print("      例如：观察表格中任意一行（同一位置），从左到右频率递减")
    print("      参考上面的频率表：维度0每6.3个token转一圈，")
    print("                        维度3约每23个token转一圈（约4倍慢）")
    print()
    print("  【结论】：不同维度组合 → 唯一标识每个位置！")


def demonstrate_absolute_vs_relative():
    print("\n\n")
    print("="*70)
    print("🤔 RoPE 丢失绝对位置了吗？")
    print("="*70)

    print("\n答案：没有！RoPE 同时包含绝对和相对位置信息！\n")

    print("📚 理解:")
    print("-"*70)
    print("1. 绝对位置信息：")
    print("   - 每个词的向量都被旋转了特定角度")
    print("   - 位置5的向量 ≠ 位置10的向量")
    print("   - 模型知道'这个词在位置5'")
    print()
    print("2. 相对位置信息（额外的！）：")
    print("   - 计算 Q·K 时，只依赖相对距离")
    print("   - 位置5看位置8 = 位置0看位置3（相对距离都是3）")
    print("   - 模型知道'这两个词相距3个位置'")
    print()
    print("3. RoPE 的优势：")
    print("   ✅ 有绝对位置信息（每个词编码了自己的位置）")
    print("   ✅ 有相对位置信息（Attention 分数只依赖相对距离）")
    print("   ✅ 两全其美！")


def demonstrate_clockwise_question():
    print("\n\n")
    print("="*70)
    print("🔄 关于顺时针/逆时针的问题")
    print("="*70)

    print("\n你的直觉很敏锐！RoPE 确实只用了一个方向的旋转。\n")

    print("❓ 为什么不需要反向旋转？")
    print("-"*70)
    print("  - RoPE 的目标：区分不同位置")
    print("  - 只要每个位置的角度不同就行")
    print("  - 顺时针和逆时针都能做到这一点")
    print("  - 所以选一个方向就够了！")

    print("\n📐 数学上：")
    print("-"*70)
    print("  如果用顺时针：")
    print("    位置0 → 0°")
    print("    位置1 → +30°  (顺时针)")
    print("    位置2 → +60°")
    print()
    print("  如果用逆时针：")
    print("    位置0 → 0°")
    print("    位置1 → -30° (逆时针)")
    print("    位置2 → -60°")
    print()
    print("  效果相同！都能区分位置。")

    print("\n✅ 实际上 RoPE 用的是顺时针（正角度）")


def visualize_position_encoding():
    print("\n\n")
    print("="*70)
    print("📊 可视化：RoPE 如何编码位置")
    print("="*70)

    # 简化示例：只用2个频率
    print("\n假设我们只用 2 个频率（实际 MiniMind 用32个）：")
    print("  频率1：快（每10个token转一圈）")
    print("  频率2：慢（每100个token转一圈）")
    print()

    positions = [0, 5, 10, 15, 50, 100]
    freq1 = 2 * math.pi / 10   # 快频率
    freq2 = 2 * math.pi / 100  # 慢频率

    print("位置\t频率1角度\t频率2角度\t组合后是否唯一？")
    print("-"*70)

    for pos in positions:
        angle1 = (pos * freq1 * 180 / math.pi) % 360
        angle2 = (pos * freq2 * 180 / math.pi) % 360
        print(f"{pos:3d}\t{angle1:7.1f}°\t{angle2:7.1f}°\t✅ 唯一")

    print("\n💡 观察:")
    print("  - 位置0和位置10：频率1角度相同（都是0°）")
    print("    但频率2角度不同（0° vs 36°）")
    print("  - 多频率组合 → 每个位置都有唯一的'指纹'！")


if __name__ == "__main__":
    print("\n🌀 RoPE 多频率机制深度解析\n")

    # 1. 多频率机制
    demonstrate_multi_frequency()

    # 2. 绝对 vs 相对位置
    demonstrate_absolute_vs_relative()

    # 3. 顺时针/逆时针
    demonstrate_clockwise_question()

    # 4. 可视化
    visualize_position_encoding()

    print("\n\n" + "="*70)
    print("📚 总结")
    print("="*70)
    print("""
1. RoPE 使用多个频率，避免"转一圈回到原点"的问题
   - 高频：编码局部位置（相邻几个词）
   - 低频：编码全局位置（整篇文档）

2. RoPE 同时包含绝对和相对位置信息
   - 绝对：每个词被旋转到特定角度
   - 相对：Attention 分数只看相对角度差

3. 只需要一个方向的旋转（顺时针）
   - 目标是区分位置，不需要双向

4. 多频率组合 → 每个位置有唯一的"编码指纹"
    """)
    print("="*70)
