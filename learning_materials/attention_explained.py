"""
🎯 Multi-Head Attention 详解
==============================

Attention 是 Transformer 的灵魂，让模型能"关注"输入中的重要部分。

核心思想：当模型处理一个词时，需要参考上下文中的其他词
例如：
  "The animal didn't cross the street because it was too tired."

  当处理 "it" 时，模型应该关注 "animal"（高注意力权重）
  而不是 "street"（低注意力权重）

MiniMind 使用的是 **Grouped Query Attention (GQA)**：
- Query 头：8 个（num_attention_heads）
- Key/Value 头：2 个（num_key_value_heads）
- 每 4 个 Q 共享 1 对 K/V（节省内存和计算）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 简化版 Attention 类（基于 model_minimind.py:150-222）
# ============================================================
class SimplifiedAttention(nn.Module):
    """
    简化的注意力机制，去掉了一些细节便于理解核心逻辑
    """
    def __init__(self, hidden_size=512, num_heads=8, num_kv_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads  # 每个头的维度
        self.n_rep = num_heads // num_kv_heads    # KV 重复次数

        # 三个投影矩阵：Q, K, V
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)

        # 输出投影
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x):
        """
        x: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # 步骤 1: 投影到 Q, K, V
        # TODO(human)

        print(f"\n" + "="*60)
        print("📊 Attention 计算流程")
        print("="*60)
        print(f"输入形状: {x.shape}")
        print(f"Q 形状: {xq.shape}")
        print(f"K 形状: {xk.shape}")
        print(f"V 形状: {xv.shape}")

        # 步骤 2: 计算注意力分数
        # scores = Q @ K^T / sqrt(head_dim)
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        print(f"\n注意力分数形状: {scores.shape}")
        print(f"分数范围: [{scores.min().item():.2f}, {scores.max().item():.2f}]")

        # 步骤 3: Causal Mask（因果掩码，确保只能看到之前的词）
        # 创建上三角矩阵，对角线以上设为 -inf
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")),
            diagonal=1
        ).to(scores.device)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # 步骤 4: Softmax 归一化（得到注意力权重）
        attn_weights = F.softmax(scores, dim=-1)
        print(f"\n注意力权重形状: {attn_weights.shape}")
        print(f"权重之和（每行应该=1）: {attn_weights[0, 0, 0].sum().item():.4f}")

        # 步骤 5: 加权求和
        output = attn_weights @ xv
        print(f"\n加权后输出: {output.shape}")

        # 步骤 6: 合并多头 + 输出投影
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)
        print(f"最终输出: {output.shape}")

        return output, attn_weights


# ============================================================
# 🧪 可视化示例：观察注意力权重
# ============================================================
def visualize_attention_pattern():
    """
    可视化一个简单例子的注意力模式
    """
    print("\n" + "="*60)
    print("🔍 注意力模式可视化")
    print("="*60)

    # 创建一个小的注意力层
    attn = SimplifiedAttention(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn.eval()

    # 创建一个短序列
    batch_size = 1
    seq_len = 5
    x = torch.randn(batch_size, seq_len, 64)

    # 前向传播
    with torch.no_grad():
        output, attn_weights = attn(x)

    # 打印第一个头的注意力权重矩阵
    print("\n📋 第 0 个头的注意力权重矩阵:")
    print("   (行=查询位置, 列=键位置)")
    print()
    weights = attn_weights[0, 0].cpu().numpy()  # [seq_len, seq_len]

    # 打印表头
    print("     ", end="")
    for j in range(seq_len):
        print(f"  K{j}  ", end="")
    print()

    # 打印每一行
    for i in range(seq_len):
        print(f"  Q{i} ", end="")
        for j in range(seq_len):
            if j > i:  # 因果掩码区域
                print("  ---  ", end="")
            else:
                print(f"{weights[i, j]:6.3f}", end=" ")
        print()

    print("\n💡 观察:")
    print("  1. 对角线右上方全是 --- (因果掩码，不能看到未来)")
    print("  2. 每行权重加起来 = 1.0 (归一化)")
    print("  3. 越靠近对角线，权重通常越大（局部性）")


# ============================================================
# 📚 GQA (Grouped Query Attention) 解释
# ============================================================
def explain_gqa():
    print("\n" + "="*60)
    print("🔬 GQA vs MHA vs MQA 对比")
    print("="*60)

    print("\n1️⃣ **MHA (Multi-Head Attention)** - 传统方案")
    print("   Query 头: 8")
    print("   Key 头:   8")
    print("   Value 头: 8")
    print("   → 参数多，计算慢，但效果最好")

    print("\n2️⃣ **MQA (Multi-Query Attention)** - 极致压缩")
    print("   Query 头: 8")
    print("   Key 头:   1  ← 所有 Q 共享 1 个 K")
    print("   Value 头: 1  ← 所有 Q 共享 1 个 V")
    print("   → 参数少，推理快，但效果稍差")

    print("\n3️⃣ **GQA (Grouped Query Attention)** - MiniMind 的选择")
    print("   Query 头: 8")
    print("   Key 头:   2  ← 每 4 个 Q 共享 1 对 K/V")
    print("   Value 头: 2")
    print("   → 平衡效果和效率！")

    # 计算参数量对比
    hidden_size = 512
    head_dim = 64

    mha_params = 3 * 8 * (hidden_size * head_dim)  # Q, K, V 各 8 头
    mqa_params = 8 * (hidden_size * head_dim) + 2 * (hidden_size * head_dim)  # Q=8, K=1, V=1
    gqa_params = 8 * (hidden_size * head_dim) + 2 * 2 * (hidden_size * head_dim)  # Q=8, K=2, V=2

    print(f"\n📊 参数量对比（hidden_size={hidden_size}）:")
    print(f"   MHA: {mha_params:,} 参数  (100%)")
    print(f"   MQA: {mqa_params:,} 参数  ({mqa_params/mha_params*100:.1f}%)")
    print(f"   GQA: {gqa_params:,} 参数  ({gqa_params/mha_params*100:.1f}%)")
    print(f"   → GQA 节省了 {(1-gqa_params/mha_params)*100:.1f}% 的 KV 参数！")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("🎯 Multi-Head Attention 深度解析")
    print("="*60)

    # 1. 基础流程演示
    visualize_attention_pattern()

    # 2. GQA 解释
    explain_gqa()

    print("\n" + "="*60)
    print("✅ 总结:")
    print("  - Attention 让模型能\"聚焦\"重要信息")
    print("  - 多头机制捕获不同类型的依赖关系")
    print("  - GQA 平衡效果和效率")
    print("  - Causal mask 确保自回归生成")
    print("="*60)
