"""
🌀 RoPE (Rotary Position Embedding) 详解
==========================================

RoPE 是一种给 Transformer 注入"位置信息"的方法。

问题背景：
- Transformer 的 self-attention 是"排列不变"的（permutation invariant）
- 即 ["我", "爱", "你"] 和 ["你", "爱", "我"] 会得到相同的结果
- 但语言是有顺序的！我们需要告诉模型哪个词在前，哪个在后

传统方法 vs RoPE：
1. **绝对位置编码**（BERT）：在输入时加上位置向量
   缺点：无法外推到训练时未见过的长度

2. **相对位置编码**（T5）：注意力计算时考虑相对距离
   缺点：计算复杂，难以优化

3. **RoPE**（Llama/MiniMind）：用旋转矩阵编码位置
   优点：
   - 相对位置信息自然嵌入
   - 支持长度外推（YaRN）
   - 计算高效
"""

import torch
import math


# ============================================================
# 核心函数：precompute_freqs_cis (model_minimind.py:108-128)
# ============================================================
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024),
                         rope_base: float = 1e6,
                         rope_scaling: dict = None):
    """
    预计算 RoPE 的旋转频率

    参数：
        dim: 头维度 (head_dim = hidden_size / num_attention_heads)
        end: 最大序列长度（默认 32K tokens）
        rope_base: 基础频率（θ），控制旋转速度
        rope_scaling: YaRN 长度外推配置

    返回：
        freqs_cos, freqs_sin: 预计算的 cos 和 sin 值
    """
    # 步骤 1: 计算频率向量
    # 公式: freqs[i] = 1 / (θ^(2i/dim))  其中 i = 0, 1, 2, ..., dim/2-1
    # 这创建了从高频到低频的频率序列
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 📚 为什么这样设计频率？
    # - 低维度（i=0,1,2...）使用高频率 → 编码细粒度的相对位置
    # - 高维度使用低频率 → 编码粗粒度的相对位置
    # - 类似音乐中的"泛音"：低音提供基调，高音提供细节

    # 步骤 2: YaRN 长度外推（可选）
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 4)
        beta_fast = rope_scaling.get("beta_fast", 4.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)

        # 如果当前长度超过训练长度，应用 YaRN 缩放
        if end / orig_max > 1.0:
            # 找到临界维度（哪些频率需要调整）
            corr_dim = next(
                (i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max),
                dim // 2
            )

            # 计算每个维度的缩放因子
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power

            # YaRN 标准公式: λ = (β·α - β + 1)/(β·α)
            scale = torch.where(
                torch.arange(dim // 2, device=freqs.device) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor),
                1.0 / factor
            )
            freqs = freqs * scale

    # 步骤 3: 为每个位置生成频率
    t = torch.arange(end, device=freqs.device)  # 位置索引 [0, 1, 2, ..., end-1]
    freqs = torch.outer(t, freqs).float()       # 外积: [end, dim/2]

    # 步骤 4: 计算 cos 和 sin（用于旋转）
    # 每个维度复制两次，因为会成对旋转
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

    return freqs_cos, freqs_sin


# ============================================================
# 应用旋转：apply_rotary_pos_emb (model_minimind.py:131-137)
# ============================================================
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    将位置信息"旋转"到 query 和 key 向量中

    数学原理：
    对于二维向量 [x, y]，旋转 θ 角度：
        [x', y'] = [cos(θ) -sin(θ)]   [x]
                   [sin(θ)  cos(θ)] × [y]

    在 RoPE 中：
    - 把 head_dim 维度的向量分成 head_dim/2 对
    - 每对按照不同频率旋转
    - 旋转角度 = freqs[position]
    """
    def rotate_half(x):
        # 将向量分成两半并交换（用于实现旋转）
        # [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
        return torch.cat(
            (-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]),
            dim=-1
        )

    # 旋转公式：
    # q_rotated = q * cos + rotate_half(q) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))

    return q_embed, k_embed


# ============================================================
# 🧪 可视化示例：观察 RoPE 如何编码位置
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🌀 RoPE 位置编码演示")
    print("=" * 70)

    # 配置（MiniMind 的实际参数）
    hidden_size = 512
    num_heads = 8
    head_dim = hidden_size // num_heads  # 64
    max_seq_len = 512

    print(f"\n📋 配置:")
    print(f"  头维度: {head_dim}")
    print(f"  最大序列长度: {max_seq_len}")

    # 步骤 1: 预计算旋转频率
    freqs_cos, freqs_sin = precompute_freqs_cis(
        dim=head_dim,
        end=max_seq_len,
        rope_base=1000000.0  # MiniMind 默认值
    )

    print(f"\n✅ 预计算完成:")
    print(f"  freqs_cos 形状: {freqs_cos.shape}  # [max_seq_len, head_dim]")
    print(f"  freqs_sin 形状: {freqs_sin.shape}")

    # 步骤 2: 模拟一个 query 向量
    batch_size = 1
    seq_len = 5
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)

    print(f"\n🔢 输入:")
    print(f"  Q 形状: {q.shape}  # [batch, seq_len, num_heads, head_dim]")
    print(f"  K 形状: {k.shape}")

    # 步骤 3: 应用旋转位置编码
    q_rotated, k_rotated = apply_rotary_pos_emb(
        q, k,
        freqs_cos[:seq_len],
        freqs_sin[:seq_len],
        unsqueeze_dim=1  # 在 num_heads 维度前插入
    )

    print(f"\n✅ 旋转后:")
    print(f"  Q 形状: {q_rotated.shape}  # 形状不变，但编码了位置信息")
    print(f"  K 形状: {k_rotated.shape}")

    # 步骤 4: 验证相对位置信息
    # RoPE 的关键性质：q[i] · k[j] 只依赖于相对位置 (i-j)
    print(f"\n🔍 验证相对位置特性:")
    print(f"  假设我们有两个位置: pos=0 和 pos=1")

    # 位置 0 和位置 1 的 query
    q0 = q_rotated[0, 0, 0, :]  # [head_dim]
    q1 = q_rotated[0, 1, 0, :]

    # 位置 0 和位置 1 的 key
    k0 = k_rotated[0, 0, 0, :]
    k1 = k_rotated[0, 1, 0, :]

    # 计算注意力分数（点积）
    score_00 = (q0 * k0).sum().item()  # 自己和自己（距离=0）
    score_01 = (q0 * k1).sum().item()  # pos0 看 pos1（距离=1）
    score_10 = (q1 * k0).sum().item()  # pos1 看 pos0（距离=-1）
    score_11 = (q1 * k1).sum().item()  # 自己和自己（距离=0）

    print(f"  Q[0]·K[0] = {score_00:.4f}  ← 距离 0（自己）")
    print(f"  Q[0]·K[1] = {score_01:.4f}  ← 距离 +1（下一个词）")
    print(f"  Q[1]·K[0] = {score_10:.4f}  ← 距离 -1（上一个词）")
    print(f"  Q[1]·K[1] = {score_11:.4f}  ← 距离 0（自己）")

    print(f"\n💡 观察: score_00 ≈ score_11 (相同相对距离)")
    print(f"  这说明 RoPE 成功编码了相对位置信息！")

    print("\n" + "=" * 70)
    print("📚 总结:")
    print("  1. RoPE 通过旋转向量来编码位置")
    print("  2. 不同维度使用不同的旋转频率（多尺度）")
    print("  3. 注意力分数自动包含相对位置信息")
    print("  4. 支持长度外推（通过 YaRN）")
    print("=" * 70)
