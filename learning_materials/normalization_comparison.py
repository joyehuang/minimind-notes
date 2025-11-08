"""
📊 不同归一化方法对比
====================

对比 LayerNorm 和 RMSNorm 的区别
"""

import torch
import torch.nn as nn
import time


class LayerNorm(nn.Module):
    """传统的 LayerNorm（BERT/GPT-2 使用）"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class RMSNorm(nn.Module):
    """RMSNorm（Llama/MiniMind 使用）"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def compare_normalizations():
    print("="*70)
    print("📊 LayerNorm vs RMSNorm 对比")
    print("="*70)

    # 创建相同的输入
    x = torch.randn(1000, 512) * 5 + 2  # 均值≈2, 标准差≈5

    print(f"\n原始输入:")
    print(f"  均值: {x.mean().item():.4f}")
    print(f"  标准差: {x.std().item():.4f}")

    # LayerNorm
    ln = LayerNorm(512)
    x_ln = ln(x)

    print(f"\n经过 LayerNorm:")
    print(f"  均值: {x_ln.mean().item():.4f}  ← 接近 0")
    print(f"  标准差: {x_ln.std().item():.4f}  ← 接近 1")

    # RMSNorm
    rms = RMSNorm(512)
    x_rms = rms(x)

    print(f"\n经过 RMSNorm:")
    print(f"  均值: {x_rms.mean().item():.4f}  ← 不一定是 0")
    print(f"  标准差: {x_rms.std().item():.4f}  ← 接近 1")

    print("\n" + "="*70)
    print("🔍 关键区别:")
    print("  LayerNorm: 强制均值=0, 标准差=1")
    print("  RMSNorm:   只控制标准差≈1, 均值可以不是 0")
    print("="*70)

    # 速度对比
    print("\n⏱️  速度对比（10000 次迭代）:")

    # LayerNorm 速度
    start = time.time()
    for _ in range(10000):
        _ = ln(x)
    ln_time = time.time() - start

    # RMSNorm 速度
    start = time.time()
    for _ in range(10000):
        _ = rms(x)
    rms_time = time.time() - start

    print(f"  LayerNorm: {ln_time:.4f} 秒")
    print(f"  RMSNorm:   {rms_time:.4f} 秒")
    print(f"  RMSNorm 快了 {(ln_time/rms_time - 1)*100:.1f}%!")

    print("\n" + "="*70)
    print("✅ 总结:")
    print("  - RMSNorm 更快，参数更少（没有 bias）")
    print("  - 在 LLM 训练中效果相当")
    print("  - 这就是为什么现代 LLM 都用 RMSNorm")
    print("="*70)


if __name__ == "__main__":
    compare_normalizations()
