"""
🧮 RMSNorm 详解
==================

RMSNorm (Root Mean Square Normalization) 是 MiniMind 使用的归一化技术，
比传统 LayerNorm 更简单、更高效。

对比：
- LayerNorm:  x_norm = (x - mean(x)) / sqrt(var(x) + eps)
- RMSNorm:    x_norm = x / sqrt(mean(x²) + eps)

关键区别：RMSNorm 不减去均值，只做缩放！
"""

import torch
import torch.nn as nn


# ============================================================
# MiniMind 中的实现（来自 model_minimind.py:95-105）
# ============================================================
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 防止除零的小常数
        # 可学习的缩放参数，初始化为全 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 核心公式：x / sqrt(mean(x²) + eps)
        # x.pow(2): 计算 x²
        # .mean(-1, keepdim=True): 沿最后一维求均值
        # torch.rsqrt: 计算 1/sqrt(x)，比 1/torch.sqrt(x) 更快
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 1. 转换为 float32 进行计算（提高数值稳定性）
        # 2. 应用归一化
        # 3. 乘以可学习的 weight 参数
        # 4. 转换回输入的原始数据类型
        return self.weight * self._norm(x.float()).type_as(x)


# ============================================================
# 📚 为什么 RMSNorm 更好？
# ============================================================
"""
1. **计算更快**：不需要计算均值，省略减法和一次遍历
2. **内存更少**：不需要存储均值统计量
3. **效果相当**：论文证明在 LLM 中效果与 LayerNorm 相当甚至更好

数学直觉：
- 对于大型语言模型，输入分布通常接近零均值
- 此时减均值的操作不那么重要
- 只做方差缩放就足够稳定训练
"""


# ============================================================
# 🧪 示例：观察 RMSNorm 的效果
# ============================================================
if __name__ == "__main__":
    # 创建一个 RMSNorm 层，处理维度为 512 的向量
    rms_norm = RMSNorm(dim=512)

    # 创建一个随机输入 (batch_size=2, seq_len=10, hidden_size=512)
    x = torch.randn(2, 10, 512)

    print("=" * 60)
    print("🔬 RMSNorm 效果演示")
    print("=" * 60)

    # 原始输入的统计量
    print(f"\n原始输入:")
    print(f"  均值: {x.mean().item():.4f}")
    print(f"  标准差: {x.std().item():.4f}")
    print(f"  最小值: {x.min().item():.4f}")
    print(f"  最大值: {x.max().item():.4f}")

    # 应用 RMSNorm
    x_normalized = rms_norm(x)

    print(f"\n归一化后:")
    print(f"  均值: {x_normalized.mean().item():.4f}")
    print(f"  标准差: {x_normalized.std().item():.4f}")
    print(f"  最小值: {x_normalized.min().item():.4f}")
    print(f"  最大值: {x_normalized.max().item():.4f}")

    # 计算每个向量的 RMS（root mean square）
    rms_before = torch.sqrt((x ** 2).mean(dim=-1)).mean()
    rms_after = torch.sqrt((x_normalized ** 2).mean(dim=-1)).mean()

    print(f"\nRMS 值 (衡量向量的平均大小):")
    print(f"  归一化前: {rms_before.item():.4f}")
    print(f"  归一化后: {rms_after.item():.4f}")
    print(f"  → RMSNorm 将向量缩放到接近标准大小！")

    print("\n" + "=" * 60)
    print("✅ RMSNorm 主要作用：控制激活值的规模，避免梯度爆炸/消失")
    print("=" * 60)
