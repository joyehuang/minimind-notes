"""
实验 1：梯度消失可视化

目的：证明归一化的必要性
方法：对比有/无归一化的激活标准差变化
数据：合成数据（随机张量）
时间：< 10 秒
输出：results/gradient_vanishing.png

运行：
    python exp1_gradient_vanishing.py
"""

import sys
sys.path.append('../../..')  # 添加项目根目录到路径

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ============================================================
# RMSNorm 实现
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


# ============================================================
# 实验主函数
# ============================================================
def run_experiment():
    """运行梯度消失对比实验"""

    print("="*70)
    print("🔬 实验 1: 梯度消失可视化")
    print("="*70)

    # 设置参数
    hidden_dim = 512
    num_layers = 10
    torch.manual_seed(42)  # 可复现

    # 初始输入
    x_init = torch.randn(100, hidden_dim)  # [batch=100, hidden_dim=512]

    print(f"\n📊 实验设置:")
    print(f"  - 隐藏维度: {hidden_dim}")
    print(f"  - 层数: {num_layers}")
    print(f"  - 批次大小: {x_init.shape[0]}")
    print(f"  - 初始标准差: {x_init.std().item():.4f}")

    # ============================================================
    # 情况 1: 无归一化
    # ============================================================
    print(f"\n🔴 运行: 无归一化网络...")

    x = x_init.clone()
    stds_without_norm = [x.std().item()]

    layers_no_norm = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]

    for i, layer in enumerate(layers_no_norm):
        x = layer(x)
        x = torch.relu(x)
        stds_without_norm.append(x.std().item())

        if (i + 1) % 2 == 0:
            print(f"  Layer {i+1}: std = {x.std().item():.6f}")

    print(f"  最终标准差: {stds_without_norm[-1]:.6f}")

    # ============================================================
    # 情况 2: 使用 RMSNorm
    # ============================================================
    print(f"\n🟢 运行: 使用 RMSNorm 的网络...")

    x = x_init.clone()
    stds_with_norm = [x.std().item()]

    layers_with_norm = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
    norms = [RMSNorm(hidden_dim) for _ in range(num_layers)]

    for i, (layer, norm) in enumerate(zip(layers_with_norm, norms)):
        x = layer(x)
        x = torch.relu(x)
        x = norm(x)  # 关键：归一化
        stds_with_norm.append(x.std().item())

        if (i + 1) % 2 == 0:
            print(f"  Layer {i+1}: std = {x.std().item():.6f}")

    print(f"  最终标准差: {stds_with_norm[-1]:.6f}")

    # ============================================================
    # 可视化对比
    # ============================================================
    print(f"\n📊 生成可视化图表...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：标准差变化曲线
    layers_x = list(range(num_layers + 1))

    ax1.plot(layers_x, stds_without_norm, 'r-o', label='Without Normalization', linewidth=2)
    ax1.plot(layers_x, stds_with_norm, 'g-s', label='With RMSNorm', linewidth=2)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Target (std=1.0)')

    ax1.set_xlabel('Layer Depth', fontsize=12)
    ax1.set_ylabel('Activation Std', fontsize=12)
    ax1.set_title('Gradient Vanishing Problem', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 对数坐标，更清晰

    # 右图：最终对比柱状图
    final_stds = [stds_without_norm[-1], stds_with_norm[-1]]
    colors = ['red', 'green']
    bars = ax2.bar(['Without Norm', 'With RMSNorm'], final_stds, color=colors, alpha=0.7)

    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Target (std=1.0)')
    ax2.set_ylabel('Final Std', fontsize=12)
    ax2.set_title('Final Layer Stability', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # 标注数值
    for bar, std in zip(bars, final_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{std:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # 保存图表
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'gradient_vanishing.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_path}")

    # ============================================================
    # 结果总结
    # ============================================================
    print("\n" + "="*70)
    print("📊 实验结果")
    print("="*70)

    print(f"\n❌ 无归一化:")
    print(f"  - 初始标准差: {stds_without_norm[0]:.4f}")
    print(f"  - 最终标准差: {stds_without_norm[-1]:.6f}")
    print(f"  - 衰减比例: {(stds_without_norm[-1] / stds_without_norm[0]):.6f}")
    print(f"  - 结论: {'梯度消失！' if stds_without_norm[-1] < 0.01 else '数值不稳定'}")

    print(f"\n✅ 使用 RMSNorm:")
    print(f"  - 初始标准差: {stds_with_norm[0]:.4f}")
    print(f"  - 最终标准差: {stds_with_norm[-1]:.4f}")
    print(f"  - 衰减比例: {(stds_with_norm[-1] / stds_with_norm[0]):.4f}")
    print(f"  - 结论: 数值稳定！")

    print("\n" + "="*70)
    print("🎯 关键发现")
    print("="*70)
    print("""
1. 梯度消失问题：
   无归一化的网络，激活标准差从 1.0 衰减到 ~0.00X
   这意味着梯度几乎为 0，模型无法学习

2. RMSNorm 的作用：
   保持每一层的激活标准差在 1.0 附近
   确保梯度能够顺利传播回前面的层

3. 为什么需要归一化：
   - 稳定训练：防止梯度消失/爆炸
   - 加速收敛：可以使用更大的学习率
   - 深层网络：让 10+ 层网络变得可训练
    """)

    plt.show()


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    run_experiment()

    print("\n" + "="*70)
    print("💭 思考题")
    print("="*70)
    print("""
1. 如果层数增加到 20 层，会发生什么？
   提示：尝试修改 num_layers = 20 重新运行

2. 如果改用 LayerNorm 而不是 RMSNorm，结果会一样吗？
   提示：两者在稳定性上效果类似

3. 为什么要用对数坐标（log scale）绘图？
   提示：因为数值跨度太大（从 1.0 到 0.001）

4. 残差连接也能缓解梯度消失，它和归一化有什么区别？
   提示：两者作用不同但互补，现代架构两者都用
    """)
