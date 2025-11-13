"""
实验 2：LayerNorm vs RMSNorm 对比

目的：对比两种归一化方法的效果和性能
方法：测试归一化效果、计算速度
数据：合成数据
时间：< 30 秒
输出：results/layernorm_vs_rmsnorm.png

运行：
    python exp2_layernorm_vs_rmsnorm.py
"""

import sys
sys.path.append('../../..')

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 两种归一化实现
# ============================================================
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


# ============================================================
# 实验主函数
# ============================================================
def run_experiment():
    """运行 LayerNorm vs RMSNorm 对比实验"""

    print("="*70)
    print("🔬 实验 2: LayerNorm vs RMSNorm 对比")
    print("="*70)

    hidden_dim = 512
    batch_size = 1000
    num_iterations = 10000

    # 创建输入（均值不为0，标准差不为1）
    torch.manual_seed(42)
    x = torch.randn(batch_size, hidden_dim) * 5 + 2  # 均值≈2, 标准差≈5

    print(f"\n📊 实验设置:")
    print(f"  - 隐藏维度: {hidden_dim}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 速度测试迭代次数: {num_iterations}")

    print(f"\n原始输入统计:")
    print(f"  - 均值: {x.mean().item():.4f}")
    print(f"  - 标准差: {x.std().item():.4f}")
    print(f"  - 数值范围: [{x.min().item():.2f}, {x.max().item():.2f}]")

    # ============================================================
    # 测试 1: 归一化效果对比
    # ============================================================
    print(f"\n🔍 测试 1: 归一化效果对比")
    print("-" * 70)

    ln = LayerNorm(hidden_dim)
    rms = RMSNorm(hidden_dim)

    x_ln = ln(x)
    x_rms = rms(x)

    print(f"\n经过 LayerNorm:")
    print(f"  - 均值: {x_ln.mean().item():.6f}  ← 强制为 0")
    print(f"  - 标准差: {x_ln.std().item():.4f}  ← 接近 1")
    print(f"  - 参数量: {sum(p.numel() for p in ln.parameters())} (weight + bias)")

    print(f"\n经过 RMSNorm:")
    print(f"  - 均值: {x_rms.mean().item():.4f}  ← 不强制为 0")
    print(f"  - 标准差: {x_rms.std().item():.4f}  ← 接近 1")
    print(f"  - 参数量: {sum(p.numel() for p in rms.parameters())} (只有 weight)")

    # ============================================================
    # 测试 2: 计算速度对比
    # ============================================================
    print(f"\n⏱️  测试 2: 计算速度对比")
    print("-" * 70)

    # LayerNorm 速度
    start = time.time()
    for _ in range(num_iterations):
        _ = ln(x)
    ln_time = time.time() - start

    # RMSNorm 速度
    start = time.time()
    for _ in range(num_iterations):
        _ = rms(x)
    rms_time = time.time() - start

    speedup = (ln_time / rms_time - 1) * 100

    print(f"\nLayerNorm:  {ln_time:.4f} 秒")
    print(f"RMSNorm:    {rms_time:.4f} 秒")
    print(f"速度提升:   {speedup:.1f}% 更快！")

    # ============================================================
    # 可视化
    # ============================================================
    print(f"\n📊 生成可视化图表...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 图1: 归一化前后的分布（直方图）
    axes[0, 0].hist(x.flatten().numpy(), bins=50, alpha=0.7, label='Original', color='gray')
    axes[0, 0].axvline(x.mean().item(), color='gray', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Original Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 图2: LayerNorm 后的分布
    axes[0, 1].hist(x_ln.detach().flatten().numpy(), bins=50, alpha=0.7, label='LayerNorm', color='blue')
    axes[0, 1].axvline(x_ln.mean().item(), color='blue', linestyle='--', linewidth=2, label=f'Mean={x_ln.mean().item():.3f}')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('After LayerNorm')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 图3: RMSNorm 后的分布
    axes[1, 0].hist(x_rms.detach().flatten().numpy(), bins=50, alpha=0.7, label='RMSNorm', color='green')
    axes[1, 0].axvline(x_rms.mean().item(), color='green', linestyle='--', linewidth=2, label=f'Mean={x_rms.mean().item():.3f}')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('After RMSNorm')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 图4: 性能对比柱状图
    metrics = ['Time (s)', 'Parameters']
    ln_metrics = [ln_time, sum(p.numel() for p in ln.parameters())]
    rms_metrics = [rms_time, sum(p.numel() for p in rms.parameters())]

    # 归一化到 [0, 1] 以便对比
    ln_norm = [ln_metrics[0]/max(ln_metrics[0], rms_metrics[0]),
               ln_metrics[1]/max(ln_metrics[1], rms_metrics[1])]
    rms_norm = [rms_metrics[0]/max(ln_metrics[0], rms_metrics[0]),
                rms_metrics[1]/max(ln_metrics[1], rms_metrics[1])]

    x_pos = range(len(metrics))
    width = 0.35

    bars1 = axes[1, 1].bar([i - width/2 for i in x_pos], ln_norm, width,
                          label='LayerNorm', color='blue', alpha=0.7)
    bars2 = axes[1, 1].bar([i + width/2 for i in x_pos], rms_norm, width,
                          label='RMSNorm', color='green', alpha=0.7)

    axes[1, 1].set_ylabel('Relative Value (Normalized)')
    axes[1, 1].set_title('Performance Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 标注实际数值
    for bars, actual_vals in zip([bars1, bars2], [ln_metrics, rms_metrics]):
        for bar, val, metric in zip(bars, actual_vals, metrics):
            height = bar.get_height()
            if 'Time' in metric:
                label = f'{val:.3f}s'
            else:
                label = f'{int(val)}'
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          label, ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # 保存图表
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'layernorm_vs_rmsnorm.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_path}")

    # ============================================================
    # 结果总结
    # ============================================================
    print("\n" + "="*70)
    print("📊 实验结果总结")
    print("="*70)

    print(f"\n| 特性 | LayerNorm | RMSNorm |")
    print(f"|------|-----------|---------|")
    print(f"| 均值 | 强制为 0 | {x_rms.mean().item():.4f} |")
    print(f"| 标准差 | {x_ln.std().item():.4f} | {x_rms.std().item():.4f} |")
    print(f"| 计算时间 | {ln_time:.4f}s | {rms_time:.4f}s |")
    print(f"| 参数量 | {sum(p.numel() for p in ln.parameters())} | {sum(p.numel() for p in rms.parameters())} |")
    print(f"| 速度优势 | - | +{speedup:.1f}% |")

    print("\n" + "="*70)
    print("🎯 关键发现")
    print("="*70)
    print("""
1. 归一化效果：
   - LayerNorm：强制均值=0，标准差=1
   - RMSNorm：只控制标准差≈1，均值可以保留

2. 为什么 RMSNorm 更快：
   - 少一步计算（不需要减均值）
   - 少一半参数（不需要 bias）
   - GPU 优化更好（rsqrt 是融合操作）

3. 为什么省略减均值也可以：
   - 在深层网络中，激活分布通常已经接近零均值
   - 只控制尺度（标准差）就足够稳定训练
   - 实验证明：两者在 LLM 上效果相当

4. 现代 LLM 的选择：
   - Llama、GPT-3+、MiniMind 都使用 RMSNorm
   - 速度更快 + 参数更少 + 效果相当 = 更好的选择
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
1. 为什么 LayerNorm 需要 bias 而 RMSNorm 不需要？
   提示：LayerNorm 减均值后需要偏移

2. 如果输入的均值本来就是 0，LayerNorm 和 RMSNorm 还有区别吗？
   提示：仍有计算步骤的区别

3. 能否设计一个实验证明"省略减均值"不影响训练效果？
   提示：训练两个小模型，对比最终 loss
    """)
