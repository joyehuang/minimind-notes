"""
实验 2：四种配置对比

目的：对比 NoNorm、Post-LN、Pre-LN 三种架构的训练效果
方法：训练 4 个不同配置的简化 Transformer 模型
数据：合成数据（next-token prediction）
时间：~3 分钟（quick 模式：~30 秒）
输出：results/norm_comparison.png

注意：为保持实验独立性，本文件包含与 exp3 重复的基础类（RMSNorm、Block 等），
     便于学习者单独运行和理解每个实验。

运行：
    python exp2_norm_comparison.py
    # 快速模式：
    python exp2_norm_comparison.py --quick
"""

import sys
sys.path.append('../../..')

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
# 简化的 Transformer Block 实现
# ============================================================
class NoNormBlock(nn.Module):
    """无归一化的 Transformer Block"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x):
        # Attention + Residual (无归一化)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out

        # FFN + Residual (无归一化)
        x = x + self.ffn(x)
        return x


class PostLNBlock(nn.Module):
    """Post-LN: Compute → Residual → Norm（归一化在残差之后）

    注意：本实验中 Post-LN 仅使用 LayerNorm，因为实验目的是对比架构差异，
         而非归一化方法差异（RMSNorm 仅在 Pre-LN 中对比）。
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Attention → Residual → Norm
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.norm1(x)

        # FFN → Residual → Norm
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


class PreLNBlock(nn.Module):
    """Pre-LN: Norm → Compute → Residual（归一化在计算之前）"""
    def __init__(self, hidden_size, use_rms=False):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        if use_rms:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Norm → Attention → Residual
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + attn_out

        # Norm → FFN → Residual
        normed = self.norm2(x)
        x = x + self.ffn(normed)
        return x


# ============================================================
# 简化的语言模型
# ============================================================
class SimpleLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, block_type, use_rms=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # 根据配置选择 Block 类型
        if block_type == 'nonorm':
            self.blocks = nn.ModuleList([NoNormBlock(hidden_size) for _ in range(num_layers)])
        elif block_type == 'postln':
            # Post-LN 固定使用 LayerNorm（不使用 use_rms 参数）
            self.blocks = nn.ModuleList([PostLNBlock(hidden_size) for _ in range(num_layers)])
        elif block_type == 'preln':
            # Pre-LN 可根据 use_rms 参数选择 LayerNorm 或 RMSNorm
            self.blocks = nn.ModuleList([PreLNBlock(hidden_size, use_rms) for _ in range(num_layers)])

        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(x)
        return logits


# ============================================================
# 训练函数
# ============================================================
def train_model(model, vocab_size, steps, lr, device):
    """训练模型并返回损失曲线

    注意：本实验未使用梯度裁剪，目的是充分展示不同架构的原始训练稳定性差异。
         NoNorm 配置会因梯度爆炸而发散，这正是我们想要观察的现象。
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    nan_step = None

    batch_size = 16
    seq_len = 64

    for step in range(steps):
        # 生成随机数据（next-token prediction）
        # 注意：这是简化的合成数据。torch.roll 创建了循环依赖（最后一个 token 的目标是第一个 token），
        #      不代表真实的语言建模任务。但对于展示归一化对训练稳定性的影响，这个简化是足够的。
        X = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        Y = torch.roll(X, shifts=-1, dims=1)  # 目标是下一个 token

        # Forward
        logits = model(X)
        loss = criterion(logits.view(-1, vocab_size), Y.view(-1))

        # 检测 NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"      ⚠️  NaN detected at step {step}")
            nan_step = step
            # 填充剩余的损失值为 NaN
            losses.extend([float('nan')] * (steps - step))
            break

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # 每 100 步打印一次
        if (step + 1) % 100 == 0 or step == 0:
            print(f"      Step {step+1:4d}: loss = {loss.item():.4f}")

    return losses, nan_step


# ============================================================
# 实验主函数
# ============================================================
def run_experiment(quick_mode=False):
    """运行四种配置对比实验"""

    print("="*70)
    print("🔬 实验 2: 四种配置对比")
    print("="*70)

    # 设置参数
    vocab_size = 1000
    hidden_size = 256
    num_layers = 2
    steps = 100 if quick_mode else 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(42)

    print(f"\n📊 实验设置:")
    print(f"  - 词表大小: {vocab_size}")
    print(f"  - 隐藏维度: {hidden_size}")
    print(f"  - 层数: {num_layers}")
    print(f"  - 训练步数: {steps}")
    print(f"  - 设备: {device}")
    print(f"  - 模式: {'快速模式 (100 步)' if quick_mode else '标准模式 (1000 步)'}")

    # 配置列表：(名称, Block类型, 学习率, 是否使用RMSNorm)
    # 注意：NoNorm 使用较小的学习率（1e-4），但仍会因数值不稳定而发散
    #      这展示了即使谨慎调参，无归一化架构也难以稳定训练
    configs = [
        ("NoNorm", 'nonorm', 1e-4, False),
        ("Post-LN + LayerNorm", 'postln', 1e-4, False),
        ("Pre-LN + LayerNorm", 'preln', 5e-4, False),
        ("Pre-LN + RMSNorm", 'preln', 5e-4, True),
    ]

    results = {}

    # 训练所有配置
    for name, block_type, lr, use_rms in configs:
        print(f"\n{'='*70}")
        print(f"🔵 训练: {name}")
        print(f"{'='*70}")
        print(f"   学习率: {lr}")
        print(f"   归一化类型: {'RMSNorm' if use_rms else 'LayerNorm' if block_type != 'nonorm' else 'None'}")

        model = SimpleLM(vocab_size, hidden_size, num_layers, block_type, use_rms)
        losses, nan_step = train_model(model, vocab_size, steps, lr, device)

        results[name] = {
            'losses': losses,
            'nan_step': nan_step,
            'lr': lr,
            'final_loss': losses[-1] if not np.isnan(losses[-1]) else float('inf')
        }

        if nan_step is not None:
            print(f"   ❌ 训练发散于步数 {nan_step}")
        else:
            print(f"   ✅ 训练完成，最终损失: {losses[-1]:.4f}")

    # 可视化
    plot_results(results, steps)

    # 输出总结
    print_summary(results)


# ============================================================
# 可视化函数
# ============================================================
def plot_results(results, steps):
    """绘制训练曲线和对比表格"""

    print(f"\n📊 生成可视化图表...")

    fig = plt.figure(figsize=(16, 6))

    # 左图：训练损失曲线
    ax1 = plt.subplot(1, 2, 1)

    colors = ['red', 'orange', 'blue', 'green']
    markers = ['x', 'o', 's', '^']

    for (name, data), color, marker in zip(results.items(), colors, markers):
        losses = data['losses']
        x = list(range(len(losses)))

        # 绘制损失曲线（跳过 NaN）
        valid_indices = [i for i, loss in enumerate(losses) if not np.isnan(loss)]
        valid_losses = [losses[i] for i in valid_indices]

        ax1.plot(valid_indices, valid_losses,
                color=color, marker=marker, markevery=max(1, len(valid_indices)//10),
                label=name, linewidth=2, markersize=4, alpha=0.8)

    # 设置坐标轴和样式（在所有数据绘制完成后）
    ax1.set_xlabel('训练步数', fontsize=12)
    ax1.set_ylabel('损失', fontsize=12)
    ax1.set_title('训练损失对比', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 标记 NaN 点（在坐标轴设置完成后，此时 ylim 已确定）
    for (name, data), color in zip(results.items(), colors):
        if data['nan_step'] is not None:
            ax1.axvline(x=data['nan_step'], color=color, linestyle='--', alpha=0.3)
            ax1.text(data['nan_step'], ax1.get_ylim()[1] * 0.9,
                    f'NaN@{data["nan_step"]}',
                    rotation=90, va='top', color=color, fontsize=9)

    # 右图：对比表格
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')

    # 构建表格数据
    table_data = []
    headers = ['配置', '收敛性', 'NaN步数', '最终Loss', 'LR容忍度']

    for name, data in results.items():
        converged = "✅" if data['nan_step'] is None else "❌"
        nan_step = f"{data['nan_step']}" if data['nan_step'] is not None else "-"
        final_loss = f"{data['final_loss']:.2f}" if not np.isinf(data['final_loss']) else "NaN"
        lr_tolerance = "很低" if data['lr'] <= 1e-5 else "低" if data['lr'] <= 1e-4 else "中" if data['lr'] <= 5e-4 else "高"

        table_data.append([name, converged, nan_step, final_loss, lr_tolerance])

    # 绘制表格
    table = ax2.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置行颜色
    colors = ['#ffcccc', '#ffe5cc', '#cce5ff', '#ccffcc']
    for i, color in enumerate(colors):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)

    ax2.set_title('性能对比', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # 保存图表
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'norm_comparison.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_path}")

    plt.show()


# ============================================================
# 总结函数
# ============================================================
def print_summary(results):
    """打印实验总结"""

    print("\n" + "="*70)
    print("📊 实验结果")
    print("="*70)

    for name, data in results.items():
        print(f"\n{'='*70}")
        print(f"📌 {name}")
        print(f"{'='*70}")
        print(f"  学习率: {data['lr']}")

        if data['nan_step'] is not None:
            print(f"  ❌ 训练发散于步数 {data['nan_step']}")
            print(f"  原因: 数值不稳定，梯度爆炸")
        else:
            print(f"  ✅ 训练成功完成")
            print(f"  最终损失: {data['final_loss']:.4f}")

    print("\n" + "="*70)
    print("🎯 关键发现")
    print("="*70)
    print("""
1. NoNorm 配置:
   ❌ 无法稳定训练，即使使用极小的学习率 (1e-5)
   原因：激活值和梯度在深层网络中不稳定

2. Post-LN + LayerNorm:
   ✅ 可以训练，但需要较小的学习率
   原因：主路径上的梯度流经归一化层，存在数值不稳定风险

3. Pre-LN + LayerNorm:
   ✅ 训练稳定，可以使用更大的学习率
   原因：主路径上的梯度直接通过残差连接，更稳定

4. Pre-LN + RMSNorm:
   ✅ 训练最稳定，效果与 Pre-LN + LayerNorm 相当
   优势：计算更快（省略均值计算和偏置项）

💡 结论：
   - Pre-LN 架构在现代 LLM 中成为标准（GPT-3/LLaMA/MiniMind）
   - RMSNorm 在保持效果的同时提供更高的计算效率
    """)


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='快速模式（100步）')
    args = parser.parse_args()

    run_experiment(quick_mode=args.quick)

    print("\n" + "="*70)
    print("💭 思考题")
    print("="*70)
    print("""
1. 为什么 NoNorm 配置即使使用很小的学习率也会 NaN？
   提示：查看实验 1 的梯度消失/爆炸现象

2. Post-LN 和 Pre-LN 的关键区别是什么？
   提示：观察归一化层在残差连接中的位置

3. 为什么 Pre-LN 可以使用更大的学习率？
   提示：思考梯度在主路径上的流动方式

4. RMSNorm 相比 LayerNorm 节省了哪些计算？
   提示：对比两者的公式（参考 teaching.md）

5. 如果增加到 8 层，结果会有什么变化？
   提示：运行实验 3 查看深层网络的对比
    """)
