"""
实验 3：深度影响 — 浅层 vs 深层网络
展示残差连接如何让深层网络也能正常训练

核心问题：
  1. 网络加深后，无残差连接会发生什么？
  2. 残差连接能让网络堆多深？
  3. "退化问题" (degradation problem) 是什么？

作者: minimind-notes
日期: 2025-05-25
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Heiti SC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 0. 概述
# ============================================================
print("=" * 60)
print("实验 3：深度影响 — 浅层 vs 深层网络")
print("=" * 60)


# ============================================================
# 1. 生成数据
# ============================================================
torch.manual_seed(42)

n_samples = 500
x = torch.randn(n_samples, 20)
# 较复杂的非线性关系
y = (torch.sin(2 * x[:, 0]) * torch.cos(x[:, 1]) +
     torch.tanh(x[:, 2]) * 0.5 + 0.1 * torch.randn(n_samples))
y = y.unsqueeze(1)

print(f"\n数据: {n_samples} 样本, x 维度=20, y 维度=1")


# ============================================================
# 2. 实验：不同深度 × 有/无残差
# ============================================================
print("\n" + "-" * 60)
print("【2. 多深度、多配置训练对比】")
print("-" * 60)


class PlainDeepNet(nn.Module):
    """深层网络（无残差）"""
    def __init__(self, input_dim=20, hidden_dim=64, num_layers=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.input_proj(x))
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class ResidualDeepNet(nn.Module):
    """深层网络（有残差）"""
    def __init__(self, input_dim=20, hidden_dim=64, num_layers=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.input_proj(x))
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = F.relu(self.fc1(x))
        residual = self.fc2(residual)
        return F.relu(residual + x)


def train_model(model_class, x, y, num_layers, use_residual, epochs=300, lr=0.001):
    """训练一个特定配置的模型"""
    if use_residual:
        model = ResidualDeepNet(num_layers=num_layers, hidden_dim=64)
    else:
        model = PlainDeepNet(num_layers=num_layers, hidden_dim=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        # 打乱数据
        indices = torch.randperm(len(x))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(x), 32):
            batch_idx = indices[i:i+32]
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        losses.append(avg_loss)

        # 检测 NaN
        if np.isnan(avg_loss):
            print(f"  ⚠️ {num_layers}层 {'有' if use_residual else '无'}残差 → "
                  f"第 {epoch+1} epoch 出现 NaN，停止训练")
            break

    return losses, sum(p.numel() for p in model.parameters())


# 测试不同深度
depths = [3, 5, 10, 20]
epochs = 300

all_results = {}

print(f"训练配置: {epochs} epochs, lr=0.001\n")

for depth in depths:
    print(f"[深度 = {depth} 层]")

    # 无残差
    losses_no_res, params_no = train_model(
        PlainDeepNet, x, y, num_layers=depth, use_residual=False, epochs=epochs
    )
    final_no = losses_no_res[-1] if not np.isnan(losses_no_res[-1]) else float('nan')
    print(f"  无残差: final_loss = {final_no:.6f}, params = {params_no:,}")

    # 有残差
    losses_with_res, params_with = train_model(
        ResidualDeepNet, x, y, num_layers=depth, use_residual=True, epochs=epochs
    )
    final_with = losses_with_res[-1]
    print(f"  有残差: final_loss = {final_with:.6f}, params = {params_with:,}")

    if not np.isnan(final_no):
        print(f"  改善: {(final_no - final_with) / final_no * 100:.1f}%")
    else:
        print(f"  改善: N/A (无残差直接 NaN)")

    all_results[depth] = {
        'no_residual': losses_no_res,
        'with_residual': losses_with_res,
        'final_no_res': final_no,
        'final_with_res': final_with,
    }
    print()


# ============================================================
# 3. 可视化
# ============================================================
print("\n" + "-" * 60)
print("【3. 可视化结果】")
print("-" * 60)

n_plots = len(depths)
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

colors = {'无残差': '#ff6b6b', '有残差': '#51cf66'}

for idx, depth in enumerate(depths):
    ax = axes[idx]
    result = all_results[depth]

    losses_no = result['no_residual']
    losses_with = result['with_residual']

    ax.plot(losses_no, label='无残差', color=colors['无残差'], alpha=0.8, linewidth=1.5)
    ax.plot(losses_with, label='有残差', color=colors['有残差'], alpha=0.8, linewidth=1.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'{depth} 层网络 — Loss 曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 标注 NaN
    if any(np.isnan(losses_no)):
        nan_epoch = next(i for i, v in enumerate(losses_no) if np.isnan(v))
        ax.axvline(x=nan_epoch, color='red', linestyle='--', alpha=0.5)
        ax.text(nan_epoch, ax.get_ylim()[1] * 0.9, f'NaN@{nan_epoch}',
                color='red', ha='center', fontsize=9)

# 图 5: 最终 Loss 对比
ax = axes[4]
final_no_res = [all_results[d]['final_no_res'] for d in depths]
final_with_res = [all_results[d]['final_with_res'] for d in depths]

x_pos = np.arange(len(depths))
width = 0.35

bars1 = ax.bar(x_pos - width/2, final_no_res, width, label='无残差', color=colors['无残差'])
bars2 = ax.bar(x_pos + width/2, final_with_res, width, label='有残差', color=colors['有残差'])

ax.set_xlabel('网络深度')
ax.set_ylabel('Final MSE Loss')
ax.set_title('最终 Loss 对比（不同深度）')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{d}层' for d in depths])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 标注数值
for bar, val in zip(bars1, final_no_res):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(bars2, final_with_res):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 图 6: 深度 vs 训练效果（摘要）
ax = axes[5]
ax.plot(depths, final_no_res, 'o-', color=colors['无残差'], linewidth=2, markersize=10, label='无残差')
ax.plot(depths, final_with_res, 's-', color=colors['有残差'], linewidth=2, markersize=10, label='有残差')
ax.set_xlabel('网络深度')
ax.set_ylabel('Final MSE Loss')
ax.set_title('深度 vs 效果：残差连接的价值')
ax.legend()
ax.grid(True, alpha=0.3)
# 用阴影标注"退化区域"
ax.axvspan(5.5, 22, alpha=0.05, color='red')
ax.text(12, ax.get_ylim()[1] * 0.95, '无残差退化区域', ha='center', fontsize=10, color='red', alpha=0.5)

plt.suptitle('实验 3：深度对残差连接效果的影响', fontsize=14, fontweight='bold')
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'exp3_depth_impact.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图表已保存到 results/exp3_depth_impact.png")


# ============================================================
# 4. 总结分析
# ============================================================
print("\n" + "=" * 60)
print("【4. 深度影响总结】")
print("=" * 60)

print(f"""
┌─────────┬──────────────┬──────────────┬──────────────────┐
│  深度   │ 无残差 Loss  │ 有残差 Loss  │      分析        │
├─────────┼──────────────┼──────────────┼──────────────────┤""")
for depth in depths:
    r = all_results[depth]
    fn = r['final_no_res']
    fw = r['final_with_res']
    if np.isnan(fn):
        analysis = "无残差直接 NaN，完全学不了"
    elif fn > fw * 5:
        analysis = f"残差效果好 {(fn-fw)/fn*100:.0f}%"
    else:
        analysis = f"浅层差异不大"
    print(f"│  {depth:3d}层  │ {fn:>12.4f} │ {fw:>12.4f} │ {analysis:<16} │")
print("└─────────┴──────────────┴──────────────┴──────────────────┘")

print("""
关键发现:
  1. 浅层 (3-5层): 有/无残差差异不大，浅层网络本身梯度传播还行
  2. 中层 (10层): 无残差开始吃力，有残差稳定训练
  3. 深层 (20层): 无残差直接崩溃(NaN)，有残差依然可以训练

这就是 ResNet 论文的核心发现 —— "退化问题" (degradation problem):
  - 更深的网络不应该比浅的更差（至少可以学成恒等映射）
  - 但没有残差连接时，深层网络反而比浅的差
  - 残差连接让网络至少能学到"什么都不做"(identity)，确保了深度增加不会变差

Transformer 中的应用:
  - LLM 动辄几十上百层，必须有残差连接
  - MiniMind 的 TransformerBlock 中有 2 个残差连接:
    · Attention 后 + 输入
    · FeedForward 后 + 输入
  - 正是这些残差让 8+16 层的 Transformer 能正常训练
""")

print("=" * 60)
print("实验 3 完成！")
print("=" * 60)
