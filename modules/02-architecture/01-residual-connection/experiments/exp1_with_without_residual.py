"""
实验 1：有/无残差连接的训练对比
直观展示残差连接如何帮助网络训练

核心问题：
  1. 残差连接真的有用吗？
  2. 有残差 vs 无残差，训练速度和最终效果差多少？
  3. 梯度是怎么流得更好的？

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
# 0. 生成模拟数据
# ============================================================
print("=" * 60)
print("实验 1：有/无残差连接的训练对比")
print("=" * 60)

torch.manual_seed(42)

# 生成一个简单的回归任务数据
# y = sin(3x₁) + cos(2x₂) + 0.1 * noise
n_samples = 500
x = torch.randn(n_samples, 10)  # 10 维输入
y = torch.sin(3 * x[:, 0]) + torch.cos(2 * x[:, 1]) + 0.1 * torch.randn(n_samples)
y = y.unsqueeze(1)

print(f"\n数据: {n_samples} 个样本, x 维度=10, y 维度=1")
print(f"x 范围: [{x.min():.2f}, {x.max():.2f}]")
print(f"y 范围: [{y.min():.2f}, {y.max():.2f}]")


# ============================================================
# 1. 定义两种网络：有残差 / 无残差
# ============================================================
print("\n" + "-" * 60)
print("【1. 构建网络：有残差 vs 无残差】")
print("-" * 60)


class PlainNet(nn.Module):
    """普通网络（无残差连接）"""
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=5):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualNet(nn.Module):
    """带残差连接的网络"""
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=5):
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
    """一个残差块: y = F(x) + x"""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        # 子网络 F(x)
        residual = F.relu(self.fc1(x))
        residual = self.fc2(residual)

        # 残差连接: y = F(x) + x
        return F.relu(residual + x)


print(f"PlainNet 参数量: {sum(p.numel() for p in PlainNet().parameters()):,}")
print(f"ResidualNet 参数量: {sum(p.numel() for p in ResidualNet().parameters()):,}")


# ============================================================
# 2. 训练对比
# ============================================================
print("\n" + "-" * 60)
print("【2. 训练对比】")
print("-" * 60)


def train_one_epoch(model, x, y, optimizer, batch_size=32):
    """训练一个 epoch，返回平均 loss"""
    model.train()
    indices = torch.randperm(len(x))
    total_loss = 0
    n_batches = 0

    for i in range(0, len(x), batch_size):
        batch_idx = indices[i:i+batch_size]
        x_batch = x[batch_idx]
        y_batch = y[batch_idx]

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = F.mse_loss(pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


# 同时训练两个网络
lr = 0.001
epochs = 500

# 为公平对比，参数量相近
plain_net = PlainNet(num_layers=5, hidden_dim=64)
residual_net = ResidualNet(num_layers=5, hidden_dim=64)

plain_opt = torch.optim.Adam(plain_net.parameters(), lr=lr)
residual_opt = torch.optim.Adam(residual_net.parameters(), lr=lr)

plain_losses = []
residual_losses = []

print(f"训练 {epochs} 个 epoch...")
for epoch in range(epochs):
    plain_loss = train_one_epoch(plain_net, x, y, plain_opt)
    residual_loss = train_one_epoch(residual_net, x, y, residual_opt)

    plain_losses.append(plain_loss)
    residual_losses.append(residual_loss)

    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1:3d}: Plain={plain_loss:.6f}, Residual={residual_loss:.6f}")

print(f"\n最终 Loss:")
print(f"  无残差: {plain_losses[-1]:.6f}")
print(f"  有残差: {residual_losses[-1]:.6f}")
print(f"  改善: {(plain_losses[-1] - residual_losses[-1]) / plain_losses[-1] * 100:.1f}%")


# ============================================================
# 3. 可视化：Loss 曲线
# ============================================================
print("\n" + "-" * 60)
print("【3. 可视化结果】")
print("-" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 图 1: 完整 Loss 曲线
axes[0].plot(plain_losses, label='无残差 (Plain)', alpha=0.8, linewidth=1.5)
axes[0].plot(residual_losses, label='有残差 (Residual)', alpha=0.8, linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('训练 Loss 曲线')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图 2: 前 50 epoch 放大
axes[1].plot(plain_losses[:50], label='无残差', alpha=0.8, linewidth=1.5)
axes[1].plot(residual_losses[:50], label='有残差', alpha=0.8, linewidth=1.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE Loss')
axes[1].set_title('前 50 epoch（初期收敛速度）')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 图 3: 最终 Loss 对比柱状图
final_losses = {
    '无残差': plain_losses[-1],
    '有残差': residual_losses[-1]
}
bars = axes[2].bar(final_losses.keys(), final_losses.values(),
                   color=['#ff6b6b', '#51cf66'])
axes[2].set_ylabel('Final MSE Loss')
axes[2].set_title('最终 Loss 对比')
axes[2].grid(True, alpha=0.3, axis='y')

# 在柱状图上标注数值
for bar, (name, val) in zip(bars, final_losses.items()):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('实验 1：残差连接对训练的影响', fontsize=14, fontweight='bold')
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'exp1_with_without_residual.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图表已保存到 results/exp1_with_without_residual.png")


# ============================================================
# 4. 梯度流分析
# ============================================================
print("\n" + "-" * 60)
print("【4. 梯度流分析】")
print("-" * 60)

# 选择一个批次，计算并比较梯度
x_batch = x[:32]
y_batch = y[:32]


def get_gradients(model, x, y):
    """获取模型中各层参数的梯度范数"""
    model.zero_grad()
    pred = model(x)
    loss = F.mse_loss(pred, y)
    loss.backward()

    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            grads[name] = param.grad.norm().item()
    return grads


plain_grads = get_gradients(plain_net, x_batch, y_batch)
residual_grads = get_gradients(residual_net, x_batch, y_batch)

print("\n梯度范数对比（越大表示梯度流越好）：")
print(f"{'层':<30} {'无残差':<15} {'有残差':<15}")
print("-" * 60)

# 对齐层名（两者结构不同，分别打印）
print("PlainNet 各层梯度:")
for name, g in plain_grads.items():
    indicator = "⚠️ 偏小" if g < 1e-3 else ("✅" if g > 1e-2 else "  ")
    print(f"  {name:<28} {g:.6f}  {indicator}")

print("\nResidualNet 各层梯度:")
for name, g in residual_grads.items():
    indicator = "⚠️ 偏小" if g < 1e-3 else ("✅" if g > 1e-2 else "  ")
    print(f"  {name:<28} {g:.6f}  {indicator}")

# 梯度可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PlainNet 梯度
plain_names = list(plain_grads.keys())
plain_values = list(plain_grads.values())
axes[0].barh(plain_names, plain_values, color='#ff6b6b')
axes[0].set_xlabel('Gradient Norm')
axes[0].set_title('无残差网络 - 各层梯度范数')
axes[0].axvline(x=1e-3, color='gray', linestyle='--', alpha=0.5, label='危险线 (1e-3)')
axes[0].legend()
axes[0].set_xscale('log')
axes[0].grid(True, alpha=0.3, axis='x')

# ResidualNet 梯度
residual_names = list(residual_grads.keys())
residual_values = list(residual_grads.values())
axes[1].barh(residual_names, residual_values, color='#51cf66')
axes[1].set_xlabel('Gradient Norm')
axes[1].set_title('有残差网络 - 各层梯度范数')
axes[1].axvline(x=1e-3, color='gray', linestyle='--', alpha=0.5, label='危险线 (1e-3)')
axes[1].legend()
axes[1].set_xscale('log')
axes[1].grid(True, alpha=0.3, axis='x')

plt.suptitle('梯度流对比：有残差 vs 无残差', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'exp1_gradient_flow.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ 梯度图已保存到 results/exp1_gradient_flow.png")


# ============================================================
# 5. 残差连接的数学原理
# ============================================================
print("\n" + "=" * 60)
print("【5. 残差连接的数学原理】")
print("=" * 60)
print("""
残差连接的核心公式:
  y = F(x) + x

其中:
  - x 是输入（身份映射，identity mapping）
  - F(x) 是子网络学到的"残差"
  - y 是输出

梯度流分析:
  正向: y = F(x) + x
  反向: ∂L/∂x = ∂L/∂y · ∂y/∂x
              = ∂L/∂y · (∂F(x)/∂x + 1)
                ↑                      ↑
          来自残差路径            来自身份路径（永远是 1）

关键洞察: "+1" 这一项保证了梯度至少为 1，不会衰减到 0
         这就是残差连接解决梯度消失的核心！

对比无残差:
  y = F(x)
  ∂L/∂x = ∂L/∂y · ∂F(x)/∂x   ← 如果 ∂F/∂x < 1，连乘后梯度急剧衰减

直观理解:
  - 无残差: 每层都可能"衰减"梯度 → 深层梯度消失
  - 有残差: 梯度始终有一条"高速公路"（+1）直达底层
""")

print("=" * 60)
print("实验 1 完成！")
print("=" * 60)
