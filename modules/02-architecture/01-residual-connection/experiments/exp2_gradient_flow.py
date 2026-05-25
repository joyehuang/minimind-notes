"""
实验 2：梯度流可视化
直观展示残差连接如何保持梯度在深层网络中的流动

核心问题：
  1. 深层网络中梯度是怎么"消失"的？
  2. 残差连接如何改变梯度的流动模式？
  3. 数值上看，有/无残差差多少？

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
print("实验 2：梯度流可视化")
print("=" * 60)


# ============================================================
# 1. 手动构造一个简单例子：5 层网络
# ============================================================
print("\n" + "-" * 60)
print("【1. 构建深层网络，监控各层梯度】")
print("-" * 60)

torch.manual_seed(42)

hidden_dim = 32
num_layers = 8

# 输入
x = torch.randn(1, hidden_dim, requires_grad=True)


class DeepNetNoResidual(nn.Module):
    """深层网络（无残差）"""
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DeepNetWithResidual(nn.Module):
    """深层网络（有残差）"""
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x  # 残差连接
        return x


net_no_res = DeepNetNoResidual(hidden_dim, num_layers)
net_with_res = DeepNetWithResidual(hidden_dim, num_layers)

# 用相同的权重初始化，保证公平对比
for p_no, p_with in zip(net_no_res.parameters(), net_with_res.parameters()):
    p_with.data = p_no.data.clone()


# ============================================================
# 2. Hook 各层输出，记录激活值
# ============================================================
print("\n" + "-" * 60)
print("【2. 记录各层输出和梯度】")
print("-" * 60)

# 注册 hook 来捕获各层输出
activations_no_res = []
activations_with_res = []


def hook_factory(storage):
    def hook(module, input, output):
        storage.append(output.detach())
    return hook


for layer in net_no_res.layers:
    layer.register_forward_hook(hook_factory(activations_no_res))

for layer in net_with_res.layers:
    layer.register_forward_hook(hook_factory(activations_with_res))

# 前向传播
out_no_res = net_no_res(x)
out_no_res.sum().backward()

out_with_res = net_with_res(x)
out_with_res.sum().backward()

# 计算各层激活值的统计量
print("\n无残差网络 - 各层输出统计：")
print(f"{'层':<8} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
for i, act in enumerate(activations_no_res):
    print(f"Layer {i+1}: {act.mean().item():10.6f} {act.std().item():10.6f} "
          f"{act.min().item():10.6f} {act.max().item():10.6f}")

print("\n有残差网络 - 各层输出统计：")
print(f"{'层':<8} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
for i, act in enumerate(activations_with_res):
    print(f"Layer {i+1}: {act.mean().item():10.6f} {act.std().item():10.6f} "
          f"{act.min().item():10.6f} {act.max().item():10.6f}")


# ============================================================
# 3. 收集各层权重的梯度
# ============================================================
print("\n" + "-" * 60)
print("【3. 梯度范数对比】")
print("-" * 60)

grads_no_res = []
grads_with_res = []

for name, param in net_no_res.named_parameters():
    if 'weight' in name and param.grad is not None:
        grads_no_res.append(param.grad.norm().item())

for name, param in net_with_res.named_parameters():
    if 'weight' in name and param.grad is not None:
        grads_with_res.append(param.grad.norm().item())

print(f"\n{'层':<8} {'无残差梯度':<18} {'有残差梯度':<18} {'比值':<12}")
print("-" * 60)
for i, (g_no, g_with) in enumerate(zip(grads_no_res, grads_with_res)):
    ratio = g_with / g_no if g_no > 0 else float('inf')
    print(f"Layer {i+1}: {g_no:<18.8f} {g_with:<18.8f} {ratio:<12.2f}x")

print(f"\n第一层（最浅）梯度比值: {grads_with_res[0] / grads_no_res[0]:.2f}x")
print(f"最后一层（最深）梯度比值: {grads_with_res[-1] / grads_no_res[-1]:.2f}x")


# ============================================================
# 4. 可视化梯度流
# ============================================================
print("\n" + "-" * 60)
print("【4. 可视化】")
print("-" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
layers = list(range(1, len(grads_no_res) + 1))

# 图 1: 梯度范数折线图
axes[0].plot(layers, grads_no_res, 'o-', color='#ff6b6b', linewidth=2,
             markersize=8, label='无残差')
axes[0].plot(layers, grads_with_res, 's-', color='#51cf66', linewidth=2,
             markersize=8, label='有残差')
axes[0].set_xlabel('层序号（1=最浅，N=最深）')
axes[0].set_ylabel('梯度范数')
axes[0].set_title('各层权重梯度范数')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# 图 2: 激活值分布对比（第一层 vs 最后一层）
# 无残差
no_res_first = activations_no_res[0].numpy().flatten()
no_res_last = activations_no_res[-1].numpy().flatten()
# 有残差
with_res_first = activations_with_res[0].numpy().flatten()
with_res_last = activations_with_res[-1].numpy().flatten()

bins = 15
axes[1].hist(no_res_first, bins=bins, alpha=0.5, label='无残差-第1层', color='#ff6b6b')
axes[1].hist(no_res_last, bins=bins, alpha=0.5, label='无残差-第8层', color='#ff8787')
axes[1].hist(with_res_first, bins=bins, alpha=0.3, label='有残差-第1层', color='#51cf66')
axes[1].hist(with_res_last, bins=bins, alpha=0.3, label='有残差-第8层', color='#69db7c')
axes[1].set_xlabel('激活值')
axes[1].set_ylabel('频数')
axes[1].set_title('激活值分布（浅层 vs 深层）')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis='y')

# 图 3: 梯度衰减曲线（归一化对比）
grads_no_res_norm = np.array(grads_no_res) / max(grads_no_res)
grads_with_res_norm = np.array(grads_with_res) / max(grads_with_res)

axes[2].plot(layers, grads_no_res_norm, 'o-', color='#ff6b6b', linewidth=2,
             markersize=8, label='无残差')
axes[2].plot(layers, grads_with_res_norm, 's-', color='#51cf66', linewidth=2,
             markersize=8, label='有残差')
axes[2].axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='衰减 99%')
axes[2].set_xlabel('层序号')
axes[2].set_ylabel('归一化梯度（相对第1层）')
axes[2].set_title('梯度衰减对比（对数坐标）')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_yscale('log')

plt.suptitle('实验 2：残差连接对梯度流的影响', fontsize=14, fontweight='bold')
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'exp2_gradient_flow.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ 图表已保存到 results/exp2_gradient_flow.png")


# ============================================================
# 5. 手动计算：为什么残差连接保持梯度
# ============================================================
print("\n" + "=" * 60)
print("【5. 残差连接的梯度数学】")
print("=" * 60)

# 用一个简单的例子说明：3 层网络
torch.manual_seed(42)

# 模拟权重
w1 = torch.tensor(0.5, requires_grad=True)  # 第1层"缩放因子"
w2 = torch.tensor(0.5, requires_grad=True)  # 第2层
w3 = torch.tensor(0.5, requires_grad=True)  # 第3层

x_val = torch.tensor(1.0)

# === 无残差 ===
h1_no = w1 * x_val  # y1 = 0.5 * 1 = 0.5
h2_no = w2 * h1_no  # y2 = 0.5 * 0.5 = 0.25
h3_no = w3 * h2_no  # y3 = 0.5 * 0.25 = 0.125

h3_no.backward()

print("无残差网络的梯度传播（所有权重=0.5）：")
print(f"  ∂y/∂w₁ = {w1.grad.item():.4f}  ← 经过 3 层连乘")
print(f"  ∂y/∂w₂ = {w2.grad.item():.4f}")
print(f"  ∂y/∂w₃ = {w3.grad.item():.4f}")
print(f"  衰减倍数 (w₃→w₁): {w3.grad.item() / w1.grad.item():.1f}x")
print(f"  公式: ∂y/∂w₁ = x · w₃ · w₂ = {x_val.item()} × 0.5 × 0.5 = 0.25")

# 重置
w1.grad = None; w2.grad = None; w3.grad = None

# === 有残差 ===
h1_w = w1 * x_val          # 0.5
h1_r = h1_w + x_val        # 0.5 + 1.0 = 1.5 (残差)
h2_w = w2 * h1_r           # 0.5 * 1.5 = 0.75
h2_r = h2_w + h1_r         # 0.75 + 1.5 = 2.25
h3_w = w3 * h2_r           # 0.5 * 2.25 = 1.125
h3_r = h3_w + h2_r         # 1.125 + 2.25 = 3.375

h3_r.backward()

print("\n有残差网络的梯度传播：")
print(f"  ∂y/∂w₁ = {w1.grad.item():.4f}  ← 残差保持梯度")
print(f"  ∂y/∂w₂ = {w2.grad.item():.4f}")
print(f"  ∂y/∂w₃ = {w3.grad.item():.4f}")
print(f"  衰减倍数 (w₃→w₁): {w3.grad.item() / w1.grad.item():.1f}x")
print(f"  关键: 残差路径(+x)提供了额外的梯度通道，不会连乘衰减")

print("\n" + "-" * 60)
print("核心公式:")
print("  无残差: y = F₃(F₂(F₁(x)))")
print("          ∂y/∂x = ∂F₃/∂h₂ · ∂F₂/∂h₁ · ∂F₁/∂x  ← 连乘，容易衰减")
print()
print("  有残差: y = (...(F₁(x) + x) + ...) + x")
print("          ∂y/∂x = Π(∂Fᵢ/∂hᵢ₋₁) + 1 + 1 + ... + 1  ← 额外常数项！")
print("                    ↑ 连乘部分              ↑ 残差提供的常数路径")

print("=" * 60)
print("实验 2 完成！")
print("=" * 60)
