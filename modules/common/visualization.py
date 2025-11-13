"""
可视化工具

提供常用的可视化函数，用于实验结果展示

包括：
- 注意力权重热力图
- 激活分布可视化
- 梯度流可视化
- 对比柱状图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    tokens: Optional[List[str]] = None,
    title: str = "Attention Weights",
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    绘制注意力权重热力图

    Args:
        attention_weights: 注意力权重 [seq_len, seq_len]
        tokens: Token 列表（用于坐标轴标签）
        title: 图表标题
        figsize: 图表大小

    Returns:
        fig: matplotlib Figure 对象
    """

    # 转换为 numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热力图
    sns.heatmap(
        attention_weights,
        cmap='YlOrRd',
        annot=False,
        fmt='.2f',
        cbar=True,
        square=True,
        ax=ax
    )

    # 设置标签
    if tokens:
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens, rotation=0)

    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_activation_distribution(
    activations: Dict[str, torch.Tensor],
    title: str = "Activation Distribution",
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    绘制激活值分布对比

    Args:
        activations: {layer_name: activation_tensor} 字典
        title: 图表标题
        figsize: 图表大小

    Returns:
        fig: matplotlib Figure 对象
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    layer_names = list(activations.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))

    # 左图：分布直方图
    for i, (name, tensor) in enumerate(activations.items()):
        values = tensor.detach().cpu().numpy().flatten()
        axes[0].hist(values, bins=50, alpha=0.5, label=name, color=colors[i])

    axes[0].set_xlabel('Activation Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图：统计指标（均值、标准差）
    means = [activations[name].mean().item() for name in layer_names]
    stds = [activations[name].std().item() for name in layer_names]

    x = np.arange(len(layer_names))
    width = 0.35

    axes[1].bar(x - width/2, means, width, label='Mean', color='skyblue')
    axes[1].bar(x + width/2, stds, width, label='Std', color='orange')

    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Statistics')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_gradient_flow(
    named_parameters,
    title: str = "Gradient Flow",
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    绘制梯度流（检查梯度消失/爆炸）

    Args:
        named_parameters: model.named_parameters() 的输出
        title: 图表标题
        figsize: 图表大小

    Returns:
        fig: matplotlib Figure 对象
    """

    ave_grads = []
    max_grads = []
    layers = []

    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(layers))
    ax.bar(x - 0.2, ave_grads, 0.4, label='Average Gradient', color='skyblue')
    ax.bar(x + 0.2, max_grads, 0.4, label='Max Gradient', color='orange')

    ax.set_xlabel('Layers')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=90, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')  # 对数坐标，便于观察

    plt.tight_layout()
    return fig


def plot_comparison_bars(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str = "Metric",
    title: str = "Comparison",
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    绘制对比柱状图

    Args:
        metrics: {config_name: {metric1: value1, metric2: value2}} 字典
        metric_name: 指标名称
        title: 图表标题
        figsize: 图表大小

    Returns:
        fig: matplotlib Figure 对象

    Example:
        metrics = {
            'No Norm': {'final_loss': 4.2, 'time': 100},
            'RMSNorm': {'final_loss': 2.1, 'time': 95},
        }
    """

    fig, ax = plt.subplots(figsize=figsize)

    # 提取配置和指标
    configs = list(metrics.keys())
    metric_keys = list(next(iter(metrics.values())).keys())

    x = np.arange(len(configs))
    width = 0.8 / len(metric_keys)

    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_keys)))

    # 绘制每个指标
    for i, key in enumerate(metric_keys):
        values = [metrics[config][key] for config in configs]
        offset = (i - len(metric_keys) / 2) * width + width / 2
        ax.bar(x + offset, values, width, label=key, color=colors[i])

    ax.set_xlabel('Configuration')
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_loss_curves(
    histories: Dict[str, List[float]],
    steps: Optional[List[int]] = None,
    title: str = "Loss Curves",
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    绘制多条 loss 曲线对比

    Args:
        histories: {config_name: loss_list} 字典
        steps: 步数列表（如果为 None，使用索引）
        title: 图表标题
        figsize: 图表大小

    Returns:
        fig: matplotlib Figure 对象
    """

    fig, ax = plt.subplots(figsize=figsize)

    for config_name, losses in histories.items():
        x = steps if steps is not None else list(range(len(losses)))

        # 检查 NaN
        valid_idx = [i for i, loss in enumerate(losses) if not np.isnan(loss)]

        if len(valid_idx) < len(losses):
            # 出现 NaN
            x_valid = [x[i] for i in valid_idx]
            losses_valid = [losses[i] for i in valid_idx]
            label = f"{config_name} (NaN @ step {x[len(valid_idx)] if len(valid_idx) < len(x) else 'end'})"
            linestyle = '--'
        else:
            x_valid = x
            losses_valid = losses
            label = config_name
            linestyle = '-'

        ax.plot(x_valid, losses_valid, label=label, linestyle=linestyle)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # 测试
    print("🎨 测试可视化工具")

    # 1. 测试注意力热力图
    attn = torch.softmax(torch.randn(5, 5), dim=-1)
    tokens = ['The', 'cat', 'sat', 'on', 'mat']
    fig = plot_attention_heatmap(attn, tokens, "测试注意力热力图")
    plt.savefig('modules/common/test_output/test_attention.png')
    plt.close()
    print("✅ 注意力热力图")

    # 2. 测试激活分布
    activations = {
        'Layer 1': torch.randn(100, 512),
        'Layer 2': torch.randn(100, 512) * 0.5,
        'Layer 3': torch.randn(100, 512) * 0.1,
    }
    fig = plot_activation_distribution(activations, "测试激活分布")
    plt.savefig('modules/common/test_output/test_activation.png')
    plt.close()
    print("✅ 激活分布")

    # 3. 测试对比柱状图
    metrics = {
        'No Norm': {'final_loss': 4.2, 'time': 100, 'memory': 2.5},
        'LayerNorm': {'final_loss': 2.5, 'time': 95, 'memory': 2.8},
        'RMSNorm': {'final_loss': 2.1, 'time': 90, 'memory': 2.6},
    }
    fig = plot_comparison_bars(metrics, "Value", "测试对比图")
    plt.savefig('modules/common/test_output/test_comparison.png')
    plt.close()
    print("✅ 对比柱状图")

    print("✅ 所有测试完成")
