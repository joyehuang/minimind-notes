"""
实验基类

提供统一的实验框架，包括：
- 自动设备检测（CPU/MPS/CUDA）
- 结果保存（图表 + 指标）
- 进度显示
- 可复现性（固定随机种子）

使用示例：
    from modules.common.experiment_base import Experiment

    class MyExperiment(Experiment):
        def __init__(self):
            super().__init__(
                name="norm_comparison",
                output_dir="modules/01-foundation/01-normalization/experiments/results"
            )

        def run(self):
            # 你的实验代码
            pass

    exp = MyExperiment()
    exp.run()
"""

import torch
import random
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Heiti SC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class Experiment:
    """实验基类"""

    def __init__(
        self,
        name: str,
        output_dir: str | Path,
        seed: int = 42,
        device: Optional[str] = None
    ):
        """
        Args:
            name: 实验名称（用于保存文件）
            output_dir: 结果输出目录
            seed: 随机种子
            device: 设备（None 表示自动检测）
        """
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seed = seed
        self.device = device or self._auto_detect_device()

        # 设置随机种子
        self._set_seed(seed)

        # 实验元数据
        self.metadata = {
            'name': name,
            'start_time': datetime.now().isoformat(),
            'device': self.device,
            'seed': seed,
        }

        print(f"🔬 实验: {name}")
        print(f"📂 输出目录: {self.output_dir}")
        print(f"🎲 随机种子: {seed}")
        print(f"💻 设备: {self.device}")
        print("-" * 60)

    def _auto_detect_device(self) -> str:
        """自动检测可用设备"""

        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        return device

    def _set_seed(self, seed: int):
        """设置所有随机种子以保证可复现性"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def save_figure(self, name: str, fig: Optional[plt.Figure] = None):
        """
        保存图表

        Args:
            name: 文件名（不含扩展名）
            fig: matplotlib Figure 对象（None 表示当前 figure）
        """
        if fig is None:
            fig = plt.gcf()

        filepath = self.output_dir / f"{name}.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"💾 保存图表: {filepath}")

    def save_metrics(self, metrics: Dict[str, Any], name: str = "metrics"):
        """
        保存实验指标

        Args:
            metrics: 指标字典
            name: 文件名（不含扩展名）
        """
        filepath = self.output_dir / f"{name}.json"

        # 添加元数据
        output = {
            'metadata': self.metadata,
            'metrics': metrics
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"💾 保存指标: {filepath}")

    def print_metrics(self, metrics: Dict[str, Any], title: str = "实验结果"):
        """
        美化打印指标

        Args:
            metrics: 指标字典
            title: 标题
        """
        print("\n" + "=" * 60)
        print(f"📊 {title}")
        print("=" * 60)

        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            elif isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        print("=" * 60 + "\n")

    def run(self):
        """子类需要实现这个方法"""
        raise NotImplementedError("子类必须实现 run() 方法")


class TrainingExperiment(Experiment):
    """训练实验基类（包含训练循环的通用逻辑）"""

    def __init__(
        self,
        name: str,
        output_dir: str | Path,
        seed: int = 42,
        device: Optional[str] = None
    ):
        super().__init__(name, output_dir, seed, device)

        # 训练历史
        self.history = {
            'loss': [],
            'step': [],
        }

    def train_step(
        self,
        model: torch.nn.Module,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        单步训练（子类可以覆盖）

        Args:
            model: 模型
            batch: 输入数据
            optimizer: 优化器

        Returns:
            loss: 损失值
        """
        raise NotImplementedError("子类需要实现 train_step()")

    def train(
        self,
        model: torch.nn.Module,
        train_data: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        num_steps: int,
        batch_size: int = 32,
        log_interval: int = 100
    ) -> Dict[str, list]:
        """
        训练循环

        Args:
            model: 模型
            train_data: 训练数据
            optimizer: 优化器
            num_steps: 训练步数
            batch_size: 批次大小
            log_interval: 日志打印间隔

        Returns:
            history: 训练历史
        """

        model.train()
        model.to(self.device)

        print(f"🚀 开始训练: {num_steps} 步")

        for step in range(num_steps):
            # 随机采样批次
            indices = torch.randint(0, len(train_data) - 1, (batch_size,))
            batch = train_data[indices].to(self.device)

            # 训练步
            loss = self.train_step(model, batch, optimizer)

            # 记录
            self.history['loss'].append(loss)
            self.history['step'].append(step)

            # 打印
            if (step + 1) % log_interval == 0:
                print(f"Step {step + 1}/{num_steps}, Loss: {loss:.4f}")

            # 检查 NaN
            if np.isnan(loss):
                print(f"⚠️ 训练在第 {step + 1} 步出现 NaN，停止训练")
                break

        print(f"✅ 训练完成")
        return self.history

    def plot_training_curve(self, histories: Dict[str, Dict[str, list]], title: str = "训练曲线"):
        """
        绘制训练曲线对比

        Args:
            histories: {config_name: history} 字典
            title: 图表标题
        """

        plt.figure(figsize=(10, 6))

        for config_name, history in histories.items():
            steps = history['step']
            losses = history['loss']

            # 检查是否出现 NaN
            if any(np.isnan(losses)):
                nan_idx = next(i for i, x in enumerate(losses) if np.isnan(x))
                steps = steps[:nan_idx]
                losses = losses[:nan_idx]
                label = f"{config_name} (NaN @ step {steps[-1] if steps else 0})"
                linestyle = '--'
            else:
                label = config_name
                linestyle = '-'

            plt.plot(steps, losses, label=label, linestyle=linestyle)

        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        self.save_figure(f"{self.name}_training_curve")
        plt.close()


if __name__ == '__main__':
    # 测试
    class DummyExperiment(Experiment):
        def run(self):
            # 测试保存功能
            metrics = {
                'accuracy': 0.95,
                'loss': 0.05,
                'config': {'lr': 0.001, 'batch_size': 32}
            }

            self.print_metrics(metrics)
            self.save_metrics(metrics)

            # 测试保存图表
            plt.figure(figsize=(8, 6))
            plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
            plt.title("Test Plot")
            self.save_figure("test_plot")

            print("✅ 测试完成")

    exp = DummyExperiment(
        name="test",
        output_dir="modules/common/test_output"
    )
    exp.run()
