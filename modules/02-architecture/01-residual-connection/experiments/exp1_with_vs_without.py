"""
实验 1：有/无残差连接的训练对比。

任务是学习接近恒等映射的函数 y = x + 0.25 * sin(2x)。
残差网络只需学习增量，普通深层网络必须重建完整输出。

运行：
    python3 exp1_with_vs_without.py
    python3 exp1_with_vs_without.py --quick
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn


SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"


class PlainNetwork(nn.Module):
    """普通深层网络：每层完全覆盖上一层表示。"""

    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


class ResidualNetwork(nn.Module):
    """残差网络：每层只学习一个较小增量。"""

    def __init__(self, dim: int, depth: int, residual_scale: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + self.residual_scale * torch.tanh(layer(x))
        return x


def initialize(model: nn.Module) -> None:
    """使用相同规则初始化，保证实验可复现。"""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)
            nn.init.zeros_(module.bias)


def train(model: nn.Module, x: torch.Tensor, target: torch.Tensor, steps: int) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []

    for _ in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def run(quick: bool) -> dict:
    torch.manual_seed(SEED)
    dim = 16
    depth = 16
    samples = 256 if quick else 1024
    steps = 100 if quick else 300

    x = torch.linspace(-1.0, 1.0, samples * dim).reshape(samples, dim)
    target = x + 0.25 * torch.sin(2.0 * x)

    torch.manual_seed(SEED)
    plain = PlainNetwork(dim, depth)
    initialize(plain)
    plain_losses = train(plain, x, target, steps)

    torch.manual_seed(SEED)
    residual = ResidualNetwork(dim, depth)
    initialize(residual)
    residual_losses = train(residual, x, target, steps)

    result = {
        "seed": SEED,
        "depth": depth,
        "steps": steps,
        "plain_initial_loss": plain_losses[0],
        "plain_final_loss": plain_losses[-1],
        "residual_initial_loss": residual_losses[0],
        "residual_final_loss": residual_losses[-1],
        "improvement_ratio": plain_losses[-1] / residual_losses[-1],
    }

    assert torch.isfinite(torch.tensor(list(result.values()), dtype=torch.float64)).all()
    assert result["residual_final_loss"] < result["plain_final_loss"] * 0.25, (
        "残差网络没有表现出预期的优化优势；请检查环境或实验实现。"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="使用更少样本和训练步数")
    args = parser.parse_args()

    result = run(args.quick)
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "exp1_with_vs_without.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("实验 1 通过：残差网络更容易学习接近恒等映射的任务")
    print(json.dumps(result, indent=2))
    print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()
