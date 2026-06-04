"""
实验 1：Transformer Block 组件消融。

合成目标同时需要跨 token 汇聚与逐 token 非线性变换：
target = x + 0.4 * mean(x) + 0.2 * sin(2x)

运行：
    python3 exp1_component_ablation.py
    python3 exp1_component_ablation.py --quick
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn


SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"


class MeanMixer(nn.Module):
    """简化的 Attention：汇聚所有 token，再广播回各位置。"""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.proj(x.mean(dim=1, keepdim=True))
        return mixed.expand_as(x)


class TokenFFN(nn.Module):
    """逐 token 非线性变换。"""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Tanh(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AblationBlock(nn.Module):
    def __init__(self, dim: int, use_mixer: bool, use_ffn: bool, use_residual: bool):
        super().__init__()
        self.use_mixer = use_mixer
        self.use_ffn = use_ffn
        self.use_residual = use_residual
        self.mixer_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.mixer = MeanMixer(dim)
        self.ffn = TokenFFN(dim)

    def combine(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return x + delta if self.use_residual else delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mixer:
            x = self.combine(x, self.mixer(self.mixer_norm(x)))
        if self.use_ffn:
            x = self.combine(x, self.ffn(self.ffn_norm(x)))
        return x


def train(configuration: tuple[bool, bool, bool], x: torch.Tensor, target: torch.Tensor, steps: int) -> float:
    torch.manual_seed(SEED)
    model = AblationBlock(x.shape[-1], *configuration)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()

    return nn.functional.mse_loss(model(x), target).item()


def run(quick: bool) -> dict:
    torch.manual_seed(SEED)
    samples = 128 if quick else 512
    steps = 200 if quick else 600
    x = torch.randn(samples, 6, 8)
    target = x + 0.4 * x.mean(dim=1, keepdim=True) + 0.2 * torch.sin(2.0 * x)

    configurations = {
        "full_block": (True, True, True),
        "without_mixer": (False, True, True),
        "without_ffn": (True, False, True),
        "without_residual": (True, True, False),
    }
    losses = {
        name: train(configuration, x, target, steps)
        for name, configuration in configurations.items()
    }

    result = {
        "seed": SEED,
        "steps": steps,
        "final_losses": losses,
        "improvement_ratios": {
            name: loss / losses["full_block"]
            for name, loss in losses.items()
            if name != "full_block"
        },
    }
    assert torch.isfinite(torch.tensor(list(losses.values()))).all()
    assert losses["full_block"] < losses["without_mixer"] * 0.75
    assert losses["full_block"] < losses["without_ffn"] * 0.75
    assert losses["full_block"] < losses["without_residual"] * 0.75
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    result = run(args.quick)
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "exp1_component_ablation.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("实验 1 通过：完整 Block 在组合任务上优于组件消融版本")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
