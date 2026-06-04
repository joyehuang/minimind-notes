"""
实验 2：梯度流对比。

保留每层激活的梯度，测量梯度从输出端传回输入端后的保留比例。

运行：
    python3 exp2_gradient_flow.py
    python3 exp2_gradient_flow.py --quick
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn


SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"


class GradientStack(nn.Module):
    def __init__(self, dim: int, depth: int, use_residual: bool):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(depth)])
        self.use_residual = use_residual
        self.residual_scale = 0.1

        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.5 / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        activations = [x]
        x.retain_grad()

        for layer in self.layers:
            branch = torch.tanh(layer(x))
            x = x + self.residual_scale * branch if self.use_residual else branch
            x.retain_grad()
            activations.append(x)

        return x, activations


def measure(use_residual: bool, dim: int, depth: int) -> dict:
    torch.manual_seed(SEED)
    model = GradientStack(dim, depth, use_residual)
    x = torch.randn(32, dim, requires_grad=True)
    output, activations = model(x)
    loss = output.square().mean()
    loss.backward()

    gradient_norms = [activation.grad.norm().item() for activation in activations]
    flow_ratio = gradient_norms[0] / gradient_norms[-1]

    return {
        "input_gradient_norm": gradient_norms[0],
        "middle_gradient_norm": gradient_norms[len(gradient_norms) // 2],
        "output_gradient_norm": gradient_norms[-1],
        "flow_ratio": flow_ratio,
        "minimum_gradient_norm": min(gradient_norms),
        "maximum_gradient_norm": max(gradient_norms),
    }


def run(quick: bool) -> dict:
    dim = 32
    depth = 24 if quick else 48
    plain = measure(False, dim, depth)
    residual = measure(True, dim, depth)
    plain_flow_floor = max(plain["flow_ratio"], 1e-30)

    result = {
        "seed": SEED,
        "depth": depth,
        "plain": plain,
        "residual": residual,
        "plain_gradient_underflowed": plain["flow_ratio"] == 0.0,
        "flow_ratio_improvement_lower_bound": residual["flow_ratio"] / plain_flow_floor,
    }

    all_values = [
        plain["input_gradient_norm"],
        plain["output_gradient_norm"],
        residual["input_gradient_norm"],
        residual["output_gradient_norm"],
        result["flow_ratio_improvement_lower_bound"],
    ]
    assert torch.isfinite(torch.tensor(all_values, dtype=torch.float64)).all()
    assert residual["flow_ratio"] > 0.05, "残差网络的输入端梯度保留比例过低。"
    assert result["flow_ratio_improvement_lower_bound"] > 100.0, (
        "残差连接没有表现出预期的梯度流改善；请检查环境或实验实现。"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="使用较浅网络")
    args = parser.parse_args()

    result = run(args.quick)
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "exp2_gradient_flow.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("实验 2 通过：残差路径显著改善输入端梯度保留")
    print(json.dumps(result, indent=2))
    print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()
