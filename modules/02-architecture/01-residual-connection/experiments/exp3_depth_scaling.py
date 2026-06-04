"""
实验 3：深度扩展与信息保留。

将同一输入依次通过 2、8、32、64 层随机子层，比较最终输出与输入的余弦相似度。

运行：
    python3 exp3_depth_scaling.py
    python3 exp3_depth_scaling.py --quick
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn


SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"


def make_layers(dim: int, depth: int) -> list[nn.Linear]:
    layers = []
    for _ in range(depth):
        layer = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(layer.weight, mean=0.0, std=0.5 / math.sqrt(dim))
        layers.append(layer)
    return layers


def forward(x: torch.Tensor, layers: list[nn.Linear], use_residual: bool) -> torch.Tensor:
    for layer in layers:
        branch = torch.tanh(layer(x))
        x = x + 0.05 * branch if use_residual else branch
    return x


def cosine_to_input(x: torch.Tensor, output: torch.Tensor) -> float:
    return nn.functional.cosine_similarity(x, output, dim=-1).mean().item()


def run(quick: bool) -> dict:
    torch.manual_seed(SEED)
    dim = 32
    depths = [2, 8, 32] if quick else [2, 8, 32, 64]
    x = torch.randn(128, dim)
    measurements = {}

    for depth in depths:
        torch.manual_seed(SEED + depth)
        layers = make_layers(dim, depth)
        plain_output = forward(x, layers, use_residual=False)
        residual_output = forward(x, layers, use_residual=True)

        measurements[str(depth)] = {
            "plain_cosine_similarity": cosine_to_input(x, plain_output),
            "residual_cosine_similarity": cosine_to_input(x, residual_output),
            "plain_output_rms": plain_output.square().mean().sqrt().item(),
            "residual_output_rms": residual_output.square().mean().sqrt().item(),
        }

    deepest = measurements[str(depths[-1])]
    result = {
        "seed": SEED,
        "depths": depths,
        "measurements": measurements,
        "deepest_similarity_gap": (
            deepest["residual_cosine_similarity"] - deepest["plain_cosine_similarity"]
        ),
    }

    assert torch.isfinite(
        torch.tensor(
            [
                deepest["plain_cosine_similarity"],
                deepest["residual_cosine_similarity"],
                result["deepest_similarity_gap"],
            ],
            dtype=torch.float64,
        )
    ).all()
    assert deepest["residual_cosine_similarity"] > 0.9, (
        "残差网络在深层堆叠后没有保留足够输入信息。"
    )
    assert result["deepest_similarity_gap"] > 0.7, (
        "有/无残差网络的信息保留差异不够明显；请检查实验实现。"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="跳过 64 层配置")
    args = parser.parse_args()

    result = run(args.quick)
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "exp3_depth_scaling.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("实验 3 通过：深度增加时，残差网络保留了更多输入信息")
    print(json.dumps(result, indent=2))
    print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()
