"""
实验 2：组件顺序会改变 Block 表达的函数。

本实验只证明 Mixer 与 FFN 通常不可交换，不声称某个顺序对所有任务都更优。
"""

import argparse
import json
from pathlib import Path

import torch


SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def mixer(x: torch.Tensor) -> torch.Tensor:
    """混合序列信息：当前位置加入全序列均值和前一位置。"""
    return 0.5 * x.mean(dim=1, keepdim=True) + 0.25 * torch.roll(x, shifts=1, dims=1)


def ffn(x: torch.Tensor) -> torch.Tensor:
    """逐 token 非线性函数。"""
    return 0.3 * torch.sin(2.0 * x) + 0.1 * x.square()


def attention_then_ffn(x: torch.Tensor) -> torch.Tensor:
    x = x + mixer(rms_norm(x))
    return x + ffn(rms_norm(x))


def ffn_then_attention(x: torch.Tensor) -> torch.Tensor:
    x = x + ffn(rms_norm(x))
    return x + mixer(rms_norm(x))


def run(quick: bool) -> dict:
    del quick
    torch.manual_seed(SEED)
    x = torch.randn(32, 7, 12)
    first = attention_then_ffn(x)
    second = ffn_then_attention(x)

    absolute_difference = (first - second).abs()
    result = {
        "seed": SEED,
        "input_shape": list(x.shape),
        "output_shape": list(first.shape),
        "mean_absolute_difference": absolute_difference.mean().item(),
        "maximum_absolute_difference": absolute_difference.max().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            first.flatten(1), second.flatten(1), dim=-1
        ).mean().item(),
    }

    assert first.shape == x.shape == second.shape
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert result["mean_absolute_difference"] > 0.01, "交换顺序后输出意外等价。"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    result = run(args.quick)
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "exp2_order_matters.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("实验 2 通过：交换 Mixer 与 FFN 顺序会改变 Block 函数")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
