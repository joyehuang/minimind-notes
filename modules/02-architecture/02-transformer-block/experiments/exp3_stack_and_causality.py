"""
实验 3：完整 Pre-Norm Decoder Block 的堆叠与因果性测试。

验证：
1. 多层堆叠后形状保持不变
2. 修改未来 token 不会影响过去位置
3. 输入梯度有限且非零
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn


SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        mixed = weights @ v
        mixed = mixed.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.output(mixed)


class FeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.SiLU(),
            nn.Linear(dim * 4, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attention = CausalSelfAttention(dim, heads)
        self.ffn = FeedForward(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class DecoderStack(nn.Module):
    def __init__(self, dim: int, heads: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(dim, heads) for _ in range(depth)])
        self.final_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


def measure(depth: int) -> dict:
    torch.manual_seed(SEED + depth)
    model = DecoderStack(dim=16, heads=4, depth=depth)
    x = torch.randn(2, 8, 16, requires_grad=True)
    changed = x.detach().clone()
    changed[:, -1, :] += 10.0

    output = model(x)
    changed_output = model(changed)
    past_difference = (output[:, :-1] - changed_output[:, :-1]).abs().max().item()

    loss = output[:, -1].square().mean()
    loss.backward()
    input_gradient_norm = x.grad.norm().item()

    return {
        "depth": depth,
        "input_shape": list(x.shape),
        "output_shape": list(output.shape),
        "past_output_max_difference_after_future_change": past_difference,
        "input_gradient_norm": input_gradient_norm,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def run(quick: bool) -> dict:
    depths = [2, 8] if quick else [2, 8, 16]
    measurements = [measure(depth) for depth in depths]
    result = {"seed": SEED, "measurements": measurements}

    for measurement in measurements:
        assert measurement["input_shape"] == measurement["output_shape"]
        assert measurement["past_output_max_difference_after_future_change"] < 1e-6
        assert math.isfinite(measurement["input_gradient_norm"])
        assert measurement["input_gradient_norm"] > 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    result = run(args.quick)
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "exp3_stack_and_causality.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("实验 3 通过：Decoder Block 可堆叠、保持因果性且梯度有限")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
