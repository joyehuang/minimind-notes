"""
FeedForward 前馈网络详解
理解"扩张-压缩"的意义

作者: joyehuang
日期: 2025-11-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("FeedForward 前馈网络：扩张-压缩的意义")
print("=" * 60)

# ============================================================
# 1. 简化的例子：2维 → 4维 → 2维
# ============================================================
print("\n【1. 简化例子：看到内容的变化】")
print("-" * 60)

# 输入：2维向量
x = torch.tensor([[1.0, 2.0]])  # [1, 2]
print(f"输入 x: {x}")
print(f"  维度: {x.shape}")

# 定义 FeedForward
hidden_dim = 4  # 中间扩展到 4 维

# 权重（简化，手工设置）
W1 = nn.Linear(2, 4, bias=False)  # 2 → 4
W2 = nn.Linear(4, 2, bias=False)  # 4 → 2

# 初始化权重（简化理解）
with torch.no_grad():
    W1.weight = nn.Parameter(torch.tensor([
        [1.0, 0.5],
        [0.5, 1.0],
        [-1.0, 0.5],
        [0.3, -0.8]
    ]))
    W2.weight = nn.Parameter(torch.tensor([
        [0.5, 0.2, -0.3, 0.4],
        [0.1, -0.5, 0.6, 0.2]
    ]))

print(f"\n权重 W1 (2→4): shape {W1.weight.shape}")
print(f"权重 W2 (4→2): shape {W2.weight.shape}")

# 前向传播
print(f"\n前向传播过程：")

# 步骤 1: 扩张
h1 = W1(x)  # [1, 4]
print(f"1. 扩张后 h1 = W1(x): {h1}")
print(f"   维度变化: [1,2] → [1,4]")

# 步骤 2: 非线性激活（关键！）
h2 = F.relu(h1)
print(f"\n2. ReLU激活后 h2 = ReLU(h1): {h2}")
print(f"   作用: 引入非线性，负数变0")

# 步骤 3: 压缩
output = W2(h2)  # [1, 2]
print(f"\n3. 压缩后 output = W2(h2): {output}")
print(f"   维度变化: [1,4] → [1,2]")

print(f"\n对比：")
print(f"  输入 x:      {x}")
print(f"  输出 output: {output}")
print(f"  维度相同，但内容完全不同！")

# ============================================================
# 2. 如果不扩张会怎样？
# ============================================================
print("\n\n【2. 对比：不扩张直接变换】")
print("-" * 60)

# 方案 A：直接 2 → 2（线性变换）
W_direct = nn.Linear(2, 2, bias=False)
with torch.no_grad():
    W_direct.weight = nn.Parameter(torch.tensor([
        [0.5, 0.3],
        [0.2, 0.4]
    ]))

output_direct = W_direct(x)
print(f"直接线性变换: {output_direct}")
print(f"  问题: 只是简单的线性组合，表达能力有限")

# 方案 B：扩张 → 激活 → 压缩（非线性）
print(f"\n扩张-压缩方案: {output}")
print(f"  优势: 经过高维空间，能做复杂的非线性变换")

# ============================================================
# 3. 真实的 MiniMind FeedForward
# ============================================================
print("\n\n【3. MiniMind 真实的 FeedForward (SwiGLU)】")
print("-" * 60)

class SimpleSwiGLU(nn.Module):
    def __init__(self, dim=8, hidden_dim=16):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # SwiGLU: gate 和 up 两个分支
        gate = self.gate_proj(x)      # [batch, seq, hidden_dim]
        up = self.up_proj(x)          # [batch, seq, hidden_dim]

        # SiLU 激活 + 门控
        hidden = F.silu(gate) * up    # 逐元素相乘

        # 压缩回原维度
        output = self.down_proj(hidden)
        return output

# 创建模型
model = SimpleSwiGLU(dim=8, hidden_dim=16)

# 输入：[1, 3, 8] (batch=1, 3个词, 8维)
x = torch.randn(1, 3, 8)
print(f"输入 x: shape {x.shape}")

# 前向传播
output = model(x)
print(f"输出 output: shape {output.shape}")

print(f"\n维度变化：")
print(f"  输入:  [1, 3, 8]")
print(f"  gate:  [1, 3, 16]  (扩张)")
print(f"  up:    [1, 3, 16]  (扩张)")
print(f"  hidden: [1, 3, 16]  (gate × up)")
print(f"  输出:  [1, 3, 8]   (压缩)")

# ============================================================
# 4. FeedForward 的作用演示
# ============================================================
print("\n\n【4. FeedForward 的实际作用】")
print("-" * 60)

# 模拟：词向量经过 FeedForward 后的变化
x = torch.tensor([
    [1.0, 2.0, 1.0, 0.5],  # "我" 的向量
    [0.5, 1.5, 2.0, 1.0],  # "爱" 的向量
    [2.0, 0.5, 1.0, 1.5],  # "编程" 的向量
])

print(f"输入（3个词的向量）:")
for i, word in enumerate(["我", "爱", "编程"]):
    print(f"  {word}: {x[i].tolist()}")

# 简单的 FFN
ffn = nn.Sequential(
    nn.Linear(4, 8, bias=False),   # 扩张
    nn.ReLU(),                      # 激活
    nn.Linear(8, 4, bias=False)    # 压缩
)

output = ffn(x)
print(f"\n输出（经过 FeedForward 后）:")
for i, word in enumerate(["我", "爱", "编程"]):
    print(f"  {word}: {output[i].tolist()}")

print(f"\n关键观察：")
print(f"  1. 每个词独立处理（没有词与词的交互）")
print(f"  2. 输入输出维度相同（4维 → 4维）")
print(f"  3. 但内容完全不同（经过了非线性变换）")

# ============================================================
# 5. 总结
# ============================================================
print("\n\n【5. 总结：FeedForward 的作用】")
print("=" * 60)
print("""
FeedForward 在做什么？
  ✅ 对每个词向量做非线性变换
  ✅ 通过高维空间增强表达能力
  ✅ 输入输出维度相同，但内容改变

为什么要"扩张-压缩"？
  ✅ 直接 768→768：只是线性变换，能力有限
  ✅ 768→2048→768：经过高维空间，能表达复杂函数

  类比：
    - 做菜：食材→切碎加工→装盘（维度没变但变熟了）
    - 照片：像素→特征提取→优化像素（质量提升了）

为什么每个词独立处理？
  ✅ Attention 已经做了"词与词交互"
  ✅ FeedForward 负责"深度思考"
  ✅ 分工明确：
     - Attention: 信息交换（开会讨论）
     - FeedForward: 独立处理（各自思考）

在 Transformer 中的位置：
  RMSNorm → Attention → 残差 → RMSNorm → FeedForward → 残差
            ↑ 信息交换              ↑ 独立思考

MiniMind 的 SwiGLU：
  - 更先进的激活函数
  - 门控机制（gate × up）
  - 性能比普通 FFN 更好
""")

print("=" * 60)
print("运行完成！")
print("=" * 60)
