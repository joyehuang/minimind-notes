# FeedForward 代码导读

> 理解 MiniMind 中 FeedForward 的真实实现

---

## 📂 代码位置

### 1. FeedForward 类

**文件**：`model/model_minimind.py`
**行数**：330-380

```python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()

        hidden_dim = config.hidden_size
        intermediate_dim = config.intermediate_size

        # SwiGLU: 三个投影矩阵
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU 公式
        # output = down(SiLU(gate(x)) * up(x))
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )
```

---

### 2. 在 TransformerBlock 中的使用

**文件**：`model/model_minimind.py`
**行数**：400-450

```python
class TransformerBlock(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, pos_ids, mask):
        # Attention + 残差
        h = x + self.attention(self.attention_norm(x), pos_ids, mask)

        # FeedForward + 残差
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
```

---

## 🔍 逐步解析

### SwiGLU 的三个投影

```python
# 输入 x: [batch, seq, hidden_dim]

# 1. 计算门控信号
gate = self.gate_proj(x)  # [batch, seq, intermediate_dim]

# 2. 计算值信号
up = self.up_proj(x)      # [batch, seq, intermediate_dim]

# 3. SiLU 激活 + 门控
hidden = F.silu(gate) * up  # [batch, seq, intermediate_dim]

# 4. 压缩回原维度
output = self.down_proj(hidden)  # [batch, seq, hidden_dim]
```

**维度变化**（MiniMind 512 配置）：
```
输入:  [batch, seq, 512]
gate:  [batch, seq, 2048]  (扩张)
up:    [batch, seq, 2048]  (扩张)
hidden: [batch, seq, 2048]  (gate × up)
输出:  [batch, seq, 512]   (压缩)
```

---

### SiLU 激活函数

```python
# F.silu(x) = x * torch.sigmoid(x)

x = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float)
silu = F.silu(x)
# tensor([-0.2384, -0.2689,  0.0000,  0.7311,  1.7616])

# 对比 ReLU
relu = F.relu(x)
# tensor([0., 0., 0., 1., 2.])
```

**特点**：
- 平滑：处处可导，梯度稳定
- 非单调：负数部分不完全为 0
- 自门控：$x \cdot \sigma(x)$

---

### 为什么用三个投影而不是两个？

**标准 FFN（两个投影）**：
```python
hidden = ReLU(W1(x))  # 768 → 2048
output = W2(hidden)   # 2048 → 768
```

**SwiGLU（三个投影）**：
```python
gate = SiLU(W_gate(x))  # 768 → 2048
up = W_up(x)            # 768 → 2048
hidden = gate * up      # 逐元素相乘
output = W_down(hidden) # 2048 → 768
```

**优势**：
1. 门控机制：动态控制信息流
2. 更强表达能力：两条路径提供不同视角
3. 实验效果更好：在各种 LLM 基准上表现更优

**参数量对比**：
- 标准 FFN：2 × d × 4d = 8d²
- SwiGLU：3 × d × (8d/3) = 8d²（调整 intermediate）

---

### 门控机制详解

```python
gate = F.silu(self.gate_proj(x))  # 门控信号：决定"开关程度"
up = self.up_proj(x)              # 值信号：实际内容
hidden = gate * up                # 逐元素相乘

# gate 的作用：
# - gate ≈ 0：关闭，up 的信息被抑制
# - gate ≈ 1：打开，up 的信息完全通过
# - 0 < gate < 1：部分通过
```

**直觉**：
- gate 像一个"音量旋钮"
- 不同维度有不同的"音量"
- 模型学习哪些信息应该放大/抑制

---

## 💡 实现技巧

### 1. 无偏置（bias=False）

```python
self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
```

**为什么不用偏置？**
- 大模型中偏置效果不明显
- 减少参数量
- 与 RMSNorm 配合更好（已经有归一化）

---

### 2. intermediate_size 的选择

```python
# MiniMind 配置
hidden_size = 512
intermediate_size = 2048  # 4x 扩张

# 如果用 SwiGLU，有些实现会调整：
# intermediate_size = int(hidden_size * 4 * 2 / 3)
# 以保持总参数量与标准 FFN 相同
```

**Llama 的做法**：
- intermediate_size = 2.7 × hidden_size（调整后）
- 或直接用 4x 但接受更多参数

---

### 3. 融合操作

```python
# 朴素实现
gate = self.gate_proj(x)
up = self.up_proj(x)
hidden = F.silu(gate) * up

# 实际可以融合 gate_proj 和 up_proj
# 减少内存读写，提高效率
gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
gate, up = gate_up.chunk(2, dim=-1)
hidden = F.silu(gate) * up
```

---

## 📊 性能考虑

### 计算量分析

```python
# FeedForward 的 FLOPs
# 假设 batch=1, seq=512, hidden=512, intermediate=2048

# gate_proj: 512 × 512 × 2048 = 536M FLOPs
# up_proj:   512 × 512 × 2048 = 536M FLOPs
# down_proj: 512 × 2048 × 512 = 536M FLOPs
# 元素乘法:  512 × 2048 ≈ 1M FLOPs

# 总计: ≈ 1.6G FLOPs per block
```

**对比 Attention**：
- Attention: ≈ 1G FLOPs（seq=512）
- FeedForward: ≈ 1.6G FLOPs
- FeedForward 占主导（约 60%）

---

### 内存使用

```python
# 中间激活内存
# gate: batch × seq × intermediate = batch × 512 × 2048 floats
# up:   batch × seq × intermediate = batch × 512 × 2048 floats

# 总激活内存 ≈ 2 × batch × 512 × 2048 × 4 bytes
#            = batch × 8 MB
```

**优化技巧**：
- 使用 checkpointing：不保存中间激活，重新计算
- 混合精度：用 BF16/FP16

---

## 🔬 实验验证

### 验证维度变化

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFeedForward(nn.Module):
    def __init__(self, dim=512, intermediate=2048):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        print(f"gate: {gate.shape}")

        up = self.up_proj(x)
        print(f"up: {up.shape}")

        hidden = F.silu(gate) * up
        print(f"hidden: {hidden.shape}")

        output = self.down_proj(hidden)
        print(f"output: {output.shape}")

        return output

# 测试
ffn = SimpleFeedForward()
x = torch.randn(2, 10, 512)  # [batch=2, seq=10, dim=512]
print(f"input: {x.shape}")
output = ffn(x)
```

### 验证门控效果

```python
# 可视化门控信号
import matplotlib.pyplot as plt

x = torch.randn(1, 5, 512)  # 5 个 token
gate = F.silu(ffn.gate_proj(x))  # [1, 5, 2048]

# 查看不同 token 的门控激活
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.hist(gate[0, i].detach().numpy(), bins=50)
    plt.title(f"Token {i}")
plt.suptitle("Gate Activations")
plt.show()
```

---

## 🔗 相关代码位置

1. **配置参数**：`model/model_minimind.py:30-65`
   - `intermediate_size`：中间维度
   - `hidden_size`：隐藏维度

2. **MoE FeedForward**：`model/model_minimind.py:380-450`
   - 专家混合版本
   - 每个专家是一个 FeedForward

3. **完整 TransformerBlock**：`model/model_minimind.py:450-500`
   - Attention + FFN 的组合

---

## 🎯 动手练习

### 练习 1：对比激活函数

实现不同激活函数的 FFN，对比输出分布：
```python
def ffn_relu(x):
    return W2(F.relu(W1(x)))

def ffn_gelu(x):
    return W2(F.gelu(W1(x)))

def ffn_swiglu(x):
    return W_down(F.silu(W_gate(x)) * W_up(x))
```

### 练习 2：可视化门控

修改代码，保存并可视化门控信号：
```python
# 在 forward 中保存
self.last_gate = F.silu(self.gate_proj(x))

# 绘制热力图
plt.imshow(model.ffn.last_gate[0].detach().numpy())
```

### 练习 3：计算实际 FLOPs

编写代码计算 FeedForward 的实际计算量：
```python
from thop import profile
flops, params = profile(ffn, inputs=(x,))
print(f"FLOPs: {flops/1e6:.2f}M, Params: {params/1e6:.2f}M")
```

---

## 📚 延伸阅读

- MiniMind 完整代码：`model/model_minimind.py`
- Llama 2 源码：[facebookresearch/llama](https://github.com/facebookresearch/llama)
- GLU 论文：[arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
