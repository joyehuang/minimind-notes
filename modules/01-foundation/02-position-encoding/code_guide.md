# Position Encoding 代码导读

> 理解 MiniMind 中 RoPE 的真实实现

---

## 📂 代码位置

### 1. 预计算旋转频率

**文件**：`model/model_minimind.py`
**行数**：108-128

```python
def precompute_freqs_cis(dim: int, end: int, rope_base: float = 1e6, rope_scaling=None):
    """预计算 RoPE 的旋转频率"""

    # 计算频率：1 / (base^(2i/dim))
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 生成位置序列 [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)

    # YaRN 长度外推（可选）
    if rope_scaling is not None:
        t = t / rope_scaling

    # 计算每个位置的旋转角度：位置 * 频率
    freqs = torch.outer(t, freqs)  # [end, dim//2]

    # 转换为复数形式（cos + i*sin）
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [end, dim//2]

    return freqs_cis
```

---

### 2. 应用旋转编码

**文件**：`model/model_minimind.py`
**行数**：131-145

```python
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """将 RoPE 应用到 Query 和 Key"""

    # 将实数向量转为复数
    # [batch, seq, heads, head_dim] -> [batch, seq, heads, head_dim//2, 2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 调整 freqs_cis 形状以便广播
    freqs_cis = freqs_cis[:, None, :]  # [seq, 1, head_dim//2]

    # 复数乘法 = 旋转
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

---

### 3. 在 Attention 中使用

**文件**：`model/model_minimind.py`
**行数**：250-290

```python
class Attention(nn.Module):
    def forward(self, x, pos_ids, mask):
        batch, seq_len, _ = x.shape

        # 计算 Q, K, V
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # 重塑为多头形式
        xq = xq.view(batch, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # ⭐ 应用 RoPE（只对 Q 和 K）
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis[pos_ids])

        # 计算注意力分数
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ... 后续 softmax 和输出
```

---

## 🔍 逐步解析

### 频率计算公式

```python
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
```

**分解**：
1. `torch.arange(0, dim, 2)`：生成 [0, 2, 4, ..., dim-2]
2. `[: (dim // 2)]`：取前 dim/2 个（因为两两配对）
3. `/ dim`：归一化到 [0, 1)
4. `rope_base ** (...)`：指数运算
5. `1.0 / ...`：取倒数得到频率

**MiniMind 参数**（head_dim=64, rope_base=1e6）：
```
freqs[0]  = 1.0           # 高频：每 2π 个位置转一圈
freqs[15] = 0.001         # 中频：每 6283 个位置转一圈
freqs[31] = 0.000001      # 低频：每 628万 个位置转一圈
```

---

### 为什么用复数？

```python
# 实数向量 → 复数
xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))

# 复数乘法 = 旋转
xq_out = xq_ * freqs_cis
```

**原因**：复数乘法天然表示 2D 旋转

$$e^{i\theta} = \cos\theta + i\sin\theta$$

$$(a + bi) \times e^{i\theta} = (a\cos\theta - b\sin\theta) + i(a\sin\theta + b\cos\theta)$$

这正是旋转矩阵的效果！

**等价的矩阵形式**：
```python
# 这两种写法等价：
# 1. 复数乘法
result = (a + bi) * (cos_θ + i*sin_θ)

# 2. 矩阵乘法
result = [[cos_θ, -sin_θ],   @  [[a],
          [sin_θ,  cos_θ]]      [b]]
```

复数形式更简洁、更快。

---

### 两两配对的原理

```python
xq_ = xq.reshape(*xq.shape[:-1], -1, 2)
# [batch, seq, heads, head_dim] → [batch, seq, heads, head_dim//2, 2]
```

**为什么要配对？**
- 2D 旋转需要两个坐标
- 每两个维度组成一对，应用同一个旋转角度
- head_dim=64 → 32 对 → 32 个不同频率

**示意图**：
```
head_dim = 64 维

[x0, x1,  x2, x3,  ..., x62, x63]
  ↓   ↓    ↓   ↓         ↓    ↓
 pair0   pair1   ...   pair31

每对应用不同频率的旋转
```

---

## 💡 实现技巧

### 1. 预计算 freqs_cis

```python
# 在模型初始化时预计算
self.freqs_cis = precompute_freqs_cis(
    dim=self.head_dim,
    end=self.max_seq_len,
    rope_base=config.rope_theta
)
```

**优点**：
- 避免每次 forward 重复计算
- 支持任意位置索引（pos_ids）

---

### 2. 使用 torch.polar

```python
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
```

**等价于**：
```python
freqs_cis = torch.exp(1j * freqs)
# 或
freqs_cis = torch.cos(freqs) + 1j * torch.sin(freqs)
```

`torch.polar(r, θ)` 直接从极坐标创建复数，更高效。

---

### 3. 只对 Q 和 K 应用

```python
xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
# V 不需要位置编码！
```

**为什么 V 不需要？**
- Q 和 K 用于计算注意力分数（需要位置信息）
- V 是"被查询的内容"（不需要位置信息）
- 位置信息已经通过 Q·K 的点积融入了

---

### 4. 保持数据类型

```python
return xq_out.type_as(xq), xk_out.type_as(xk)
```

**为什么？**
- 复数运算需要 float32
- 但模型可能用 BF16/FP16
- 转回原始类型保持一致

---

## 📊 性能考虑

### 内存效率

```python
# 好：预计算并存储
self.register_buffer('freqs_cis', precompute_freqs_cis(...))

# 差：每次 forward 计算
freqs_cis = precompute_freqs_cis(...)  # 浪费计算
```

### 计算效率

复数乘法比矩阵乘法快：
- 矩阵：4 次乘法 + 2 次加法
- 复数：2 次乘法 + 2 次加法（利用 GPU 优化）

---

## 🔬 实验验证

### 验证相对位置性质

```python
# 位置 5 和 8
q5 = apply_rotary_emb(q, freqs_cis[5])
k8 = apply_rotary_emb(k, freqs_cis[8])
score_5_8 = q5 @ k8.T

# 位置 100 和 103（相对距离也是 3）
q100 = apply_rotary_emb(q, freqs_cis[100])
k103 = apply_rotary_emb(k, freqs_cis[103])
score_100_103 = q100 @ k103.T

# 两个分数应该相等（只依赖相对距离）
assert torch.allclose(score_5_8, score_100_103)
```

---

## 🔗 相关代码位置

1. **配置参数**：`model/model_minimind.py:30-65`
   - `rope_theta`：基础频率（默认 1e6）
   - `max_position_embeddings`：最大序列长度

2. **YaRN 支持**：`model/model_minimind.py:120-125`
   - `inference_rope_scaling`：长度外推系数

3. **完整 Attention**：`model/model_minimind.py:250-330`
   - 包含 GQA（Grouped Query Attention）

---

## 🎯 动手练习

### 练习 1：可视化旋转

修改 `exp2_multi_frequency.py`，绘制不同频率的旋转曲线：
```python
import matplotlib.pyplot as plt

freqs = precompute_freqs_cis(dim=64, end=100)
for i in [0, 15, 31]:
    plt.plot(freqs[:, i].real, label=f'freq_{i}')
plt.legend()
plt.show()
```

### 练习 2：验证相对位置

编写代码验证：位置 (5, 8) 和 (100, 103) 的注意力分数相等。

### 练习 3：对比绝对位置编码

实现一个简单的绝对位置编码，对比 RoPE 的长度外推能力。

---

## 📚 延伸阅读

- MiniMind 完整代码：`model/model_minimind.py`
- Llama 2 源码：[facebookresearch/llama](https://github.com/facebookresearch/llama)
- RoFormer 论文：[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
