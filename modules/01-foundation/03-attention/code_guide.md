# Attention 代码导读

> 理解 MiniMind 中 Attention 的真实实现

---

## 📂 代码位置

### 1. Attention 类

**文件**：`model/model_minimind.py`
**行数**：250-330

```python
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads       # 8
        self.n_kv_heads = config.num_key_value_heads   # 2 (GQA)
        self.head_dim = config.hidden_size // self.n_heads  # 64
        self.n_rep = self.n_heads // self.n_kv_heads   # 4

        # QKV 投影
        self.wq = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x, pos_ids, mask):
        batch, seq_len, _ = x.shape

        # 1. 计算 Q, K, V
        xq = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # 2. 应用 RoPE
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis[pos_ids])

        # 3. GQA：扩展 KV 以匹配 Q 的头数
        xk = repeat_kv(xk, self.n_rep)  # [batch, seq, n_heads, head_dim]
        xv = repeat_kv(xv, self.n_rep)

        # 4. 转置以便矩阵乘法
        xq = xq.transpose(1, 2)  # [batch, n_heads, seq, head_dim]
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 5. 计算注意力分数
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 6. 应用因果掩码
        if mask is not None:
            scores = scores + mask

        # 7. Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # 8. 加权求和
        output = torch.matmul(attn_weights, xv)

        # 9. 合并头 + 输出投影
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(output)
```

---

### 2. GQA：repeat_kv 函数

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 KV 头扩展以匹配 Q 头数"""
    if n_rep == 1:
        return x

    batch, seq_len, n_kv_heads, head_dim = x.shape

    # [batch, seq, n_kv_heads, 1, head_dim]
    x = x[:, :, :, None, :]

    # 扩展并重塑
    x = x.expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
    return x.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
```

**效果**：
- 输入：`[batch, seq, 2, 64]`（2 个 KV 头）
- n_rep = 4
- 输出：`[batch, seq, 8, 64]`（8 个头，匹配 Q）

---

## 🔍 关键实现细节

### 1. 缩放因子

```python
scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
```

**为什么除以 $\sqrt{d_k}$？**
- 点积的期望方差 = d_k
- 大方差 → softmax 梯度消失
- 除以 $\sqrt{d_k}$ 使方差 = 1

---

### 2. 因果掩码

```python
if mask is not None:
    scores = scores + mask
```

**mask 的值**：
- 0：允许注意
- $-\infty$：禁止注意（softmax 后 = 0）

**生成方式**：
```python
mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
```

---

### 3. Flash Attention（可选）

```python
if self.flash_attn:
    output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask)
else:
    # 手动实现
    scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
    ...
```

**Flash Attention 优势**：
- 内存效率更高（不存储完整 attention 矩阵）
- 速度更快（融合操作）

---

## 💡 实现技巧

### 1. 形状变换顺序

```python
# 输入：[batch, seq, hidden]
xq = self.wq(x)                    # [batch, seq, n_heads * head_dim]
xq = xq.view(batch, seq, n_heads, head_dim)  # 分头
xq = xq.transpose(1, 2)            # [batch, n_heads, seq, head_dim]
```

**为什么要 transpose？**
- 矩阵乘法需要 `[..., seq, dim] @ [..., dim, seq]`
- transpose 后形状匹配

---

### 2. contiguous() 的必要性

```python
output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
```

**为什么需要 contiguous？**
- `transpose` 不改变内存布局，只改变视图
- `view` 需要连续内存
- `contiguous()` 重新排列内存

---

## 🎯 动手练习

### 练习 1：可视化注意力权重

```python
# 保存 attention weights
attn_weights = F.softmax(scores, dim=-1)
# 绘制热力图
plt.imshow(attn_weights[0, 0].detach().numpy())
```

### 练习 2：移除缩放因子

修改代码，移除 `/math.sqrt(self.head_dim)`，观察 softmax 输出的变化。

### 练习 3：实现 KV Cache

在推理时缓存之前的 K、V，避免重复计算。

---

## 📚 延伸阅读

- MiniMind 完整代码：`model/model_minimind.py`
- Flash Attention 论文：[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- PyTorch SDPA：[scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
