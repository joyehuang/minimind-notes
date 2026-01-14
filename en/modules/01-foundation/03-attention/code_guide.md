---
title: Attention Code Walkthrough | MiniMind LLM Training
description: Understand the real Attention implementation in MiniMind. Learn QKV computation, multi-head attention, and scaled dot-product attention details.
keywords: Attention code, attention implementation, QKV source, multi-head attention code, Transformer attention source, LLM attention
---

# Attention Code Walkthrough

> Understand the real Attention implementation in MiniMind

---

## 📂 Code locations

### 1. Attention class

**File**: `model/model_minimind.py`
**Lines**: 250-330

```python
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads       # 8
        self.n_kv_heads = config.num_key_value_heads   # 2 (GQA)
        self.head_dim = config.hidden_size // self.n_heads  # 64
        self.n_rep = self.n_heads // self.n_kv_heads   # 4

        # QKV projections
        self.wq = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x, pos_ids, mask):
        batch, seq_len, _ = x.shape

        # 1. compute Q, K, V
        xq = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # 2. apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis[pos_ids])

        # 3. GQA: expand KV heads to match Q heads
        xk = repeat_kv(xk, self.n_rep)  # [batch, seq, n_heads, head_dim]
        xv = repeat_kv(xv, self.n_rep)

        # 4. transpose for matmul
        xq = xq.transpose(1, 2)  # [batch, n_heads, seq, head_dim]
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 5. attention scores
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 6. causal mask
        if mask is not None:
            scores = scores + mask

        # 7. softmax
        attn_weights = F.softmax(scores, dim=-1)

        # 8. weighted sum
        output = torch.matmul(attn_weights, xv)

        # 9. merge heads + output projection
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(output)
```

---

### 2. GQA: repeat_kv function

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q heads"""
    if n_rep == 1:
        return x

    batch, seq_len, n_kv_heads, head_dim = x.shape

    # [batch, seq, n_kv_heads, 1, head_dim]
    x = x[:, :, :, None, :]

    # expand and reshape
    x = x.expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
    return x.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
```

**Effect**:
- Input: `[batch, seq, 2, 64]` (2 KV heads)
- n_rep = 4
- Output: `[batch, seq, 8, 64]` (match Q heads)

---

## 🔍 Key implementation details

### 1. Scaling factor

```python
scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
```

**Why divide by $\sqrt{d_k}$?**
- dot product variance grows with d_k
- large variance → softmax saturates
- divide by $\sqrt{d_k}$ to keep variance ~1

---

### 2. Causal mask

```python
if mask is not None:
    scores = scores + mask
```

**Mask values**:
- 0: allowed
- $-\infty$: disallowed (softmax → 0)

**How it’s built**:
```python
mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
```

---

### 3. Flash Attention (optional)

```python
if self.flash_attn:
    output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask)
else:
    # manual implementation
    scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
    ...
```

**Flash Attention advantages**:
- better memory efficiency (no full attention matrix)
- faster (fused ops)

---

## 💡 Implementation tips

### 1. Shape transformation order

```python
# input: [batch, seq, hidden]
xq = self.wq(x)                    # [batch, seq, n_heads * head_dim]
xq = xq.view(batch, seq, n_heads, head_dim)  # split heads
xq = xq.transpose(1, 2)            # [batch, n_heads, seq, head_dim]
```

**Why transpose?**
- matmul needs `[..., seq, dim] @ [..., dim, seq]`
- transpose aligns dimensions

---

### 2. Why contiguous() matters

```python
output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
```

**Why call contiguous()?**
- `transpose` changes the view, not the underlying memory layout
- `view` requires contiguous memory
- `contiguous()` rearranges memory so view works

---

## 🎯 Hands-on exercises

### Exercise 1: visualize attention weights

```python
# save attention weights
attn_weights = F.softmax(scores, dim=-1)
# plot heatmap
plt.imshow(attn_weights[0, 0].detach().numpy())
```

### Exercise 2: remove scaling factor

Remove `/ math.sqrt(self.head_dim)` and observe changes in softmax output.

### Exercise 3: implement KV cache

Cache previous K, V during inference to avoid recomputation.

---

## 📚 Further reading

- MiniMind full code: `model/model_minimind.py`
- Flash Attention paper: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- PyTorch SDPA: [scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
