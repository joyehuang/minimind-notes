# Attention 自测题

> 完成以下题目检验你的理解程度

---

## 📝 选择题

### Q1: Self-Attention 中为什么要除以 $\sqrt{d_k}$？

A. 为了加速计算
B. 为了减少参数量
C. 为了防止 softmax 饱和，稳定梯度
D. 为了支持更长的序列

<details>
<summary>点击查看答案</summary>

**答案：C**

**解析**：
- 点积的方差随 $d_k$ 增大而增大
- 当 $d_k$ 很大时，点积值会很大
- 大数值输入 softmax 会导致梯度接近 0（饱和）
- 除以 $\sqrt{d_k}$ 使点积的方差稳定在 1 左右

**例子**（d_k=64）：
- 未缩放：分数可能达到 64（softmax 几乎全是 0 和 1）
- 缩放后：分数约 8（softmax 分布更平滑）

</details>

---

### Q2: Q、K、V 三个矩阵分别代表什么含义？

A. Query=查询、Key=钥匙、Value=价值
B. Query=问题、Key=关键词、Value=答案
C. Query=我想找什么、Key=我有什么标签、Value=我的实际内容
D. Query=输入、Key=输出、Value=中间结果

<details>
<summary>点击查看答案</summary>

**答案：C**

**解析**：

**图书馆类比**：
- **Query（查询）**：你想找什么书？（搜索关键词）
- **Key（索引）**：每本书的关键词标签
- **Value（内容）**：书的实际内容

**实际作用**：
- Q：当前 token 作为"查询者"时关注什么
- K：当前 token 作为"被查询者"时展示什么
- V：当前 token 实际要传递的信息

**为什么需要三个不同的投影？**
- 让模型学习从不同角度看待同一个 token
- Q 和 K 用于计算"相关性"
- V 是实际被提取的"内容"

</details>

---

### Q3: Multi-Head Attention 的主要优势是什么？

A. 减少计算量
B. 减少参数量
C. 让模型学习多种不同的关系模式
D. 支持更长的序列

<details>
<summary>点击查看答案</summary>

**答案：C**

**解析**：

**单头的局限**：
- 只能学习一种"关注模式"
- 例如只能关注语法关系，无法同时关注语义关系

**多头的优势**：
- 不同头学习不同的模式：
  - Head 1：语法关系（主谓宾）
  - Head 2：语义关系（同义词）
  - Head 3：位置关系（相邻词）
  - Head 4：代词指代关系
  - ...
- 最后拼接起来，融合多种信息

**参数量分析**：
- 单头：hidden_size → hidden_size（一个大投影）
- 多头：hidden_size → n_heads × head_dim（等价，但分头处理）
- 总参数量相同，但表达能力更强

</details>

---

### Q4: GQA（Grouped Query Attention）的作用是什么？

A. 提高模型精度
B. 减少 KV Cache 内存，加速推理
C. 增加模型容量
D. 支持更多的注意力头

<details>
<summary>点击查看答案</summary>

**答案：B**

**解析**：

**MHA 的问题**：
- 每个头有独立的 K、V
- KV Cache 大小 = n_heads × seq_len × head_dim × 2
- 长序列推理时内存占用很大

**GQA 的解决**：
- 多个 Q 头共享一组 K、V
- 例如：8 个 Q 头，2 个 KV 头 → 每 4 个 Q 共享一组 KV
- KV Cache 减少 75%

**MiniMind 配置**：
```python
n_heads = 8        # Q 头数
n_kv_heads = 2     # KV 头数
n_rep = 4          # 每组重复次数
```

**注意**：GQA 在精度上略有损失，但推理速度和内存效率大幅提升

</details>

---

### Q5: 因果掩码（Causal Mask）的作用是什么？

A. 防止模型过拟合
B. 加速训练
C. 让模型只能看到"过去"的 token，不能看到"未来"
D. 减少内存使用

<details>
<summary>点击查看答案</summary>

**答案：C**

**解析**：

**语言模型的约束**：
- 生成式模型需要"自回归"：根据前面的词预测下一个词
- 训练时不能"作弊"看到未来的词

**掩码的实现**：
```python
mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
```

**效果**：
- 上三角为 $-\infty$
- softmax($-\infty$) = 0 → 完全忽略未来位置
- 下三角（包括对角线）正常计算

**示例**（seq_len=4）：
```
[[0, -∞, -∞, -∞],
 [0,  0, -∞, -∞],
 [0,  0,  0, -∞],
 [0,  0,  0,  0]]
```
- 位置 0 只能看自己
- 位置 3 可以看 0、1、2、3

</details>

---

### Q6: repeat_kv 函数在 GQA 中的作用是什么？

A. 减少 KV 头数
B. 将 KV 头扩展以匹配 Q 头数
C. 压缩 KV 信息
D. 对 KV 进行归一化

<details>
<summary>点击查看答案</summary>

**答案：B**

**解析**：

**GQA 的挑战**：
- Q 有 8 个头，KV 只有 2 个头
- 计算 Q·K 时维度不匹配

**repeat_kv 的解决**：
```python
def repeat_kv(x, n_rep):
    # x: [batch, seq, n_kv_heads, head_dim]
    # n_rep = n_heads // n_kv_heads = 4

    # 扩展并重塑
    x = x[:, :, :, None, :]  # 添加维度
    x = x.expand(..., n_rep, ...)  # 复制 4 次
    return x.reshape(..., n_kv_heads * n_rep, ...)  # 合并
```

**效果**：
- 输入：[batch, seq, 2, 64]
- 输出：[batch, seq, 8, 64]
- 每个 KV 头被复制 4 次

**为什么不直接训练 8 个 KV 头？**
- 推理时 KV Cache 更小（只存 2 个头）
- 训练时也节省内存
- 精度损失很小

</details>

---

### Q7: Flash Attention 相比标准实现的优势是什么？

A. 更高的精度
B. 更少的参数
C. 更高的内存效率和速度
D. 更好的泛化能力

<details>
<summary>点击查看答案</summary>

**答案：C**

**解析**：

**标准实现的问题**：
- 需要存储完整的 attention 矩阵：[batch, heads, seq, seq]
- 序列长度 4096 → 矩阵大小 16M 个元素
- 内存是 $O(N^2)$

**Flash Attention 的优化**：
- 分块计算，不存储完整矩阵
- 融合多个操作（QK、softmax、×V）
- 内存是 $O(N)$

**MiniMind 代码**：
```python
if self.flash_attn:
    output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask)
else:
    # 手动实现（用于学习）
    scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
    ...
```

**实际加速**：长序列下可快 2-4 倍

</details>

---

## 🎯 综合问答题

### Q8: 实战问题

假设你在调试一个 Attention 模块，发现所有 token 的注意力权重几乎均匀分布（每个位置都是 ~1/seq_len），这可能是什么问题？如何解决？

<details>
<summary>点击查看参考答案</summary>

**可能的原因**：

1. **Q 和 K 没有正确初始化**：
   - 投影矩阵初始值太小
   - 导致 Q·K 分数接近 0
   - softmax(0, 0, ..., 0) ≈ 均匀分布

2. **缩放因子问题**：
   - 除以了过大的值
   - 或忘记开根号（除以 d_k 而不是 √d_k）

3. **head_dim 设置错误**：
   - head_dim 过大导致点积方差过大
   - 但这通常会导致极端分布，不是均匀分布

4. **没有学习到有意义的模式**：
   - 训练数据问题
   - 模型容量不足

**诊断方法**：

```python
# 检查 Q·K 分数（softmax 之前）
scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(head_dim)
print(f"scores mean: {scores.mean()}, std: {scores.std()}")

# 正常情况：mean ≈ 0, std ≈ 1
# 问题情况：std 太小（接近 0）
```

**解决方案**：

1. **检查投影矩阵初始化**：
   ```python
   # 使用 Xavier 或 Kaiming 初始化
   nn.init.xavier_uniform_(self.wq.weight)
   ```

2. **验证缩放因子**：
   ```python
   # 确保是 sqrt(head_dim)，不是 head_dim
   scale = math.sqrt(self.head_dim)
   ```

3. **可视化注意力**：
   ```python
   plt.imshow(attn_weights[0, 0].detach().numpy())
   plt.title("Attention weights")
   plt.colorbar()
   ```

</details>

---

### Q9: 概念理解

为什么 Self-Attention 中 Q、K、V 都来自同一个输入 x，但还需要三个不同的投影矩阵？直接用 x 做 Q、K、V 不行吗？

<details>
<summary>点击查看参考答案</summary>

**直接用 x 的问题**：

如果 Q = K = V = x，则：
```python
scores = x @ x.T  # 自己和自己的点积
```

这相当于计算每个 token 与其他 token 的"余弦相似度"（内积）。

**问题**：
1. **对称性**：token_i 对 token_j 的注意力 = token_j 对 token_i 的注意力
   - 但语言中关系往往是不对称的
   - "猫吃鱼"：猫应该注意鱼，但鱼不一定要注意猫

2. **表达能力有限**：
   - 只能表达"相似度"这一种关系
   - 无法学习"主谓关系"、"修饰关系"等

**三个投影的意义**：

```python
Q = x @ W_Q  # "作为查询者，我关注什么特征？"
K = x @ W_K  # "作为被查询者，我展示什么特征？"
V = x @ W_V  # "我实际要传递什么内容？"
```

**优势**：
1. **非对称性**：Q 和 K 不同，允许非对称关系
2. **角色分离**：查询角度、被查询角度、内容传递可以不同
3. **表达能力**：可以学习任意复杂的关系模式

**类比**：
- 图书馆场景：
  - 你的问题（Q）：用自然语言描述需求
  - 书的索引（K）：用关键词标签
  - 书的内容（V）：实际文字
- 三者用不同的"语言"，通过匹配找到正确的内容

</details>

---

### Q10: 代码理解

解释以下代码中 `contiguous()` 的必要性：

```python
output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
```

<details>
<summary>点击查看参考答案</summary>

**背景**：

PyTorch tensor 有两个概念：
1. **存储（Storage）**：实际的内存布局
2. **视图（View）**：如何解释这块内存

**transpose 的行为**：

```python
# 假设 output 形状是 [batch, n_heads, seq, head_dim]
# 内存布局是按这个顺序排列的

output = output.transpose(1, 2)
# 现在形状是 [batch, seq, n_heads, head_dim]
# 但内存布局没变！只是改变了"视图"
```

**问题**：

`view()` 要求 tensor 在内存中是连续的，但 `transpose` 后内存不连续：

```python
# 原始内存顺序（简化示例）：
# [head0_pos0, head0_pos1, head1_pos0, head1_pos1, ...]

# transpose 后逻辑顺序：
# [head0_pos0, head1_pos0, head0_pos1, head1_pos1, ...]

# 内存不连续 → view 会报错
```

**contiguous() 的作用**：

```python
output = output.transpose(1, 2).contiguous()
# 1. 重新分配内存
# 2. 按新的逻辑顺序排列数据
# 3. 现在可以安全使用 view
```

**性能考虑**：

- `contiguous()` 需要拷贝内存，有开销
- 但这是必要的开销
- 替代方案：`reshape()` 会自动处理，但不够显式

**最佳实践**：

```python
# 明确知道需要连续内存时，显式调用
output = output.transpose(1, 2).contiguous().view(...)

# 或使用 reshape（隐式处理）
output = output.transpose(1, 2).reshape(...)
```

</details>

---

## ✅ 完成检查

完成所有题目后，检查你是否达到：

- [ ] **Q1-Q7 全对**：基础知识扎实
- [ ] **Q8 能提出 2+ 诊断方法**：具备调试能力
- [ ] **Q9 能解释投影矩阵的意义**：深刻理解设计原则
- [ ] **Q10 能解释 contiguous 的必要性**：理解 PyTorch 内存模型

如果还有不清楚的地方，回到 [teaching.md](./teaching.md) 复习，或重新运行实验代码。

---

## 🎓 进阶挑战

想要更深入理解？尝试：

1. **修改实验代码**：
   - 实现一个没有缩放因子的 Attention，观察 softmax 输出
   - 实现 MQA（Multi-Query Attention），对比 GQA
   - 可视化不同头学到的注意力模式

2. **阅读论文**：
   - [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
   - [GQA Paper](https://arxiv.org/abs/2305.13245) - Grouped Query Attention
   - [Flash Attention](https://arxiv.org/abs/2205.14135) - 高效 Attention 实现

3. **实现变体**：
   - 实现 Cross-Attention（Q 和 KV 来自不同输入）
   - 实现 Sliding Window Attention
   - 实现 Sparse Attention

---

**下一步**：前往 [04. FeedForward](../04-feedforward) 学习前馈网络！
