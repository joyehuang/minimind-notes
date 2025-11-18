# FeedForward 自测题

> 完成以下题目检验你的理解程度

---

## 📝 选择题

### Q1: FeedForward 为什么要"扩张-压缩"？

A. 为了减少计算量
B. 为了减少参数量
C. 为了在高维空间中增强表达能力
D. 为了加速推理

<details>
<summary>点击查看答案</summary>

**答案：C**

**解析**：
- 直接 768 → 768：只是线性变换，表达能力有限
- 扩张到高维空间：更容易分离不同的模式
- 激活函数创造非线性边界
- 压缩回来时保留了判别性特征

**类比**：
- 在 2D 空间中，一条直线无法分开环形分布的点
- 映射到 3D 空间后，一个平面就可以分开
- 然后投影回 2D，结果是非线性的分割

</details>

---

### Q2: SwiGLU 使用几个投影矩阵？

A. 1 个
B. 2 个
C. 3 个
D. 4 个

<details>
<summary>点击查看答案</summary>

**答案：C**

**解析**：

SwiGLU 使用三个投影：
1. `gate_proj`：计算门控信号
2. `up_proj`：计算值信号
3. `down_proj`：压缩回原维度

**公式**：
$$\text{SwiGLU}(x) = W_{\text{down}} \cdot (\text{SiLU}(W_{\text{gate}} \cdot x) \odot W_{\text{up}} \cdot x)$$

**对比标准 FFN**（只有 2 个投影）：
$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x)$$

</details>

---

### Q3: SiLU 激活函数的公式是什么？

A. $\max(0, x)$
B. $x \cdot \sigma(x)$
C. $x \cdot \tanh(x)$
D. $\text{sign}(x) \cdot \max(0, |x|)$

<details>
<summary>点击查看答案</summary>

**答案：B**

**解析**：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**特点**：
- 也叫 Swish 激活函数
- 平滑：处处可导
- 自门控：输入乘以自己的 sigmoid
- 非单调：负数部分不完全为 0

**A 是 ReLU**
**C 是 Mish**（另一种激活函数）

</details>

---

### Q4: SwiGLU 中门控机制的作用是什么？

A. 加速计算
B. 减少参数
C. 动态控制信息流通
D. 增加非线性层数

<details>
<summary>点击查看答案</summary>

**答案：C**

**解析**：

```python
gate = SiLU(gate_proj(x))  # 门控信号
up = up_proj(x)            # 值信号
hidden = gate * up         # 逐元素相乘
```

**门控的作用**：
- gate ≈ 0：关闭，up 的信息被抑制
- gate ≈ 1：打开，up 的信息完全通过
- 0 < gate < 1：部分通过

**直觉**：
- 像"音量旋钮"控制每个维度
- 模型学习哪些信息应该放大/抑制
- 比单纯的激活函数更灵活

</details>

---

### Q5: FeedForward 与 Attention 的主要区别是什么？

A. FeedForward 更快
B. Attention 处理词间关系，FeedForward 独立处理每个位置
C. FeedForward 参数更少
D. Attention 使用激活函数，FeedForward 不使用

<details>
<summary>点击查看答案</summary>

**答案：B**

**解析**：

| 组件 | 操作 | 类比 |
|------|------|------|
| **Attention** | 有 seq × seq 的交互 | 开会讨论 |
| **FeedForward** | 每个位置独立处理 | 各自思考 |

**分工**：
- Attention：信息融合，让词与词交换信息
- FeedForward：特征变换，深度处理每个词

**计算特点**：
- Attention：计算复杂度 O(n²d)
- FeedForward：计算复杂度 O(nd²)

</details>

---

### Q6: 为什么 SwiGLU 通常比 ReLU 效果更好？

A. 计算更快
B. 参数更少
C. 梯度更平滑，保留部分负数信息
D. 内存使用更少

<details>
<summary>点击查看答案</summary>

**答案：C**

**解析**：

**ReLU 的问题**：
```python
ReLU(x) = max(0, x)
# x = -1 → 0 (完全丢失)
# 梯度：x < 0 时梯度为 0 (死神经元)
```

**SiLU 的优势**：
```python
SiLU(x) = x * sigmoid(x)
# x = -1 → -0.27 (保留部分信息)
# 梯度：处处非零，平滑
```

**实验结果**：
- GLU 系列在 LLM 基准上表现更好
- 特别是长序列任务
- 门控机制提供额外的表达能力

</details>

---

### Q7: MiniMind 中 intermediate_size 通常是 hidden_size 的几倍？

A. 2 倍
B. 3 倍
C. 4 倍
D. 8 倍

<details>
<summary>点击查看答案</summary>

**答案：C**

**解析**：

```python
# MiniMind 配置
hidden_size = 512
intermediate_size = 2048  # 4x 扩张
```

**原因**：
- 4x 扩张是 Transformer 的标准做法
- 提供足够的中间空间
- 参数量和表达能力的平衡

**注意**：
- 如果用 SwiGLU（3 个投影），有些实现会调整：
  - intermediate_size = hidden_size × 4 × 2/3
  - 以保持总参数量与标准 FFN（2 个投影）相同

</details>

---

## 🎯 综合问答题

### Q8: 实战问题

如果你发现 FeedForward 的输出总是接近 0，可能是什么问题？如何调试？

<details>
<summary>点击查看参考答案</summary>

**可能的原因**：

1. **权重初始化问题**：
   - 权重初始化太小
   - 导致输出值太小

2. **门控信号问题**：
   - gate_proj 的输出总是负数
   - SiLU(负数) ≈ 0
   - 导致 gate * up ≈ 0

3. **梯度消失**：
   - 深层网络中梯度传不回来
   - 权重没有更新

4. **数值精度问题**：
   - 使用了太低的精度（如 FP16）
   - 小数值被截断为 0

**调试方法**：

```python
# 1. 检查中间值
gate = self.gate_proj(x)
print(f"gate mean: {gate.mean()}, std: {gate.std()}")

silu_gate = F.silu(gate)
print(f"silu_gate mean: {silu_gate.mean()}, std: {silu_gate.std()}")

up = self.up_proj(x)
print(f"up mean: {up.mean()}, std: {up.std()}")

hidden = silu_gate * up
print(f"hidden mean: {hidden.mean()}, std: {hidden.std()}")

# 2. 检查权重初始化
for name, param in self.named_parameters():
    print(f"{name}: mean={param.mean():.6f}, std={param.std():.6f}")

# 3. 检查梯度
output.sum().backward()
for name, param in self.named_parameters():
    if param.grad is not None:
        print(f"{name} grad: {param.grad.abs().mean():.6f}")
```

**解决方案**：

1. **调整初始化**：
   ```python
   nn.init.xavier_uniform_(self.gate_proj.weight)
   ```

2. **检查 RMSNorm**：
   - 确保输入已经归一化
   - 避免数值范围过大或过小

3. **使用混合精度**：
   - 关键计算用 FP32
   - 其他部分用 BF16

</details>

---

### Q9: 概念理解

为什么 FeedForward 要在每个位置独立处理，而不是像 Attention 一样进行全局交互？

<details>
<summary>点击查看参考答案</summary>

**设计考虑**：

1. **分工明确**：
   - Attention 已经做了"信息融合"
   - FeedForward 负责"深度处理"
   - 避免重复功能

2. **计算效率**：
   - 全局交互：O(n²d)
   - 独立处理：O(nd²)
   - 当 n > d 时，独立处理更高效

3. **参数效率**：
   - 全局交互需要位置相关的参数
   - 独立处理的参数可以复用

4. **理论基础**：
   - Transformer 的设计思想：
     - Attention = 全局信息路由
     - FFN = 局部特征变换
   - 类似 CNN 中的空间卷积 + 1x1 卷积

**类比**：
- 开会（Attention）：大家交换信息
- 思考（FFN）：各自消化、整理
- 两者交替进行，效果最好

**实验验证**：
- 论文证明这种分工设计效果很好
- 混合设计（如全 FFN 或全 Attention）效果更差

</details>

---

### Q10: 代码理解

解释以下代码的每一步：

```python
def forward(self, x):
    return self.down_proj(
        F.silu(self.gate_proj(x)) * self.up_proj(x)
    )
```

<details>
<summary>点击查看参考答案</summary>

**逐步解析**：

```python
def forward(self, x):
    # x: [batch, seq, hidden_dim]
    # 例如: [32, 512, 512]

    # Step 1: 计算门控信号
    gate = self.gate_proj(x)
    # gate: [batch, seq, intermediate_dim]
    # 例如: [32, 512, 2048]

    # Step 2: SiLU 激活
    gate_activated = F.silu(gate)
    # SiLU(x) = x * sigmoid(x)
    # 平滑激活，保留部分负数信息
    # gate_activated: [32, 512, 2048]

    # Step 3: 计算值信号
    up = self.up_proj(x)
    # up: [32, 512, 2048]

    # Step 4: 门控相乘
    hidden = gate_activated * up
    # 逐元素相乘
    # gate_activated 作为"开关"控制 up 的信息
    # hidden: [32, 512, 2048]

    # Step 5: 压缩回原维度
    output = self.down_proj(hidden)
    # output: [32, 512, 512]

    return output
```

**关键点**：

1. **两条并行路径**：gate_proj 和 up_proj 独立计算
2. **门控机制**：SiLU(gate) 控制 up 的信息流
3. **维度变化**：512 → 2048 → 512（扩张-压缩）
4. **无偏置**：所有 Linear 都是 bias=False

**为什么这样写？**
- 简洁：一行代码
- 高效：现代深度学习框架会优化这种写法
- 明确：直接对应 SwiGLU 公式

</details>

---

## ✅ 完成检查

完成所有题目后，检查你是否达到：

- [ ] **Q1-Q7 全对**：基础知识扎实
- [ ] **Q8 能提出 2+ 调试方法**：具备调试能力
- [ ] **Q9 能解释设计思想**：理解架构设计
- [ ] **Q10 能逐步解释代码**：理解实现细节

如果还有不清楚的地方，回到 [teaching.md](./teaching.md) 复习，或重新运行实验代码。

---

## 🎓 进阶挑战

想要更深入理解？尝试：

1. **修改实验代码**：
   - 对比 ReLU、GELU、SiLU 的激活分布
   - 可视化门控信号的模式
   - 测量不同 intermediate_size 的效果

2. **阅读论文**：
   - [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - GLU 系列详细对比
   - [Searching for Activation Functions](https://arxiv.org/abs/1710.05941) - Swish 激活函数

3. **实现变体**：
   - 实现 GeGLU（GELU 门控）
   - 实现标准 FFN（对比效果）
   - 实现 MoE 版本的 FeedForward

---

**下一步**：前往 [05. Residual Connection](../../02-architecture/05-residual-connection) 学习残差连接！
