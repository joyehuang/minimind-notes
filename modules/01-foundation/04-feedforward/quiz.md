---
title: FeedForward 自测题 | minimind从零理解llm训练
description: FeedForward（前馈网络）模块自测题，检验你对扩张-压缩结构、SwiGLU 激活函数和 FFN 知识存储的理解程度。
keywords: 前馈网络自测题, FeedForward测试, FFN测试题, Transformer前馈网络测试, LLM训练测试
---

# FeedForward 自测题

> 完成以下题目检验你的理解程度

---

## 🎮 交互式自测（推荐）

<script setup>
const quizData = [
  {
    question: 'FeedForward 为什么要"扩张-压缩"？',
    type: 'single',
    options: [
      { label: 'A', text: '为了减少计算量' },
      { label: 'B', text: '为了减少参数量' },
      { label: 'C', text: '为了在高维空间中增强表达能力' },
      { label: 'D', text: '为了加速推理' }
    ],
    correct: [2],
    explanation: `
      <strong>正确答案：C</strong><br><br>
      <strong>扩张-压缩架构</strong>：<ul>
        <li>输入：hidden_size (例如 512)</li>
        <li>扩张：intermediate_size (例如 2048，4倍)</li>
        <li>压缩：回到 hidden_size (512)</li>
      </ul>
      <strong>为什么要扩张？</strong><ul>
        <li>在高维空间中，模型有更大的"表达空间"</li>
        <li>可以学习更复杂的非线性变换</li>
        <li>增强特征提取能力</li>
      </ul>
      <strong>类比</strong>：就像在更大的画布上作画，有更多空间发挥
    `
  },
  {
    question: 'SwiGLU 使用几个投影矩阵？',
    type: 'single',
    options: [
      { label: 'A', text: '1 个' },
      { label: 'B', text: '2 个' },
      { label: 'C', text: '3 个' },
      { label: 'D', text: '4 个' }
    ],
    correct: [2],
    explanation: `
      <strong>正确答案：C</strong><br><br>
      <strong>SwiGLU 结构</strong>：<ul>
        <li><code>w1</code>: hidden_size → intermediate_size（门控路径）</li>
        <li><code>w2</code>: intermediate_size → hidden_size（输出投影）</li>
        <li><code>w3</code>: hidden_size → intermediate_size（值路径）</li>
      </ul>
      <strong>计算流程</strong>：<br>
      <code>output = w2(SiLU(w1(x)) * w3(x))</code><br><br>
      比标准 FFN 多了 w3，用于门控机制
    `
  },
  {
    question: 'SiLU 激活函数的公式是什么？',
    type: 'single',
    options: [
      { label: 'A', text: 'max(0, x)' },
      { label: 'B', text: 'x * sigmoid(x)' },
      { label: 'C', text: 'tanh(x)' },
      { label: 'D', text: '1 / (1 + e^(-x))' }
    ],
    correct: [1],
    explanation: `
      <strong>正确答案：B</strong><br><br>
      <strong>SiLU (Swish) 公式</strong>：<br>
      <code>SiLU(x) = x · σ(x) = x · (1 / (1 + e^(-x)))</code><br><br>
      <strong>特点</strong>：<ul>
        <li>平滑的非线性函数</li>
        <li>负值不会完全消失（与 ReLU 不同）</li>
        <li>有界下界（约 -0.28），无上界</li>
        <li>梯度更平滑，训练更稳定</li>
      </ul>
      <strong>对比</strong>：ReLU = max(0, x)，SiLU 更柔和
    `
  },
  {
    question: 'SwiGLU 中门控机制的作用是什么？',
    type: 'single',
    options: [
      { label: 'A', text: '加速计算' },
      { label: 'B', text: '减少参数量' },
      { label: 'C', text: '选择性地控制信息流' },
      { label: 'D', text: '防止梯度消失' }
    ],
    correct: [2],
    explanation: `
      <strong>正确答案：C</strong><br><br>
      <strong>门控机制</strong>：<br>
      <code>gate = SiLU(w1(x))</code><br>
      <code>value = w3(x)</code><br>
      <code>output = gate * value</code><br><br>
      <strong>作用</strong>：<ul>
        <li>gate 决定"开多大"</li>
        <li>value 决定"传什么信息"</li>
        <li>两者相乘实现选择性传递</li>
        <li>类似 LSTM 的门控思想</li>
      </ul>
      模型可以学习在不同位置控制信息流的强度
    `
  },
  {
    question: 'FeedForward 与 Attention 的主要区别是什么？',
    type: 'single',
    options: [
      { label: 'A', text: 'FFN 有更多参数' },
      { label: 'B', text: 'FFN 独立处理每个位置，Attention 混合所有位置' },
      { label: 'C', text: 'FFN 计算更快' },
      { label: 'D', text: 'FFN 效果更好' }
    ],
    correct: [1],
    explanation: `
      <strong>正确答案：B</strong><br><br>
      <strong>Attention</strong>：<ul>
        <li>混合所有位置的信息</li>
        <li>位置 i 的输出依赖于所有位置</li>
        <li>负责"信息交互"</li>
      </ul>
      <strong>FeedForward</strong>：<ul>
        <li>独立处理每个位置</li>
        <li>位置 i 的输出只依赖位置 i</li>
        <li>负责"特征变换"</li>
      </ul>
      <strong>比喻</strong>：Attention 是讨论（交流信息），FFN 是个人思考（深化理解）
    `
  },
  {
    question: '为什么 SwiGLU 通常比 ReLU 效果更好？',
    type: 'single',
    options: [
      { label: 'A', text: '计算更快' },
      { label: 'B', text: '参数更少' },
      { label: 'C', text: '门控机制提供更丰富的非线性，梯度更平滑' },
      { label: 'D', text: '实现更简单' }
    ],
    correct: [2],
    explanation: `
      <strong>正确答案：C</strong><br><br>
      <strong>ReLU 的局限</strong>：<ul>
        <li>硬截断：负值直接变 0</li>
        <li>梯度不连续</li>
        <li>Dead ReLU 问题（神经元永久失活）</li>
      </ul>
      <strong>SwiGLU 的优势</strong>：<ul>
        <li>平滑的激活函数（SiLU）</li>
        <li>门控机制增加表达能力</li>
        <li>梯度更稳定，训练更容易</li>
        <li>实验证明在 LLM 上效果更好</li>
      </ul>
      代价：计算量略增加（3个投影 vs 2个）
    `
  },
  {
    question: 'MiniMind 中 intermediate_size 通常是 hidden_size 的几倍？',
    type: 'single',
    options: [
      { label: 'A', text: '2 倍' },
      { label: 'B', text: '4 倍' },
      { label: 'C', text: '8 倍' },
      { label: 'D', text: '16 倍' }
    ],
    correct: [1],
    explanation: `
      <strong>正确答案：B</strong><br><br>
      <strong>标准配置</strong>：<ul>
        <li>hidden_size = 512 → intermediate_size = 2048 (4倍)</li>
        <li>hidden_size = 768 → intermediate_size = 3072 (4倍)</li>
      </ul>
      <strong>为什么是 4 倍？</strong><ul>
        <li>平衡表达能力和计算成本</li>
        <li>Llama、GPT 等主流模型都用 4 倍</li>
        <li>实验证明是较好的折中</li>
      </ul>
      <strong>变体</strong>：<ul>
        <li>MoE 模型可能用更大倍数</li>
        <li>小模型可能用 2-3 倍节省参数</li>
      </ul>
    `
  }
]
</script>

<InteractiveQuiz :questions="quizData" quiz-id="feedforward" />

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
