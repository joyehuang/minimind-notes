# Tier 2: Architecture（架构组装）

> 将基础组件组装成完整的 Transformer

---

## 🎯 学习目标

完成本层后，你将能够：
- ✅ 理解残差连接的作用机制
- ✅ 理解 Pre-Norm Transformer Block 的组件编排
- ✅ 理解梯度流在深层网络中的传播
- ✅ 从零实现完整的 Transformer Block

---

## 📚 模块列表

### [01. Residual Connection（残差连接）](./01-residual-connection)

**核心问题**：
- 为什么需要残差连接？
- 残差连接如何解决梯度消失问题？
- 残差连接的数学本质是什么？

**关键实验**：
- 实验 1：有/无残差连接的训练对比
- 实验 2：梯度流可视化
- 实验 3：深度影响（浅层 vs 深层网络）

**预计时长**：1 小时

**前置知识**：
- Tier 1: Foundation（所有模块）

---

### [02. Transformer Block（Transformer 块）](./02-transformer-block)

**核心问题**：
- 如何组装 Norm、Attention、FFN、Residual？
- 为什么组件顺序是这样的？
- Pre-Norm vs Post-Norm 的区别是什么？

**关键实验**：
- 实验 1：组件消融（去掉某个组件的影响）
- 实验 2：顺序影响（不同编排方式的对比）
- 实验 3：层数影响（2 层 vs 8 层）

**预计时长**：1.5 小时

**前置知识**：
- 01. Residual Connection

---

## 🏗️ 从组件到架构

### Transformer Block 的组装逻辑

```
输入 x
  ├─ 保存为 residual
  ↓
RMSNorm（归一化，稳定分布）
  ↓
Attention（关注相关信息）+ RoPE（位置信息）
  ↓
+ residual（残差连接 #1）
  ├─ 保存当前状态
  ↓
RMSNorm（再次归一化）
  ↓
FeedForward（处理信息，存储知识）
  ↓
+ residual（残差连接 #2）
  ↓
输出
```

**设计要点**：
1. **Pre-Norm**：归一化在子层之前
   - 保证每个子层的输入分布稳定
   - 梯度流更平滑

2. **双残差**：Attention 和 FFN 各一个
   - 梯度可以跳过复杂的子层
   - 增量学习：只学习"调整量"

3. **顺序固定**：Attention → FFN
   - Attention：找到相关信息
   - FFN：处理这些信息

---

## 🚀 学习建议

### 学习路径

**必须按顺序**：
1. Residual Connection（理解残差的作用）
2. Transformer Block（理解整体组装）

### 实践项目（可选）

完成本层后，尝试：

**项目 1：从零实现 Transformer Decoder**
```python
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=8, hidden_size=512, ...):
        # 堆叠多个 Transformer Block
        self.blocks = nn.ModuleList([
            TransformerBlock(...) for _ in range(num_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
```

**项目 2：训练一个 Tiny 模型**
- 数据：TinyShakespeare (1MB)
- 模型：4 层 Transformer，256 hidden size
- 目标：困惑度 < 2.5
- 时间：约 30 分钟（CPU）

---

## 📊 完成检查

完成本层后，你应该能够回答：

### 理论理解
- [ ] 残差连接的数学表达式是什么？
- [ ] 为什么残差连接能解决梯度消失？
- [ ] Pre-Norm 和 Post-Norm 的梯度流有什么区别？
- [ ] 为什么 Attention 要在 FFN 之前？

### 实践能力
- [ ] 能从零实现残差连接
- [ ] 能从零实现 Pre-Norm Transformer Block
- [ ] 能堆叠多个 Block 形成完整模型
- [ ] 能训练一个简单的语言模型

### 设计直觉
- [ ] 能解释为什么深层网络需要残差连接
- [ ] 能解释 Pre-Norm 为什么比 Post-Norm 稳定
- [ ] 能解释双残差的必要性
- [ ] 能画出完整的 Transformer Block 数据流图

---

## 🎓 下一步

完成本层后，前往：
👉 **Tier 3: Training（训练流程）**（待开发）

学习如何训练这些模型：
- 数据准备和 Tokenization
- 优化器和学习率调度
- 分布式训练
- 评估和调试

---

## 🔗 相关资源

### 论文
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - ResNet 原始论文
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - Pre-LN vs Post-LN

### 代码实现
- MiniMind: `model/model_minimind.py:359-380`（TransformerBlock）
- MiniMind: `model/model_minimind.py:430-520`（完整模型）

### 可视化工具
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
