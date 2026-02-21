---
title: MiniMind 模块化教学导航 | minimind从零理解llm训练
description: MiniMind 模块化教学导航，通过对照实验理解 LLM 训练的每个设计选择。包含基础组件和架构组装的完整学习路径。
keywords: LLM训练教程, Transformer教程, 模块化教学, 大模型训练, 深度学习教程
---

# MiniMind 模块化教学

> 通过对照实验理解 LLM 训练的每个设计选择

---

## 📚 模块导航

### 🧱 Tier 1: Foundation（基础组件）

核心问题：**Transformer 的基本构建块是如何工作的？**

| 模块 | 核心问题 | 预计时长 | 状态 |
|------|---------|---------|------|
| [01-normalization](./01-foundation/01-normalization) | 为什么需要归一化？Pre-LN vs Post-LN？ | 1 小时 | ✅ |
| [02-position-encoding](./01-foundation/02-position-encoding) | 为什么选择 RoPE？如何实现长度外推？ | 1.5 小时 | ✅ |
| [03-attention](./01-foundation/03-attention) | QKV 的直觉是什么？为什么需要多头？ | 2 小时 | ✅ |
| [04-feedforward](./01-foundation/04-feedforward) | FFN 存储了什么知识？为什么需要扩张？ | 1 小时 | ✅ |

**完成标准**：
- ✅ 理解每个组件的数学原理
- ✅ 运行对照实验，观察去掉该组件的影响
- ✅ 能用自己的话解释设计选择

---

### 🏗️ Tier 2: Architecture（架构组装）

核心问题：**如何将基础组件组装成完整的 Transformer？**

| 模块 | 核心问题 | 预计时长 | 状态 |
|------|---------|---------|------|
| [01-residual-connection](./02-architecture/01-residual-connection) | 为什么需要残差连接？如何稳定梯度流？ | 1 小时 | 📋 |
| [02-transformer-block](./02-architecture/02-transformer-block) | 如何编排组件顺序？为什么是这个顺序？ | 1.5 小时 | 📋 |

**完成标准**：
- ✅ 理解残差连接的作用
- ✅ 理解 Pre-Norm 架构的优势
- ✅ 能从零实现一个 Transformer Block

---

### 🚀 Tier 3: Training（训练流程）

_（后续扩展）_

---

### 🎓 Tier 4: Advanced（进阶主题）

_（后续扩展）_

---

## ⚡ 快速开始

### 准备环境

```bash
# 1. 激活虚拟环境
source venv/bin/activate

# 2. 下载实验数据（约 60 MB）
cd modules/common
python data_sources.py --download-all
```

### 30 分钟快速体验

运行三个关键实验，快速理解核心设计选择：

```bash
# 实验 1：为什么需要归一化？（5 分钟）
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py

# 实验 2：为什么用 RoPE？（10 分钟）
cd ../../02-position-encoding/experiments
python exp2_rope_vs_absolute.py --quick

# 实验 3：为什么需要残差连接？（5 分钟）
cd ../../../02-architecture/01-residual-connection/experiments
python exp1_with_vs_without.py --quick
```

### 系统学习路径

**建议顺序**：
1. **Foundation 层**（5.5 小时）
   - 按顺序学习 01 → 02 → 03 → 04
   - 每个模块：阅读 `teaching.md` → 运行实验 → 做自测题

2. **Architecture 层**（2.5 小时）
   - 理解如何组装基础组件

3. **实践项目**（可选）
   - 从零训练一个 tiny 模型
   - 在真实任务上测试

---

## 📖 学习方法

### 每个模块的学习流程

```
1. 阅读 README.md        # 了解模块概览（5 分钟）
   ↓
2. 阅读 teaching.md      # 理解核心概念（20 分钟）
   ↓
3. 运行实验代码          # 验证理论（20 分钟）
   ↓
4. 阅读 code_guide.md   # 理解真实实现（10 分钟）
   ↓
5. 完成 quiz.md          # 自测巩固（5 分钟）
```

### 实验代码使用

所有实验都支持：

```bash
# 完整运行（推荐）
python exp_xxx.py

# 快速模式（仅验证概念，< 2 分钟）
python exp_xxx.py --quick

# 查看帮助
python exp_xxx.py --help
```

实验结果保存在各模块的 `experiments/results/` 目录。

---

## 🎯 设计理念

### 1️⃣ 原理优先，而非命令复制

- ❌ "运行这个命令就能训练模型"
- ✅ "理解为什么要这样设计"

### 2️⃣ 对照实验验证

每个设计选择都通过实验回答：
- **不这样做会怎样？**
- **其他方案为什么不行？**

### 3️⃣ 渐进式学习

- 从单个组件 → 组装架构 → 完整训练
- 每一步都有清晰的目标和验证

### 4️⃣ 可在普通笔记本上运行

- 所有实验基于 TinyShakespeare（1MB）或 TinyStories（10-50MB）
- 无需 GPU（CPU/MPS 均可）
- 每个实验 < 10 分钟

---

## 🛠️ 通用工具

模块提供了以下通用工具（位于 `modules/common/`）：

### data_sources.py - 数据集管理

```python
from modules.common.data_sources import get_experiment_data

# 获取 TinyShakespeare
text = get_experiment_data('shakespeare')

# 获取 TinyStories 子集
texts = get_experiment_data('tinystories', size_mb=10)
```

### experiment_base.py - 实验基类

```python
from modules.common.experiment_base import Experiment

class MyExperiment(Experiment):
    def run(self):
        # 实验代码
        pass
```

### visualization.py - 可视化工具

```python
from modules.common.visualization import (
    plot_attention_heatmap,
    plot_activation_distribution,
    plot_gradient_flow,
    plot_loss_curves
)
```

详细文档见各文件的 docstring。

---

## 🤝 贡献指南

欢迎补充：
- 新的对照实验
- 更好的直觉类比
- 可视化图表
- Bug 修复

提交前请确保：
- [ ] 实验可以独立运行
- [ ] 代码有充分的中文注释
- [ ] 结果可复现（固定随机种子）
- [ ] 遵循现有的文件结构

---

## 📜 致谢

本教学模块基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 项目。

所有实验代码链接到原仓库的真实实现，帮助学习者理解工业级代码。

---

## 📞 相关文档

- 📝 [个人学习日志](../docs/learning_log.md)
- 📚 [知识库](../docs/knowledge_base.md)
- 🗺️ [学习路线图](../ROADMAP.md)
- 🏠 [项目主页](../README.md)
