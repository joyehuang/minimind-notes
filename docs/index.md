---
layout: home

hero:
  name: "MiniMind 学习笔记"
  text: "深入理解 LLM 训练的每个设计选择"
  tagline: 从零开始训练语言模型 | 理论+实验+实践
  actions:
    - theme: brand
      text: 📅 学习日志
      link: /learning_log
    - theme: alt
      text: 📚 知识库
      link: /knowledge_base
    - theme: alt
      text: 🧱 模块教学
      link: /modules/

features:
  - icon: 📝
    title: 学习日志
    details: 记录每日学习进度、问题和思考
    link: /learning_log
    linkText: 查看日志

  - icon: 📚
    title: 知识库
    details: 系统化整理的技术知识和问答记录
    link: /knowledge_base
    linkText: 浏览知识库

  - icon: 🧱
    title: 模块化教学
    details: 4个基础组件 + 2个架构模块，通过对照实验理解设计选择
    link: /modules/
    linkText: 开始学习

  - icon: 💻
    title: 可执行代码示例
    details: 理解归一化、位置编码、注意力机制的可运行示例
    link: /learning_materials/README
    linkText: 运行代码

  - icon: 🎯
    title: 三条学习路径
    details: 快速体验(30分钟) / 系统学习(6小时) / 深度掌握(30小时)
    link: /ROADMAP
    linkText: 选择路径

  - icon: 🔬
    title: 对照实验验证
    details: 通过可执行实验回答"不这样做会怎样？"
    link: /modules/01-foundation/01-normalization/
    linkText: 查看实验
---

## 🎯 当前学习进度

<div class="progress-container">

**阶段**: 第一阶段 - Transformer 核心组件学习中

**完成度**: 2/4

- ✅ **RMSNorm (归一化)** - 理解梯度稳定机制
- ✅ **RoPE (位置编码)** - 理解多频率旋转机制
- ⏳ **Attention (注意力机制)** - 学习中
- ⏳ **FeedForward (前馈网络)** - 待学习

</div>

## 🚀 快速开始

::: code-group

```bash [激活环境]
# 克隆仓库
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes

# 激活虚拟环境
source venv/bin/activate
```

```bash [运行实验]
# 实验 1: 为什么需要归一化?
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py

# 实验 2: 理解 RoPE 位置编码
cd ../../02-position-encoding/experiments
python exp1_rope_basics.py

# 实验 3: Attention 如何工作?
cd ../../03-attention/experiments
python exp1_attention_basics.py
```

```bash [测试模型]
# 测试预训练模型
python eval_llm.py --load_from ./MiniMind2
```

:::

## 🎯 学习路径

<div class="path-cards">

### ⚡ 快速体验 (30分钟)

通过 3 个关键实验理解核心设计选择

- 为什么需要归一化?
- 为什么用 RoPE?
- Attention 如何工作?

[开始体验 →](/docs/guide/quick-start)

### 📚 系统学习 (6小时)

完整掌握 Transformer 所有基础组件

- Foundation 4个模块
- Architecture 组装
- 从零实现 Transformer Block

[查看路线 →](/docs/guide/systematic)

### 🎓 深度掌握 (30+小时)

从零训练一个完整的 LLM

- 数据准备 + Tokenizer 训练
- Pretrain → SFT → LoRA
- RLHF / RLAIF 进阶

[完整路线 →](/docs/guide/mastery)

</div>

## 📖 模块概览

### 🧱 基础组件 (Foundation)

| 模块 | 核心问题 | 时长 | 状态 |
|------|---------|------|------|
| [归一化](/modules/01-foundation/01-normalization/) | 为什么需要归一化? Pre-LN vs Post-LN? | 1h | ✅ |
| [位置编码](/modules/01-foundation/02-position-encoding/) | 为什么选择 RoPE? 如何实现长度外推? | 1.5h | ✅ |
| [注意力机制](/modules/01-foundation/03-attention/) | QKV 的直觉是什么? 为什么需要多头? | 2h | ✅ |
| [前馈网络](/modules/01-foundation/04-feedforward/) | FFN 存储了什么知识? 为什么需要扩张? | 1h | ✅ |

### 🏗️ 架构组装 (Architecture)

| 模块 | 核心问题 | 时长 | 状态 |
|------|---------|------|------|
| [残差连接](/modules/02-architecture/) | 为什么需要残差连接? 如何稳定梯度流? | 1h | 📋 |
| [Transformer Block](/modules/02-architecture/) | 如何编排组件顺序? 为什么是这个顺序? | 1.5h | 📋 |

## 💡 设计理念

::: tip 原理优先,而非命令复制
不是"运行这个命令就能训练模型",而是"理解为什么要这样设计"
:::

::: tip 对照实验验证
每个设计选择都通过实验回答:**不这样做会怎样?** **其他方案为什么不行?**
:::

::: tip 可在普通笔记本运行
所有实验基于 TinyShakespeare (1MB) 或 TinyStories (10-50MB),无需 GPU,每个实验 < 10 分钟
:::

## 🔗 相关资源

- 📦 **原项目**: [jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- 📝 **学习日志**: [我的学习记录](/learning_log)
- 📚 **知识库**: [技术知识整理](/knowledge_base)
- 💻 **代码示例**: [可执行示例](/learning_materials/README)
- 🗺️ **学习路线**: [完整路线图](/ROADMAP)

<style>
.progress-container {
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  margin: 1rem 0;
  border: 1px solid var(--vp-c-divider);
}

.path-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}

.path-cards > div {
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
  transition: all 0.3s ease;
}

.path-cards > div:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px var(--vp-c-brand-soft);
}

.path-cards h3 {
  margin-top: 0;
  color: var(--vp-c-brand-1);
}
</style>
