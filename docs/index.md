---
layout: home
title: MiniMind 学习指南 | minimind从零理解llm训练
description: 选择学习路径，通过可执行实验深入理解 LLM 训练的核心概念。
keywords: LLM学习指南, Transformer教程, MiniMind, 大模型训练

hero:
  name: "MiniMind"
  text: "学习指南"
  tagline: 原理 + 实验 + 实践
  actions:
    - theme: brand
      text: 从模块开始
      link: /modules/01-foundation/01-normalization/teaching
    - theme: alt
      text: 快速体验 (30分钟)
      link: /docs/guide/quick-start
    - theme: alt
      text: 学习路线图
      link: /ROADMAP
---

## 学习模块

<div class="modules-grid">

### 基础组件

<div class="module-cards">

#### [01 归一化 (Normalization)](/modules/01-foundation/01-normalization/teaching)
**重点**: Pre-LN vs Post-LN，为什么需要归一化
**时长**: 1 小时 | **状态**: 完成

[开始学习 →](/modules/01-foundation/01-normalization/teaching)

---

#### [02 位置编码 (Position Encoding)](/modules/01-foundation/02-position-encoding/teaching)
**重点**: RoPE 与位置编码的设计选择
**时长**: 1.5 小时 | **状态**: 完成

[开始学习 →](/modules/01-foundation/02-position-encoding/teaching)

---

#### [03 注意力机制 (Attention)](/modules/01-foundation/03-attention/teaching)
**重点**: Q/K/V 的直觉，多头注意力
**时长**: 2 小时 | **状态**: 完成

[开始学习 →](/modules/01-foundation/03-attention/teaching)

---

#### [04 前馈网络 (FeedForward)](/modules/01-foundation/04-feedforward/teaching)
**重点**: FFN 的设计与 SwiGLU 激活函数
**时长**: 1 小时 | **状态**: 完成

[开始学习 →](/modules/01-foundation/04-feedforward/teaching)

</div>

### 架构组装

<div class="module-cards">

#### [Transformer Block 组装](/modules/02-architecture/)
**重点**: 将基础组件组装成完整的 Transformer Block
**时长**: 2.5 小时 | **状态**: 开发中

[查看架构总览 →](/modules/02-architecture/)

</div>

</div>

## 快速开始

<QuickStartTimeline />

### 运行第一个实验

::: code-group

```bash [1. 克隆仓库]
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes
source venv/bin/activate
```

```bash [2. 运行实验 1]
# 实验 1：为什么需要归一化？
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py

# 你将观察到：
# ❌ 无归一化：激活值标准差衰减（梯度消失）
# ✅ RMSNorm：激活值标准差保持稳定
```

```bash [3. 阅读教学文档]
# 阅读教学笔记，了解 Why/What/How
cat modules/01-foundation/01-normalization/teaching.md
```

:::

## 学习理念

::: tip ✅ 原理优先
先跑实验，再读理论。重点理解每个设计选择**为什么**存在。
:::

::: tip 🔬 实验驱动学习
每个模块都包含对照实验，回答："不这样做会怎样？"
:::

::: tip 💻 低门槛
TinyShakespeare (1MB) 或 TinyStories (10-50MB)，CPU 上几分钟即可运行，学习阶段无需 GPU。
:::

## 相关资源

<div class="resource-grid">

**原项目**
[jingyaogong/minimind](https://github.com/jingyaogong/minimind)

**学习路线图**
[Roadmap](/ROADMAP)

**可执行示例**
[学习材料](/learning_materials/README)

**学习笔记**
[学习日志](/learning_log) · [知识库](/knowledge_base)

</div>

<style>
.modules-grid {
  margin: 2rem 0;
}

.module-cards {
  display: grid;
  gap: 1.5rem;
  margin: 1rem 0 2rem 0;
}

.module-cards h4 {
  margin: 0 0 0.5rem 0;
  font-size: 1.2em;
}

.module-cards h4 a {
  text-decoration: none;
  color: var(--vp-c-brand-1);
  transition: color 0.2s;
}

.module-cards h4 a:hover {
  color: var(--vp-c-brand-2);
}

.module-cards p {
  margin: 0.5rem 0;
  color: var(--vp-c-text-2);
}

.module-cards hr {
  margin: 1.5rem 0;
  border: none;
  border-top: 1px solid var(--vp-c-divider);
}

.resource-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin: 1rem 0;
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
}

.resource-grid strong {
  display: block;
  margin-bottom: 0.5rem;
  color: var(--vp-c-brand-1);
}

.resource-grid a {
  color: var(--vp-c-text-1);
  text-decoration: none;
  transition: color 0.2s;
}

.resource-grid a:hover {
  color: var(--vp-c-brand-1);
}

@media (max-width: 768px) {
  .resource-grid {
    grid-template-columns: 1fr;
  }
}
</style>
