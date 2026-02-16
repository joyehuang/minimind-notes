---
layout: home
title: MiniMind 学习指南 | minimind从零理解llm训练
description: MiniMind 学习指南首页，提供快速体验、系统学习和深度掌握三条学习路径。从零开始理解大语言模型训练，适合准备大模型岗位面试的同学。
keywords: LLM学习指南, 大模型训练教程, Transformer教程, LLM面试准备, 大模型实习准备

hero:
  name: "MiniMind"
  text: "从零理解 LLM 训练"
  tagline: 开源教程 · 对照实验 · 可执行代码
  actions:
    - theme: brand
      text: 🚀 开始学习
      link: /modules/01-foundation/01-normalization/teaching
    - theme: alt
      text: ⚡ 30分钟快速体验
      link: /docs/guide/quick-start
    - theme: alt
      text: 📚 查看完整路线
      link: /ROADMAP

features:
  - icon: 🎯
    title: 30分钟快速体验
    details: 通过 3 个关键实验理解核心设计选择
    link: /docs/guide/quick-start
    linkText: 开始体验

  - icon: 📚
    title: 6小时系统学习
    details: 完整掌握 Transformer 所有基础组件
    link: /docs/guide/systematic
    linkText: 查看路线

  - icon: 🎓
    title: 30小时深度掌握
    details: 从零训练一个完整的 LLM
    link: /docs/guide/mastery
    linkText: 完整路线

  - icon: 🔬
    title: 对照实验验证
    details: 每个设计选择都通过实验回答 "不这样做会怎样？"
    link: /modules/01-foundation/01-normalization/
    linkText: 查看实验

  - icon: 💻
    title: 可执行代码
    details: 所有实验可在普通笔记本运行，无需 GPU
    link: /learning_materials/README
    linkText: 运行代码

  - icon: 🧱
    title: 模块化教学
    details: 4个基础组件 + 2个架构模块，逐步深入
    link: /modules/
    linkText: 查看模块
---

## 📚 学习模块

<div class="modules-grid">

### 🧱 基础组件 (Foundation)

<div class="module-cards">

#### [01 归一化](/modules/01-foundation/01-normalization/teaching)
**核心问题**: 为什么需要归一化？Pre-LN vs Post-LN？
**时长**: 1小时 | **状态**: ✅ 完成

[开始学习 →](/modules/01-foundation/01-normalization/teaching)

---

#### [02 位置编码](/modules/01-foundation/02-position-encoding/teaching)
**核心问题**: 为什么选择 RoPE？如何实现长度外推？
**时长**: 1.5小时 | **状态**: ✅ 完成

[开始学习 →](/modules/01-foundation/02-position-encoding/teaching)

---

#### [03 注意力机制](/modules/01-foundation/03-attention/teaching)
**核心问题**: QKV 的直觉是什么？为什么需要多头？
**时长**: 2小时 | **状态**: ✅ 完成

[开始学习 →](/modules/01-foundation/03-attention/teaching)

---

#### [04 前馈网络](/modules/01-foundation/04-feedforward/teaching)
**核心问题**: FFN 存储了什么知识？为什么需要扩张？
**时长**: 1小时 | **状态**: ✅ 完成

[开始学习 →](/modules/01-foundation/04-feedforward/teaching)

</div>

### 🏗️ 架构组装 (Architecture)

<div class="module-cards">

#### [残差连接与 Transformer Block](/modules/02-architecture/)
**核心问题**: 如何将组件组装成完整的 Transformer？
**时长**: 2.5小时 | **状态**: 📋 规划中

[查看内容 →](/modules/02-architecture/)

</div>

</div>

## 🚀 快速开始

<QuickStartTimeline />

### 💻 运行实验

::: code-group

```bash [1. 环境准备]
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash [2. 运行第一个实验]
# 实验: 为什么需要归一化？
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py

# 你会看到:
# ❌ 无归一化: 梯度消失
# ✅ 有 RMSNorm: 梯度稳定
```

```bash [3. 查看教学文档]
# 理解背后的原理
cat modules/01-foundation/01-normalization/teaching.md
```

:::

## 💡 教学特色

::: tip 🎯 原理优先，而非命令复制
不是告诉你"运行这个命令就能训练模型"，而是让你理解"为什么要这样设计"
:::

::: tip 🔬 对照实验验证
每个设计选择都通过实验回答两个问题:
- **不这样做会怎样？**
- **其他方案为什么不行？**
:::

::: tip 💻 可在普通笔记本运行
所有实验基于 TinyShakespeare (1MB) 或 TinyStories (10-50MB)
无需 GPU，每个实验 < 10 分钟
:::

## 🔗 相关资源

<div class="resource-grid">

**📦 原项目**
[jingyaogong/minimind](https://github.com/jingyaogong/minimind)

**🗺️ 学习路线**
[完整路线图](/ROADMAP)

**💻 代码示例**
[可执行示例](/learning_materials/README)

**📝 学习笔记**
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
