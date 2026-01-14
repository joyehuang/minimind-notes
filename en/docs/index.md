---
layout: home
title: MiniMind Learning Guide | MiniMind LLM Training
description: Choose a learning path and start experimenting with core LLM training concepts.
keywords: LLM training guide, Transformer tutorial, MiniMind

hero:
  name: "MiniMind"
  text: "Learning Guide"
  tagline: Principles + experiments + practice
  actions:
    - theme: brand
      text: Start with a module
      link: /modules/01-foundation/01-normalization/teaching
    - theme: alt
      text: Quick Start (30 min)
      link: /en/docs/guide/quick-start
    - theme: alt
      text: Roadmap
      link: /en/ROADMAP
---

## Learning modules

<div class="modules-grid">

### Foundation

<div class="module-cards">

#### [01 Normalization](/modules/01-foundation/01-normalization/teaching)
**Focus**: Pre-LN vs Post-LN, why normalization matters
**Time**: 1 hour | **Status**: Complete

[Start learning →](/modules/01-foundation/01-normalization/teaching)

---

#### [02 Position Encoding](/modules/01-foundation/02-position-encoding/teaching)
**Focus**: RoPE and position encoding choices
**Time**: 1.5 hours | **Status**: Complete

[Start learning →](/modules/01-foundation/02-position-encoding/teaching)

---

#### [03 Attention](/modules/01-foundation/03-attention/teaching)
**Focus**: Q/K/V, multi-head attention
**Time**: 2 hours | **Status**: Complete

[Start learning →](/modules/01-foundation/03-attention/teaching)

---

#### [04 FeedForward](/modules/01-foundation/04-feedforward/teaching)
**Focus**: FFN design and SwiGLU
**Time**: 1 hour | **Status**: Complete

[Start learning →](/modules/01-foundation/04-feedforward/teaching)

</div>

### Architecture

<div class="module-cards">

#### [Transformer Block Assembly](/modules/02-architecture/)
**Focus**: assemble components into a Transformer block
**Time**: 2.5 hours | **Status**: In progress

[Open architecture overview →](/modules/02-architecture/)

</div>

</div>

## Quick Start

<QuickStartTimeline />

### Run the first experiment

::: code-group

```bash [1. Clone the repo]
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes
source venv/bin/activate
```

```bash [2. Run Experiment 1]
# Experiment 1: Why normalization?
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py

# What you will see:
# ❌ No normalization: activation std drops (vanishing gradients)
# ✅ RMSNorm: activation std stays stable
```

```bash [3. Read the teaching notes]
# Read teaching notes for the why/what/how
cat modules/01-foundation/01-normalization/teaching.md
```

:::

## Learning principles

::: tip ✅ Principles first
Run experiments first, then read theory. Focus on **why** each design choice exists.
:::

::: tip 🔬 Experiment-driven learning
Each module includes experiments that answer: “What breaks if we don’t do this?”
:::

::: tip 💻 Low barrier
TinyShakespeare (1MB) or TinyStories (10–50MB) run on CPU in minutes. GPU is optional for learning.
:::

## Resources

<div class="resource-grid">

**Upstream project**
[jingyaogong/minimind](https://github.com/jingyaogong/minimind)

**Learning roadmap**
[Roadmap](/en/ROADMAP)

**Executable examples**
[Learning materials](/learning_materials/README)

**Learning notes**
[Learning log](/learning_log) · [Knowledge base](/knowledge_base)

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
