---
layout: home
title: MiniMind LLM Training | Understand every design choice
description: Understand LLM training through controlled experiments. Modular teaching, code examples, and practical guides for interview prep.
keywords: LLM training, Transformer tutorial, deep learning, LLM interview prep, MiniMind

hero:
  name: "minimind"
  text: "Understand LLM training from scratch"
  tagline: No more black-box training — understand every design choice via controlled experiments
  actions:
    - theme: brand
      text: 🚀 Start now
      link: #quick-start
    - theme: alt
      text: 📖 Learning Roadmap
      link: /en/ROADMAP
    - theme: alt
      text: 💻 View code
      link: https://github.com/joyehuang/minimind-notes
---

<FeaturesCards />
<LearningPathCards />
<ModulesGrid />
<TerminalCode />

## 💡 Why choose this tutorial?

::: tip 🎯 No more “just make it run”
Have you ever followed a tutorial, got the code working, but still didn’t know why? This tutorial uses **controlled experiments** to show what breaks and why other choices fail.
:::

::: tip 🔬 Every design choice is backed by experiments
No more armchair theory — each module includes **runnable comparison experiments** so you can see real effects. Theory + practice, down to the details.
:::

::: tip 💻 Low barrier for learning experiments
**Learning-stage experiments**: TinyShakespeare (1MB) and similar micro datasets, runnable on CPU in minutes.
**Full training**: If you want to train a full model from scratch, you will need a GPU (MiniMind original project: single NVIDIA 3090, about 2 hours).
:::

<style>
/* Ensure tip text stays readable in dark mode */
:global(.dark) .vp-doc .custom-block {
  color: var(--vp-c-text-1);
}

:global(.dark) .vp-doc .custom-block p {
  color: var(--vp-c-text-1);
}

:global(.dark) .vp-doc .custom-block strong {
  color: var(--vp-c-text-1);
}
</style>

## 🔗 Resources

<div class="resource-grid">

**📦 Upstream project**
[jingyaogong/minimind](https://github.com/jingyaogong/minimind)

**🗺️ Learning roadmap**
[Full roadmap](/en/ROADMAP)

**💻 Code examples**
[Executable examples](/learning_materials/README)

**📝 Learning notes**
[Learning log](/learning_log) · [Knowledge base](/knowledge_base)

</div>

<style>
.resource-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
  padding: 2rem;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  border: 1px solid var(--vp-c-divider);
}

/* Dark theme tuning */
:global(.dark) .resource-grid {
  background: var(--vp-c-accent-bg);
  border: 1px solid var(--vp-c-divider);
}

:global(.dark) .resource-grid strong {
  color: var(--vp-c-brand-1);
}

.resource-grid strong {
  display: block;
  margin-bottom: 0.75rem;
  color: var(--vp-c-brand-1);
  font-size: 1.05em;
}

.resource-grid a {
  color: var(--vp-c-text-1);
  text-decoration: none;
  transition: color 0.2s;
  font-weight: 500;
}

:global(.dark) .resource-grid a {
  color: var(--vp-c-text-1);
}

.resource-grid a:hover {
  color: var(--vp-c-brand-1);
}

:global(.dark) .resource-grid a:hover {
  color: var(--vp-c-brand-2);
}

@media (max-width: 768px) {
  .resource-grid {
    grid-template-columns: 1fr;
    padding: 1.5rem;
  }
}
</style>
