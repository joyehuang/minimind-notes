---
title: Quick Start (30 min) | MiniMind LLM Training
description: Three experiments to grasp core LLM training design choices in 30 minutes.
keywords: LLM quick start, normalization, RoPE, attention, experiments
---

# Quick Start (30 min)

Run three short experiments to understand the most important LLM design choices.

## Setup (5 min)

```bash
# 1. Clone the repo
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes

# 2. Activate your virtual environment
source venv/bin/activate

# 3. Download experiment datasets (optional)
cd modules/common
python data_sources.py --download-all
cd ../..
```

## Experiment 1: Normalization (10 min)

Observe gradient vanishing and see why Pre-LN + RMSNorm is stable.

```bash
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py
```

Next: `/en/modules/01-foundation/01-normalization/`

---

## Experiment 2: RoPE (10 min)

Compare absolute position encoding and learn why RoPE extrapolates better.

```bash
cd ../../02-position-encoding/experiments
python exp1_rope_basics.py
```

Next: `/en/modules/01-foundation/02-position-encoding/`

---

## Experiment 3: Attention (10 min)

Understand how Q/K/V and attention weights work.

```bash
cd ../../03-attention/experiments
python exp1_attention_basics.py
```

Next: `/en/modules/01-foundation/03-attention/`

---

## Where to go next

- Systematic Study: `/en/docs/guide/systematic`
- Deep Mastery: `/en/docs/guide/mastery`
- Roadmap: `/en/ROADMAP`
