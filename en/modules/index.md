---
title: MiniMind Modular Teaching | MiniMind LLM Training
description: Modular teaching navigation for MiniMind. Understand each LLM training design choice through controlled experiments.
keywords: LLM training tutorial, Transformer tutorial, modular learning
---

# MiniMind Modular Teaching

> Understand every LLM design choice through controlled experiments

---

## 📚 Module Navigator

### 🧱 Tier 1: Foundation (core components)

Core question: **How do the basic building blocks of a Transformer work?**

| Module | Core question | Time | Status |
|------|---------|---------|------|
| [01-normalization](/en/modules/01-foundation/01-normalization) | Why normalization? Pre-LN vs Post-LN? | 1 hour | ✅ |
| [02-position-encoding](/en/modules/01-foundation/02-position-encoding) | Why RoPE? How does extrapolation work? | 1.5 hours | ✅ |
| [03-attention](/en/modules/01-foundation/03-attention) | What is the intuition behind QKV? Why multi-head? | 2 hours | ✅ |
| [04-feedforward](/en/modules/01-foundation/04-feedforward) | What does FFN store? Why expansion? | 1 hour | ✅ |

**Completion criteria**:
- ✅ Understand the math behind each component
- ✅ Run controlled experiments and observe what breaks if removed
- ✅ Explain design choices in your own words

---

### 🏗️ Tier 2: Architecture (assembly)

Core question: **How do we assemble components into a full Transformer?**

| Module | Core question | Time | Status |
|------|---------|---------|------|
| [01-residual-connection](/en/modules/02-architecture/01-residual-connection) | Why residuals? How do they stabilize gradients? | 1 hour | 📋 |
| [02-transformer-block](/en/modules/02-architecture/02-transformer-block) | Why this assembly order? | 1.5 hours | 📋 |

**Completion criteria**:
- ✅ Understand residual connections
- ✅ Understand why Pre-Norm works better
- ✅ Implement a Transformer block from scratch

---

### 🚀 Tier 3: Training

_(Planned)_

---

### 🎓 Tier 4: Advanced

_(Planned)_

---

## 📋 System Requirements

### Python Version
- **Recommended**: Python 3.10+
- **Minimum**: Python 3.10

Some utility code uses Python 3.10+ union type syntax (e.g., `str | list`). Earlier versions will not work.

### Dependencies
```bash
pip install torch requests datasets matplotlib numpy
```

See: [Environment Setup Guide](../docs/guide/environment-setup.md)

---

## ⚡ Quick Start

### Environment setup

```bash
# 1. Activate your virtual environment
source venv/bin/activate

# 2. Download experiment data (~60 MB)
cd modules/common
python data_sources.py --download-all
```

### 30-minute quick experience

Run three key experiments to grasp core design choices:

```bash
# Experiment 1: Why normalization? (5 min)
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py

# Experiment 2: Why RoPE? (10 min)
cd ../../02-position-encoding/experiments
python exp2_rope_vs_absolute.py --quick

# Experiment 3: Why residual connections? (5 min)
cd ../../../02-architecture/01-residual-connection/experiments
python exp1_with_vs_without.py --quick
```

### Systematic study path

**Recommended order**:
1. **Foundation layer** (5.5 hours)
   - Study 01 → 02 → 03 → 04 in order
   - Each module: read `teaching.md` → run experiments → finish quiz

2. **Architecture layer** (2.5 hours)
   - Learn how to assemble components

3. **Practice project** (optional)
   - Train a tiny model from scratch
   - Test on a real task

---

## 📖 Learning method

### The recommended flow for each module

```
1. Read README.md        # Overview (5 min)
   ↓
2. Read teaching.md      # Core concepts (20 min)
   ↓
3. Run experiments       # Validate theory (20 min)
   ↓
4. Read code_guide.md    # Understand implementation (10 min)
   ↓
5. Finish quiz.md        # Self-check (5 min)
```

### Experiment usage

All experiments support:

```bash
# Full run (recommended)
python exp_xxx.py

# Quick mode (concept check, < 2 min)
python exp_xxx.py --quick

# Help
python exp_xxx.py --help
```

Experiment results are saved under each module’s `experiments/results/` directory.

---

## 🎯 Design philosophy

### 1️⃣ Principles first, not command copying

- ❌ “Run this command and you’ll get a model”
- ✅ “Understand why the design works”

### 2️⃣ Validate with controlled experiments

Each design choice answers:
- **What breaks if we remove it?**
- **Why do other options fail?**

### 3️⃣ Progressive learning

- Single components → assembled architecture → full training
- Clear goals and validation at every step

### 4️⃣ Runs on a normal laptop

- Experiments use TinyShakespeare (1MB) or TinyStories (10–50MB)
- No GPU required (CPU/MPS works)
- Each experiment < 10 minutes

---

## 🛠️ Common tools

Shared tools live in `modules/common/`:

### data_sources.py - Dataset manager

```python
from modules.common.data_sources import get_experiment_data

# TinyShakespeare
text = get_experiment_data('shakespeare')

# TinyStories subset
texts = get_experiment_data('tinystories', size_mb=10)
```

### experiment_base.py - Experiment base class

```python
from modules.common.experiment_base import Experiment

class MyExperiment(Experiment):
    def run(self):
        # experiment code
        pass
```

### visualization.py - Visualization helpers

```python
from modules.common.visualization import (
    plot_attention_heatmap,
    plot_activation_distribution,
    plot_gradient_flow,
    plot_loss_curves
)
```

See docstrings in each file or [`modules/common/README.md`](../modules/common/README.md) for details.

#### ⚠️ Migration Notice

**2026-02**: `datasets.py` has been renamed to `data_sources.py` to avoid naming conflict with HuggingFace datasets library.

For detailed migration guide, see [modules/common/README.md](../modules/common/README.md) or [PR #20](https://github.com/joyehuang/minimind-notes/pull/20).

---

## 🤝 Contribution guide

Contributions welcome:
- New controlled experiments
- Better intuitive analogies
- Visualizations
- Bug fixes

Before submitting, please ensure:
- [ ] Experiments run independently
- [ ] Code has sufficient Chinese comments
- [ ] Results are reproducible (fixed random seeds)
- [ ] Follows the existing file structure

---

## 📜 Acknowledgements

This teaching module is based on [jingyaogong/minimind](https://github.com/jingyaogong/minimind).

All experiments link to real implementations in the upstream repository to help learners understand production-grade code.

---

## 📞 Related documents

- 📝 [Personal learning log](../docs/learning_log.md)
- 📚 [Knowledge base](../docs/knowledge_base.md)
- 🗺️ [Learning roadmap](../ROADMAP.md)
- 🏠 [Project home](../README.md)
