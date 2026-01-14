---
title: Learning Roadmap | MiniMind LLM Training
description: Three learning paths - Quick Start (30 min) / Systematic Study (6 hours) / Deep Mastery (30+ hours). Learn LLM training from scratch and prepare for interviews.
keywords: LLM learning roadmap, transformer tutorial, LLM interview prep
---

# MiniMind Learning Roadmap

> Three paths: Quick Start / Systematic Study / Deep Mastery

---

## 🎯 Choose your learning path

Pick the path that fits your time and goals:

| Path | Time | Goal | Best for |
|------|------|------|---------|
| [⚡ Quick Start](#quick-start-30-min) | 30 min | Understand core design choices | Quick intro to LLM training | 
| [📚 Systematic Study](#systematic-study-6-hours) | 6 hours | Master core components | Deeper understanding of Transformer | 
| [🎓 Deep Mastery](#deep-mastery-30-hours) | 30+ hours | Train a model from scratch | Full training pipeline mastery |

---

<a id="quick-start-30-min"></a>
## ⚡ Quick Start (30 min)

**Goal**: Understand key design choices in modern LLMs through 3 experiments.

### Environment setup (5 min)

```bash
# 1. Clone the repo
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes

# 2. Activate your virtual environment
source venv/bin/activate

# 3. Download experiment data (optional, some experiments do not need it)
cd modules/common
python datasets.py --download-all
cd ../..
```

---

### Experiment 1: Why normalization? (10 min)

**Question**: Why are deep networks hard to train?

```bash
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py
```

**You will see**:
- ❌ No normalization: activation std drops from 1.0 to 0.016 (vanishing gradients)
- ✅ RMSNorm: std stays stable around 1.0

**Key insight**: Normalization is the stabilizer for deep network training.

---

### Experiment 2: Why RoPE? (10 min)

**Question**: How does Transformer know token order?

```bash
cd ../../02-position-encoding/experiments
python exp1_rope_basics.py
```

**You will see**:
- Attention is permutation-invariant by itself (cannot distinguish order)
- RoPE encodes position by rotating vectors
- Relative position relationships emerge naturally

**Key insight**: RoPE provides absolute + relative position and supports length extrapolation.

---

### Experiment 3: How Attention works (10 min)

**Question**: What do Q, K, V mean?

```bash
cd ../../03-attention/experiments
python exp1_attention_basics.py
```

**You will see**:
- Query: what I want to find
- Key: what information I provide
- Value: what content I pass after matching
- Visualized attention weights

**Key insight**: Attention lets the model learn which tokens matter.

---

### 📊 After 30 minutes, you will understand

- ✅ Why modern LLMs use Pre-LN + RMSNorm
- ✅ Why RoPE is better than absolute position encoding
- ✅ The math and intuition of attention

**Next steps**:
- Want to go deeper? Continue to [📚 Systematic Study](#systematic-study-6-hours)
- Want theory? Read [teaching.md](modules/01-foundation/01-normalization/teaching.md)

---

<a id="systematic-study-6-hours"></a>
## 📚 Systematic Study (6 hours)

**Goal**: Master all core Transformer components.

### Learning path

#### Stage 1: Foundation - 5.5 hours

Study the four core modules in order:

**1. Normalization (1 hour)**
```bash
cd modules/01-foundation/01-normalization
```

What to do:
- 📖 Read [teaching.md](modules/01-foundation/01-normalization/teaching.md) (30 min)
- 🔬 Run all experiments (20 min)
  ```bash
  cd experiments
  bash run_all.sh
  ```
- 📝 Finish [quiz.md](modules/01-foundation/01-normalization/quiz.md) (10 min)

**Completion criteria**:
- [ ] Explain gradient vanishing/explosion
- [ ] Implement RMSNorm from scratch
- [ ] Understand Pre-LN vs Post-LN

---

**2. Position Encoding (1.5 hours)**
```bash
cd ../02-position-encoding
```

What to do:
- 📖 Read [teaching.md](modules/01-foundation/02-position-encoding/teaching.md) (40 min)
- 🔬 Run experiments 1-3 (40 min)
  ```bash
  cd experiments
  python exp1_rope_basics.py
  python exp2_multi_frequency.py
  python exp3_why_multi_frequency.py
  ```
- 📝 Self-check (10 min)

**Completion criteria**:
- [ ] Understand permutation invariance in Attention
- [ ] Explain RoPE rotation
- [ ] Understand multi-frequency roles

---

**3. Attention (2 hours)**
```bash
cd ../03-attention
```

What to do:
- 🔬 Run all experiments (1.5 hours)
  ```bash
  cd experiments
  python exp1_attention_basics.py
  python exp2_qkv_explained.py
  python exp3_multihead_attention.py
  ```
- 💻 Read source (30 min)
  - `model/model_minimind.py:250-330`

**Completion criteria**:
- [ ] Understand Q, K, V roles
- [ ] Understand multi-head benefits
- [ ] Understand GQA (Grouped Query Attention)

---

**4. FeedForward (1 hour)**
```bash
cd ../04-feedforward
```

What to do:
- 🔬 Run experiments (40 min)
  ```bash
  cd experiments
  python exp1_feedforward.py
  ```
- 💻 Understand SwiGLU activation (20 min)

**Completion criteria**:
- [ ] Understand FFN expand-compress pattern
- [ ] Understand Attention vs FFN roles
- [ ] Implement SwiGLU from scratch

---

#### Stage 2: Architecture - 0.5 hours

**What to do**:
- 📖 Read [02-architecture/README.md](modules/02-architecture/README.md) (30 min)
- Understand how components assemble into a Transformer block

**Completion criteria**:
- [ ] Draw the data flow of a Pre-LN Transformer block
- [ ] Understand residual connections
- [ ] Implement a Transformer block from scratch

---

### 📊 After 6 hours, you will be able to

- ✅ Understand all core Transformer components
- ✅ Explain design choices via experiments
- ✅ Implement a simple Transformer

**Next steps**:
- Want to train a model? Continue to [🎓 Deep Mastery](#deep-mastery-30-hours)
- Want to apply it? Refer to MiniMind training scripts

---

<a id="deep-mastery-30-hours"></a>
## 🎓 Deep Mastery (30+ hours)

**Goal**: Train a full LLM from scratch.

### Learning path

#### Week 1: Fundamentals (6 hours)
- ✅ Complete [📚 Systematic Study](#systematic-study-6-hours)

---

#### Week 2: Data preparation (8 hours)

**What to do**:
1. **Tokenizer training** (2 hours)
   ```bash
   python scripts/train_tokenizer.py
   ```
   - Understand BPE
   - Train a custom tokenizer

2. **Data cleaning and preprocessing** (4 hours)
   - Read `dataset/lm_dataset.py`
   - Understand packing strategies
   - Create your own dataset

3. **Data format conversion** (2 hours)
   - Pretrain format
   - SFT format
   - DPO format

**Completion criteria**:
- [ ] Train a custom tokenizer
- [ ] Understand data formats per stage
- [ ] Prepare training data

---

#### Week 3: Model training (10 hours)

**1. Pretraining (4 hours)**
```bash
cd trainer
python train_pretrain.py \
    --data_path ../dataset/pretrain_hq.jsonl \
    --epochs 1 \
    --batch_size 32 \
    --hidden_size 512 \
    --num_hidden_layers 8
```

**Key points**:
- Understand the causal language modeling objective
- Monitor training curves (loss, learning rate)
- Debug common issues (NaN, OOM)

---

**2. Supervised Fine-Tuning (3 hours)**
```bash
python train_full_sft.py \
    --data_path ../dataset/sft_mini_512.jsonl \
    --from_weight pretrain
```

**Key points**:
- Understand instruction tuning
- Compare pretrain vs SFT

---

**3. LoRA Fine-Tuning (3 hours)**
```bash
python train_lora.py \
    --data_path ../dataset/lora_identity.jsonl \
    --from_weight full_sft
```

**Key points**:
- Understand parameter-efficient fine-tuning (PEFT)
- Learn the math behind LoRA
- Domain adaptation strategies

**Completion criteria**:
- [ ] Train a small model successfully (perplexity < 3.0)
- [ ] Understand the full path: pretrain → SFT → LoRA
- [ ] Debug training issues

---

#### Week 4: Advanced topics (6+ hours)

**Optional tracks**:

1. **RLHF / RLAIF** (4 hours)
   - DPO (Direct Preference Optimization)
   - PPO/GRPO (Reinforcement Learning)

2. **Inference optimization** (2 hours)
   - KV cache
   - Flash Attention
   - Quantization (INT8/INT4)

3. **Evaluation and analysis** (2 hours)
   - C-Eval / MMLU benchmarks
   - Error analysis
   - Ablation studies

---

### 📊 After 30 hours, you will be able to

- ✅ Train a usable LLM from scratch
- ✅ Understand the full training pipeline
- ✅ Debug and optimize training
- ✅ Fine-tune on your own data

---

## 🛠️ Learning resources

### 📖 Documentation

- **Module teaching docs**: `modules/*/teaching.md`
- **Code guides**: `modules/*/code_guide.md`
- **Upstream docs**: refer to the original MiniMind README

### 🔬 Experiments

- **Visualization experiments**: no data needed, run fast (< 1 min)
- **Training experiments**: require data, verify behavior (< 10 min)
- **Full training**: use upstream scripts (hours)

### 💬 Community

- **Issues**: https://github.com/joyehuang/minimind-notes/issues
- **Upstream project**: https://github.com/jingyaogong/minimind

---

## 📝 Learning tips

### 1. Experiment first, then theory

- ❌ Do not read everything before running experiments
- ✅ Build intuition by running experiments first, then read theory

### 2. Learn by comparison

Each module answers through contrast:
- **What breaks if we don’t do this?**
- **Why do other approaches fail?**

### 3. Iterate in passes

- **First pass**: skim for the big picture
- **Second pass**: dive into details and math
- **Third pass**: implement yourself to solidify

### 4. Keep notes

Record your learning in `docs/`:
- `learning_log.md`: learning log
- `knowledge_base.md`: knowledge summaries

---

## 🎯 Checklist

### ⚡ Quick Start done

- [ ] Understand normalization
- [ ] Understand RoPE
- [ ] Understand attention

### 📚 Systematic Study done

- [ ] Finish the 4 Foundation modules
- [ ] Implement a Transformer block from scratch
- [ ] Pass all module quizzes

### 🎓 Deep Mastery done

- [ ] Train a pretrain model
- [ ] Finish SFT and LoRA
- [ ] Apply to your own data

---

**Ready to start?** Choose your path and begin. 🚀
