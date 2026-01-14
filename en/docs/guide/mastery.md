---
title: Deep Mastery (30+ hours) | MiniMind LLM Training
description: Train a full LLM from scratch. Learn the end-to-end pipeline including data prep, architecture, training strategies, and optimization.
keywords: LLM mastery, full training pipeline, tokenizer, pretraining, SFT, LoRA
---

# 🎓 Deep Mastery (30+ hours)

> Train a complete LLM from scratch

## 🎯 Learning goals

After 30 hours you will be able to:
- ✅ Train a usable LLM from scratch
- ✅ Understand the full training pipeline
- ✅ Debug and optimize training runs
- ✅ Fine-tune models on your own data

## 📋 Learning path

### Week 1: Fundamentals (6 hours)

✅ Complete [📚 Systematic Study](/en/docs/guide/systematic)

---

### Week 2: Data preparation (8 hours)

#### 1. Tokenizer training (2 hours)

```bash
python scripts/train_tokenizer.py
```

**What to learn**:
- Understand the BPE algorithm
- Train a custom tokenizer

#### 2. Data cleaning and preprocessing (4 hours)

- Read `dataset/lm_dataset.py`
- Understand packing strategies
- Create your own dataset

#### 3. Data format conversion (2 hours)

- Pretrain format
- SFT format
- DPO format

**Completion criteria**:
- [ ] Train a custom tokenizer
- [ ] Understand data formats for each training stage
- [ ] Prepare training data

---

### Week 3: Model training (10 hours)

#### 1. Pretraining (4 hours)

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

#### 2. Supervised Fine-Tuning (3 hours)

```bash
python train_full_sft.py \
    --data_path ../dataset/sft_mini_512.jsonl \
    --from_weight pretrain
```

**Key points**:
- Understand the role of instruction tuning
- Compare pretrain vs SFT behavior

---

#### 3. LoRA Fine-Tuning (3 hours)

```bash
python train_lora.py \
    --data_path ../dataset/lora_identity.jsonl \
    --from_weight full_sft
```

**Key points**:
- Understand parameter-efficient fine-tuning (PEFT)
- Learn the math behind LoRA
- Domain adaptation strategy

**Completion criteria**:
- [ ] Train a small model successfully (perplexity < 3.0)
- [ ] Understand the full path: pretrain → SFT → LoRA
- [ ] Debug training issues

---

### Week 4: Advanced topics (6+ hours)

#### Optional track 1: RLHF / RLAIF (4 hours)

- DPO (Direct Preference Optimization)
- PPO/GRPO (Reinforcement Learning)

#### Optional track 2: Inference optimization (2 hours)

- KV cache
- Flash Attention
- Quantization (INT8/INT4)

#### Optional track 3: Evaluation and analysis (2 hours)

- C-Eval / MMLU benchmarks
- Error analysis
- Ablation studies

---

## 🔗 References

- 📖 [MiniMind upstream repo](https://github.com/jingyaogong/minimind)
- 📝 [CLAUDE.md](/CLAUDE) - full command reference
- 💻 [Training scripts](/trainer/) - all training code

---

## 📝 Learning tips

### 1. Experiment first, then theory

❌ Do not read all the docs before trying things
✅ Run experiments first to build intuition, then read theory

### 2. Learn by comparison

Each module answers via experiments:
- **What breaks if we don’t do this?**
- **Why do other options fail?**

### 3. Iterate in passes

- **First pass**: skim to grasp the big picture
- **Second pass**: go deep on details and math
- **Third pass**: implement yourself to solidify understanding

### 4. Keep notes

Record your progress in [Learning Log](/learning_log)

---

## 🎯 Checklist

### Fundamentals
- [ ] Complete Systematic Study (6 hours)

### Data preparation
- [ ] Train a custom tokenizer
- [ ] Prepare training data
- [ ] Understand data formats

### Model training
- [ ] Finish Pretrain
- [ ] Finish SFT
- [ ] Finish LoRA

### Advanced topics
- [ ] Try RLHF/RLAIF
- [ ] Optimize inference performance
- [ ] Evaluate model quality

---

Ready to begin the deep mastery journey? 🚀
