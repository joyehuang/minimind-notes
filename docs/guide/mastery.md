---
title: 深度掌握 (30+小时) | minimind从零理解llm训练
description: 从零训练完整 LLM。学习端到端的训练流程，包括数据准备、模型架构、训练策略和优化技巧。
keywords: LLM深度掌握, 完整训练流程, Tokenizer, 预训练, SFT, LoRA, RLHF
---

# 🎓 深度掌握（30+ 小时）

> 从零训练一个完整的 LLM

## 🎯 学习目标

完成 30 小时学习后，你将能够：
- ✅ 从零训练一个可用的 LLM
- ✅ 理解完整的训练流程
- ✅ 调试和优化训练过程
- ✅ 在自有数据上微调模型

## 📋 学习路径

### 第一周：基础（6 小时）

✅ 完成[📚 系统学习](/docs/guide/systematic)

---

### 第二周：数据准备（8 小时）

#### 1. 训练 Tokenizer（2 小时）

```bash
python scripts/train_tokenizer.py
```

**学习要点**：
- 理解 BPE 算法原理
- 训练自定义 Tokenizer

#### 2. 数据清洗与预处理（4 小时）

- 阅读 `dataset/lm_dataset.py`
- 理解数据 Packing 策略
- 创建自己的数据集

#### 3. 数据格式转换（2 小时）

- 预训练格式
- SFT 格式
- DPO 格式

**完成标准**：
- [ ] 训练一个自定义 Tokenizer
- [ ] 理解每个训练阶段的数据格式
- [ ] 准备好训练数据

---

### 第三周：模型训练（10 小时）

#### 1. 预训练（4 小时）

```bash
cd trainer
python train_pretrain.py \
    --data_path ../dataset/pretrain_hq.jsonl \
    --epochs 1 \
    --batch_size 32 \
    --hidden_size 512 \
    --num_hidden_layers 8
```

**关键要点**：
- 理解因果语言建模目标
- 监控训练曲线（loss、learning rate）
- 调试常见问题（NaN、OOM）

---

#### 2. 监督微调 SFT（3 小时）

```bash
python train_full_sft.py \
    --data_path ../dataset/sft_mini_512.jsonl \
    --from_weight pretrain
```

**关键要点**：
- 理解指令微调的作用
- 对比预训练和 SFT 后的模型行为

---

#### 3. LoRA 微调（3 小时）

```bash
python train_lora.py \
    --data_path ../dataset/lora_identity.jsonl \
    --from_weight full_sft
```

**关键要点**：
- 理解参数高效微调（PEFT）
- 学习 LoRA 的数学原理
- 领域适应策略

**完成标准**：
- [ ] 成功训练一个小模型（困惑度 < 3.0）
- [ ] 理解完整路径：预训练 → SFT → LoRA
- [ ] 能调试训练问题

---

### 第四周：进阶主题（6+ 小时）

#### 可选方向 1：RLHF / RLAIF（4 小时）

- DPO（直接偏好优化）
- PPO/GRPO（强化学习）

#### 可选方向 2：推理优化（2 小时）

- KV Cache
- Flash Attention
- 量化（INT8/INT4）

#### 可选方向 3：评估与分析（2 小时）

- C-Eval / MMLU 基准测试
- 错误分析
- 消融实验

---

## 🔗 参考资料

- 📖 [MiniMind 原项目](https://github.com/jingyaogong/minimind)
- 📝 [CLAUDE.md](/CLAUDE) — 完整命令参考
- 💻 [训练脚本](/trainer/) — 所有训练代码

---

## 📝 学习建议

### 1. 先实验，后理论

❌ 不要先把所有文档读完再动手
✅ 先跑实验建立直觉，再阅读理论加深理解

### 2. 对比学习

每个模块通过实验回答：
- **不这样做会怎样？**
- **其他方案为什么不行？**

### 3. 多轮迭代

- **第一遍**：浏览全局，建立整体认知
- **第二遍**：深入细节和数学推导
- **第三遍**：自己动手实现，巩固理解

### 4. 记录笔记

在[学习日志](/learning_log)中记录你的进度和思考

---

## 🎯 总检查清单

### 基础
- [ ] 完成系统学习（6 小时）

### 数据准备
- [ ] 训练自定义 Tokenizer
- [ ] 准备训练数据
- [ ] 理解数据格式

### 模型训练
- [ ] 完成预训练
- [ ] 完成 SFT
- [ ] 完成 LoRA

### 进阶主题
- [ ] 尝试 RLHF/RLAIF
- [ ] 优化推理性能
- [ ] 评估模型质量

---

准备好开始深度掌握之旅了吗？
