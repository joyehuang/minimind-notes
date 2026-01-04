---
title: 学习路线图
description: 三条学习路径 - 快速体验/系统学习/深度掌握
---

# MiniMind 学习路线图

> 三条路径：快速体验 / 系统学习 / 深度掌握

---

## 🎯 选择你的学习路径

根据你的时间和目标，选择合适的学习路径：

| 路径 | 时长 | 目标 | 适合人群 |
|------|------|------|---------|
| [⚡ 快速体验](#-快速体验-30-分钟) | 30 分钟 | 理解核心设计选择 | 想快速了解 LLM 训练原理 |
| [📚 系统学习](#-系统学习-6-小时) | 6 小时 | 掌握基础组件 | 想深入理解 Transformer |
| [🎓 深度掌握](#-深度掌握-30-小时) | 30+ 小时 | 从零训练模型 | 想完整掌握 LLM 训练流程 |

---

## ⚡ 快速体验（30 分钟）

**目标**：通过 3 个关键实验，理解现代 LLM 的核心设计选择

### 准备环境（5 分钟）

```bash
# 1. 克隆仓库
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 下载实验数据（可选，部分实验不需要）
cd modules/common
python datasets.py --download-all
cd ../..
```

---

### 实验 1：为什么需要归一化？（10 分钟）

**问题**：深层网络为什么难训练？

```bash
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py
```

**你会看到**：
- ❌ 无归一化：激活标准差从 1.0 衰减到 0.016（梯度消失）
- ✅ 有 RMSNorm：标准差保持稳定在 1.0 附近

**关键发现**：归一化是深层网络训练的"稳定器"

---

### 实验 2：为什么用 RoPE 位置编码？（10 分钟）

**问题**：Transformer 如何知道词的顺序？

```bash
cd ../../02-position-encoding/experiments
python exp1_rope_basics.py
```

**你会看到**：
- Attention 本身是"排列不变"的（无法区分顺序）
- RoPE 通过旋转向量编码位置信息
- 自动产生相对位置关系

**关键发现**：RoPE 既有绝对位置，又有相对位置，还支持长度外推

---

### 实验 3：Attention 如何工作？（10 分钟）

**问题**：Attention 的 Q、K、V 是什么意思？

```bash
cd ../../03-attention/experiments
python exp1_attention_basics.py
```

**你会看到**：
- Query（查询）：我想找什么信息？
- Key（键）：我提供什么信息？
- Value（值）：找到后传递什么内容？
- 注意力权重可视化

**关键发现**：Attention 让模型自动学习"哪些词相关"

---

### 📊 30 分钟后你将理解

- ✅ 为什么现代 LLM 用 Pre-LN + RMSNorm
- ✅ 为什么 RoPE 比绝对位置编码好
- ✅ Attention 的数学原理和直觉

**下一步**：
- 想深入？继续 [📚 系统学习](#-系统学习-6-小时)
- 想实践？阅读 [teaching.md](modules/01-foundation/01-normalization/teaching.md) 理解理论

---

## 📚 系统学习（6 小时）

**目标**：完整掌握 Transformer 的所有基础组件

### 学习路径

#### 阶段 1：Foundation（基础组件）- 5.5 小时

按顺序学习 4 个核心模块：

**1. Normalization（1 小时）**
```bash
cd modules/01-foundation/01-normalization
```

学习内容：
- 📖 阅读 [teaching.md](modules/01-foundation/01-normalization/teaching.md)（30 分钟）
- 🔬 运行所有实验（20 分钟）
  ```bash
  cd experiments
  bash run_all.sh
  ```
- 📝 完成 [quiz.md](modules/01-foundation/01-normalization/quiz.md)（10 分钟）

**完成标准**：
- [ ] 能解释梯度消失/爆炸问题
- [ ] 能从零实现 RMSNorm
- [ ] 理解 Pre-LN vs Post-LN 的区别

---

**2. Position Encoding（1.5 小时）**
```bash
cd ../02-position-encoding
```

学习内容：
- 📖 阅读 [teaching.md](modules/01-foundation/02-position-encoding/teaching.md)（40 分钟）
- 🔬 运行实验 1-3（40 分钟）
  ```bash
  cd experiments
  python exp1_rope_basics.py
  python exp2_multi_frequency.py
  python exp3_why_multi_frequency.py
  ```
- 📝 自测（10 分钟）

**完成标准**：
- [ ] 理解 Attention 的排列不变性
- [ ] 能解释 RoPE 的旋转原理
- [ ] 理解多频率机制的作用

---

**3. Attention（2 小时）**
```bash
cd ../03-attention
```

学习内容：
- 🔬 运行所有实验（1.5 小时）
  ```bash
  cd experiments
  python exp1_attention_basics.py
  python exp2_qkv_explained.py
  python exp3_multihead_attention.py
  ```
- 💻 阅读源码（30 分钟）
  - `model/model_minimind.py:250-330`

**完成标准**：
- [ ] 理解 Q、K、V 的作用
- [ ] 理解 Multi-Head 的优势
- [ ] 理解 GQA（Grouped Query Attention）

---

**4. FeedForward（1 小时）**
```bash
cd ../04-feedforward
```

学习内容：
- 🔬 运行实验（40 分钟）
  ```bash
  cd experiments
  python exp1_feedforward.py
  ```
- 💻 理解 SwiGLU 激活函数（20 分钟）

**完成标准**：
- [ ] 理解 FFN 的扩张-压缩机制
- [ ] 理解 Attention vs FFN 的分工
- [ ] 能从零实现 SwiGLU

---

#### 阶段 2：Architecture（架构组装）- 0.5 小时

**学习内容**：
- 📖 阅读 [02-architecture/README.md](modules/02-architecture/README.md)（30 分钟）
- 理解如何将组件组装成 Transformer Block

**完成标准**：
- [ ] 能画出 Pre-LN Transformer Block 的数据流图
- [ ] 理解残差连接的作用
- [ ] 能从零实现一个 Transformer Block

---

### 📊 6 小时后你将掌握

- ✅ Transformer 的所有基础组件
- ✅ 每个设计选择的原因（通过对照实验）
- ✅ 从零实现一个简单的 Transformer

**下一步**：
- 想训练模型？继续 [🎓 深度掌握](#-深度掌握-30-小时)
- 想应用？参考原 MiniMind 仓库的训练脚本

---

## 🎓 深度掌握（30+ 小时）

**目标**：从零训练一个完整的 LLM

### 学习路径

#### 第 1 周：理论基础（6 小时）
- ✅ 完成 [📚 系统学习](#-系统学习-6-小时)

---

#### 第 2 周：数据准备（8 小时）

**学习内容**：
1. **Tokenizer 训练**（2 小时）
   ```bash
   python scripts/train_tokenizer.py
   ```
   - 理解 BPE 算法
   - 训练自定义 tokenizer

2. **数据清洗和预处理**（4 小时）
   - 阅读 `dataset/lm_dataset.py`
   - 理解数据打包（packing）策略
   - 创建自己的数据集

3. **数据格式转换**（2 小时）
   - Pretrain 格式
   - SFT 格式
   - DPO 格式

**完成标准**：
- [ ] 能训练一个自定义 tokenizer
- [ ] 理解不同训练阶段的数据格式
- [ ] 准备好训练数据

---

#### 第 3 周：模型训练（10 小时）

**1. Pretraining（4 小时）**
```bash
cd trainer
python train_pretrain.py \
    --data_path ../dataset/pretrain_hq.jsonl \
    --epochs 1 \
    --batch_size 32 \
    --hidden_size 512 \
    --num_hidden_layers 8
```

**学习要点**：
- 理解 Causal Language Modeling 目标
- 监控训练曲线（loss、learning rate）
- 调试常见问题（NaN、OOM）

---

**2. Supervised Fine-Tuning（3 小时）**
```bash
python train_full_sft.py \
    --data_path ../dataset/sft_mini_512.jsonl \
    --from_weight pretrain
```

**学习要点**：
- 理解指令微调的作用
- 对比 pretrain vs SFT 的效果

---

**3. LoRA Fine-Tuning（3 小时）**
```bash
python train_lora.py \
    --data_path ../dataset/lora_identity.jsonl \
    --from_weight full_sft
```

**学习要点**：
- 理解参数高效微调（PEFT）
- LoRA 的数学原理
- 领域适配策略

**完成标准**：
- [ ] 成功训练一个小模型（困惑度 < 3.0）
- [ ] 理解 pretrain → SFT → LoRA 的完整流程
- [ ] 能调试训练问题

---

#### 第 4 周：进阶主题（6+ 小时）

**可选方向**：

1. **RLHF / RLAIF**（4 小时）
   - DPO（Direct Preference Optimization）
   - PPO/GRPO（Reinforcement Learning）

2. **推理优化**（2 小时）
   - KV Cache
   - Flash Attention
   - 量化（INT8/INT4）

3. **评估和分析**（2 小时）
   - C-Eval / MMLU 基准测试
   - 错误分析
   - 消融实验

---

### 📊 30 小时后你将能够

- ✅ 从零训练一个可用的 LLM
- ✅ 理解完整的训练流程
- ✅ 调试和优化模型
- ✅ 在自己的数据上微调模型

---

## 🛠️ 学习资源

### 📖 文档

- **模块教学文档**：`modules/*/teaching.md`
- **代码导读**：`modules/*/code_guide.md`
- **原仓库文档**：参考 MiniMind 原始 README

### 🔬 实验代码

- **可视化实验**：无需数据，快速运行（< 1 分钟）
- **训练实验**：需要数据，验证效果（< 10 分钟）
- **完整训练**：原仓库训练脚本（数小时）

### 💬 社区

- **问题反馈**：[GitHub Issues](https://github.com/joyehuang/minimind-notes/issues)
- **原项目**：[MiniMind](https://github.com/jingyaogong/minimind)

---

## 📝 学习建议

### 1. 先实验，再理论

- ❌ 不要先读完所有文档再动手
- ✅ 先跑实验，建立直觉，再看理论

### 2. 对比学习

每个模块都通过对比实验回答：
- **不这样做会怎样？**
- **其他方案为什么不行？**

### 3. 循序渐进

- **第一遍**：快速过一遍，理解大框架
- **第二遍**：深入细节，理解数学原理
- **第三遍**：自己实现，巩固理解

### 4. 记录笔记

在 `docs/` 文件夹记录你的学习过程：
- `learning_log.md`：学习日志
- `knowledge_base.md`：知识点整理

---

## 🎯 检查清单

### ⚡ 快速体验完成

- [ ] 理解归一化的作用
- [ ] 理解 RoPE 的原理
- [ ] 理解 Attention 机制

### 📚 系统学习完成

- [ ] 完成 Foundation 4 个模块
- [ ] 能从零实现 Transformer Block
- [ ] 通过所有模块的自测题

### 🎓 深度掌握完成

- [ ] 训练一个 pretrain 模型
- [ ] 完成 SFT 和 LoRA 微调
- [ ] 在自己的数据上应用

---

**准备好了吗？** 选择你的路径，开始学习吧！🚀
