# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Learning Notes Maintenance

**IMPORTANT**: This repository uses a **three-tier note system** for organized learning:

### Note System Structure

```
minimind/
├── notes.md                ← 总索引（entry point）
├── learning_log.md         ← 学习日志（chronological）
├── knowledge_base.md       ← 知识库（topical）
└── learning_materials/     ← 可执行示例代码
```

### 1. notes.md - 总索引
- **Purpose**: Central index and navigation hub
- **Content**:
  - Links to other documents
  - Current progress overview
  - Quick reference table
  - File structure diagram
- **Update**: When structure changes or new sections are added

### 2. learning_log.md - 学习日志
- **Purpose**: Chronological record of learning journey
- **Content**:
  - Date-stamped entries (format: `### 2025-MM-DD: Topic`)
  - Daily completed tasks (✅ checkbox list)
  - Problems encountered and solutions (🐛 section)
  - Personal thoughts and reflections (💭 section)
  - Learning plans for next session
- **Update**: At the end of each learning session
- **Format Example**:
  ```markdown
  ### 2025-11-07: 深度理解 Transformer 核心组件

  #### ✅ 完成事项
  - [x] 理解 RMSNorm 原理
  - [x] 理解 RoPE 位置编码

  #### 💭 个人思考
  - **收获**: ...
  - **疑问解答**: ...
  ```

### 3. knowledge_base.md - 知识库
- **Purpose**: Systematic knowledge organization by topic
- **Content**:
  - Technical concepts and principles (numbered sections)
  - Comparison tables (e.g., RMSNorm vs LayerNorm)
  - Mathematical formulas
  - Code snippets with explanations
  - Q&A records (separate section at bottom)
- **Update**: When new concepts are learned or questions are answered
- **Structure**:
  ```markdown
  ## 1. Topic Name
  ### 1.1 Subtopic
  [Detailed explanation]

  ## 问答记录
  ### Q: Question?
  **A**: Answer
  ```

### 4. learning_materials/ - 学习辅助材料
- **Purpose**: Executable code examples for hands-on learning
- **Content**:
  - Python files demonstrating concepts
  - README.md with usage instructions
  - Organized by topic (normalization, position encoding, attention)
- **Update**: When creating new learning examples
- **Naming**: Descriptive names like `rope_basics.py`, `why_normalization.py`

### Update Workflow

**IMPORTANT**: Every conversation round with new content MUST update the notes system.

**After each learning session OR after answering user questions**:

1. **Update learning_log.md**:
   - Add new date section (if new day)
   - Add new subsection for additional learning within the same day
   - List completed tasks
   - Record problems and solutions
   - Write personal reflections
   - **Record user questions and answers**

2. **Update knowledge_base.md**:
   - Add new knowledge sections
   - **Add Q&A records for ALL user questions** (even follow-up questions)
   - Add comparison tables if needed
   - Number questions sequentially (Q1, Q2, Q3...)
   - Mark particularly important questions with ⭐️

3. **Update notes.md**:
   - Update progress indicator
   - Add new date to "按日期查找"
   - Update file structure if needed

4. **Create learning materials** (if applicable):
   - Write executable examples
   - Update learning_materials/README.md
   - Add references in learning_log.md

5. **Update learning_materials/README.md** (if new files created):
   - Add new file descriptions
   - Update recommended learning order
   - Mark important files with ⭐️

### When to Update Notes

Update notes in these scenarios:

1. ✅ **After teaching a new concept**
   - Add to knowledge_base.md
   - Add to learning_log.md

2. ✅ **After answering user questions**
   - Add Q&A to knowledge_base.md
   - Add reflection to learning_log.md
   - **Even if it's a follow-up question in the same conversation**

3. ✅ **After solving a problem**
   - Add to learning_log.md (problems section)

4. ✅ **After creating learning materials**
   - Update all three files
   - Add file references

5. ✅ **User explicitly requests note updates**
   - Follow user's guidance on what to record

### Notes Update Checklist

Before ending a conversation, ensure:
- [ ] All user questions have Q&A entries in knowledge_base.md
- [ ] New learning has date-stamped entry in learning_log.md
- [ ] New files are listed in learning_materials/README.md
- [ ] Question numbers are updated sequentially
- [ ] Important discoveries are marked with ⭐️

**Quick Reference**: See `NOTE_UPDATE_GUIDE.md` for detailed templates and examples.

### Interactive Learning Approach

- The user prefers to learn at a **slower pace with deep understanding**
- Use **dialogue and questions** to help clarify concepts before moving forward
- For each knowledge point, organize the Q&A discussion into knowledge_base.md
- Don't rush through multiple concepts - **focus on one at a time** until the user fully understands
- Create executable examples in learning_materials/ to demonstrate concepts
- Always ask if the user is ready to continue before moving to the next topic

### Key Principles

1. **Separation of Concerns**:
   - Chronological (learning_log.md) vs Topical (knowledge_base.md)
   - Theory (knowledge_base.md) vs Practice (learning_materials/)

2. **Easy Navigation**:
   - notes.md provides quick links to all sections
   - Clear table of contents in each document

3. **No Information Loss**:
   - When reorganizing, preserve all content
   - Move, don't delete

4. **User-Friendly**:
   - Clear headings and formatting
   - Emoji markers for quick scanning (✅ ❌ 💡 🐛 💭 etc.)
   - Code examples with explanations

## Git Workflow for Learning Notes

**IMPORTANT**: After completing each learning session, commit and push notes to the remote repository.

### Remote Repository Setup

This repository has two remotes:
- `origin`: Main MiniMind project (https://github.com/jingyaogong/minimind.git)
- `notes`: Personal learning notes backup (https://github.com/joyehuang/minimind-notes.git)

### Commit Workflow

**After each learning session:**

1. **Stage note files**:
   ```bash
   git add notes.md learning_log.md knowledge_base.md learning_materials/
   ```

2. **Commit with concise message**:
   ```bash
   # Use simple, descriptive commit messages
   # Examples:
   git commit -m "学习 RMSNorm 归一化原理"
   git commit -m "理解 RoPE 多频率机制"
   git commit -m "添加 Attention 学习材料"
   git commit -m "完成环境搭建和首次运行"
   ```

3. **Push to notes remote**:
   ```bash
   git push notes master
   ```

### Commit Message Guidelines

- **DO**: Use concise, descriptive Chinese messages (one sentence)
- **DO**: Focus on what was learned (e.g., "学习 Attention 机制原理")
- **DON'T**: Include generic phrases like "Generated with Claude Code"
- **DON'T**: Include emojis or formatting in commit messages
- **DON'T**: Make multi-paragraph commit messages

### When to Commit

Commit after:
1. Completing a major concept (e.g., after learning RMSNorm)
2. Adding new learning materials (e.g., new .py examples)
3. Solving a significant problem (documented in learning_log.md)
4. End of each learning session (even if work is in progress)

### Example Workflow

```bash
# After learning session on Attention
git add notes.md learning_log.md knowledge_base.md learning_materials/
git commit -m "学习 Attention 注意力机制基础"
git push notes master
```

### Important Notes

- **DO NOT** commit generated model weights, datasets, or cache files
- **DO NOT** push to `origin` (main MiniMind repo) - only push to `notes`
- All commits should only include learning note files:
  - `notes.md`
  - `learning_log.md`
  - `knowledge_base.md`
  - `learning_materials/*.py`
  - `learning_materials/README.md`
  - `CLAUDE.md` (when updating guidelines)
  - `NOTE_UPDATE_GUIDE.md` (when updating templates)

## Project Overview

**MiniMind** is an educational implementation of a complete large language model (LLM) training pipeline from scratch. The project aims to train ultra-small language models (starting at just 25.8M parameters) using minimal resources (3 RMB + 2 hours on a single NVIDIA 3090 GPU).

Key differentiators:
- All core algorithms implemented from scratch using PyTorch (not abstracted behind third-party libraries)
- Complete training pipeline: tokenizer training, pretraining, supervised fine-tuning (SFT), LoRA, RLHF (DPO), RLAIF (PPO/GRPO/SPO), and model distillation
- Compatible with transformers, trl, peft, and third-party inference engines (llama.cpp, vllm, ollama)
- Supports both Dense and MoE (Mixture of Experts) architectures
- Includes distilled reasoning model capabilities (MiniMind-Reason, inspired by DeepSeek-R1)

## Core Architecture

### Model Structure

The codebase implements two main architectures:

1. **MiniMind-Dense**: Transformer Decoder-Only architecture similar to Llama3.1
   - Pre-normalization with RMSNorm on inputs (not outputs)
   - SwiGLU activation function (instead of ReLU)
   - Rotary Position Embeddings (RoPE) instead of absolute position embeddings
   - Supports YaRN algorithm for long-context extrapolation

2. **MiniMind-MoE**: Mixture of Experts based on DeepSeek-V2/V3
   - Shared + routed expert architecture
   - Fine-grained expert splitting
   - Load balancing loss for expert utilization

### Directory Structure

```
minimind/
├── model/                    # Model implementations
│   ├── model_minimind.py    # Main MiniMindConfig and MiniMindForCausalLM
│   └── model_lora.py        # LoRA implementation from scratch
├── dataset/                  # Dataset handling
│   └── lm_dataset.py        # Dataset classes for all training stages
├── trainer/                  # Training scripts (all stages)
│   ├── train_pretrain.py    # Pretraining
│   ├── train_full_sft.py    # Supervised fine-tuning
│   ├── train_lora.py        # LoRA fine-tuning
│   ├── train_dpo.py         # Direct Preference Optimization (RLHF)
│   ├── train_ppo.py         # Proximal Policy Optimization (RLAIF)
│   ├── train_grpo.py        # Group Relative Policy Optimization (RLAIF)
│   ├── train_spo.py         # Simple Policy Optimization (RLAIF)
│   ├── train_distillation.py      # White-box distillation
│   ├── train_distill_reason.py    # Reasoning model distillation (R1-style)
│   └── trainer_utils.py     # Shared training utilities
├── scripts/                  # Inference and utilities
│   ├── train_tokenizer.py   # Custom tokenizer training
│   ├── serve_openai_api.py  # OpenAI-compatible API server
│   ├── web_demo.py          # Streamlit web UI
│   └── convert_model.py     # Model format conversion
└── eval_llm.py              # Model evaluation and chat interface
```

## Commands Reference

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

### Testing Existing Models

```bash
# Download model (choose one)
git clone https://huggingface.co/jingyaogong/MiniMind2
# or
git clone https://www.modelscope.cn/models/gongjy/MiniMind2

# Command-line chat interface
python eval_llm.py --load_from ./MiniMind2

# Start web UI (requires streamlit)
streamlit run scripts/web_demo.py

# Third-party inference
ollama run jingyaogong/minimind2
vllm serve ./MiniMind2/ --served-model-name "minimind"
```

### Training Pipeline (all commands run from `./trainer` directory)

**Important**: All training scripts should be executed from the `./trainer` directory:
```bash
cd trainer
```

#### 1. Pretraining
```bash
# Single GPU
python train_pretrain.py

# Multi-GPU (DDP)
torchrun --nproc_per_node N train_pretrain.py

# Common arguments:
# --data_path ../dataset/pretrain_hq.jsonl
# --epochs 1
# --batch_size 32
# --learning_rate 5e-4
# --max_seq_len 512
# --hidden_size 512        # or 768 for larger model
# --num_hidden_layers 8    # or 16 for larger model
# --use_moe 0             # 1 to enable MoE
# --use_wandb             # Enable wandb/swanlab logging
```

Output: `../out/pretrain_*.pth`

#### 2. Supervised Fine-Tuning (SFT)
```bash
# Single GPU
python train_full_sft.py

# Multi-GPU
torchrun --nproc_per_node N train_full_sft.py

# Common arguments (similar to pretrain):
# --data_path ../dataset/sft_mini_512.jsonl
# --from_weight pretrain  # Load pretrained weights
```

Output: `../out/full_sft_*.pth`

#### 3. LoRA Fine-Tuning
```bash
python train_lora.py

# Common arguments:
# --data_path ../dataset/lora_identity.jsonl  # or lora_medical.jsonl
# --from_weight full_sft  # Base model to add LoRA to
# --lora_r 8              # LoRA rank
# --lora_alpha 16
```

Output: `../out/lora/lora_*_*.pth`

Test LoRA:
```bash
cd ..
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

#### 4. RLHF: Direct Preference Optimization (DPO)
```bash
python train_dpo.py

# Common arguments:
# --data_path ../dataset/dpo.jsonl
# --from_weight full_sft
# --beta 0.1  # KL penalty coefficient
```

Output: `../out/dpo_*.pth`

#### 5. RLAIF: PPO/GRPO/SPO

**Prerequisites**:
- Download reward model to sibling directory:
```bash
cd ../..  # Go to parent of minimind
git clone https://modelscope.cn/Shanghai_AI_Laboratory/internlm2-1_8b-reward.git
# or
git clone https://huggingface.co/internlm/internlm2-1_8b-reward
```

```bash
cd minimind/trainer

# PPO (Proximal Policy Optimization)
python train_ppo.py

# GRPO (Group Relative Policy Optimization)
python train_grpo.py

# SPO (Simple Policy Optimization)
python train_spo.py

# Common arguments:
# --data_path ../dataset/rlaif-mini.jsonl
# --from_weight dpo
# --reward_model_path ../../internlm2-1_8b-reward
```

Output: `../out/ppo_*.pth`, `../out/grpo_*.pth`, `../out/spo_*.pth`

#### 6. Reasoning Model Training (R1-style)
```bash
python train_distill_reason.py

# Common arguments:
# --data_path ../dataset/r1_mix_1024.jsonl
# --from_weight dpo  # Usually based on RLHF model
# --max_seq_len 1024  # Match data max length
```

Output: `../out/reason_*.pth`

The reasoning model uses special tags:
- `<think>思考过程</think>` for chain-of-thought reasoning
- `<answer>最终回答</answer>` for final response

#### 7. White-box Distillation
```bash
python train_distillation.py

# This is primarily for educational reference
# Requires teacher model of same architecture
```

### Training Resumption

All training scripts support checkpoint resumption:
- Checkpoints saved in `../checkpoints/` directory
- Use `--resume` flag to continue from last checkpoint
- Supports cross-GPU resumption (can change number of GPUs)
- Wandb/SwanLab logging continuity maintained

### Monitoring Training

```bash
# Enable wandb (requires VPN outside China)
# Login first: wandb login
python train_*.py --use_wandb

# Enable SwanLab (China-friendly, API compatible with wandb)
# Modify import in training script: import swanlab as wandb
python train_*.py --use_wandb
```

### Running Tests

The project doesn't have traditional unit tests, but you can evaluate models on benchmarks:
- C-Eval
- C-MMLU
- OpenBookQA

Refer to third-party evaluation frameworks or the README for detailed benchmark instructions.

## Key Implementation Details

### Model Configuration

Key parameters in `model/model_minimind.py`:

```python
MiniMindConfig(
    hidden_size=512,           # 512 for small, 768 for base
    num_hidden_layers=8,       # 8 for small, 16 for base
    num_attention_heads=8,
    num_key_value_heads=2,     # GQA (Grouped Query Attention)
    vocab_size=6400,           # Custom minimind tokenizer
    rope_theta=1000000.0,      # RoPE base frequency
    max_position_embeddings=32768,
    use_moe=False,             # Enable for MoE variant
    n_routed_experts=4,        # MoE: number of experts
    n_shared_experts=1,        # MoE: shared experts
    num_experts_per_tok=2,     # MoE: top-k routing
    flash_attn=True,           # Use Flash Attention
    inference_rope_scaling=False,  # YaRN long-context extrapolation
)
```

### Dataset Formats

**Pretrain** (`pretrain_hq.jsonl`):
```json
{"text": "如何才能摆脱拖延症？ 治愈拖延症并不容易，但以下建议可能有所帮助..."}
```

**SFT** (`sft_*.jsonl`):
```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "再见"},
    {"role": "assistant", "content": "再见！"}
  ]
}
```

**DPO** (`dpo.jsonl`):
```json
{
  "chosen": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "good answer"}
  ],
  "rejected": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "bad answer"}
  ]
}
```

**RLAIF** (`rlaif-mini.jsonl`):
```json
{
  "conversations": [
    {"role": "user", "content": "请解释一下什么是光合作用？"},
    {"role": "assistant", "content": "无"}
  ]
}
```
Note: Assistant content is ignored during RLAIF training (model generates responses on-policy).

**Reasoning** (`r1_mix_1024.jsonl`): Same as SFT format, but assistant content uses `<think>...</think><answer>...</answer>` tags.

### Tokenizer

- Custom tokenizer with 6400 vocab size (minimind_tokenizer)
- Located in `./model/tokenizer.json` and `./model/tokenizer_config.json`
- To train new tokenizer: `python scripts/train_tokenizer.py` (usually not needed)
- Uses special tokens: `<|im_start|>`, `<|im_end|>`, `<think>`, `<answer>`, `<tool_call>`, etc.

### Model Weights Naming Convention

Saved weights follow pattern: `{stage}_{dimension}[_moe].pth`
- `pretrain_512.pth` - Pretrained small model
- `full_sft_768.pth` - SFT'd base model
- `dpo_512.pth` - DPO small model
- `reason_768.pth` - Reasoning base model
- `pretrain_640_moe.pth` - MoE variant

### Chat Template

The model uses a chat template similar to ChatML:
```
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>
```

For reasoning models:
```
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>
{reasoning_process}
</think>
<answer>
{final_answer}
</answer><|im_end|>
```

## Training Tips

1. **Quick Start (Fastest)**: Use `pretrain_hq.jsonl` + `sft_mini_512.jsonl` for a functional chatbot in ~2 hours on a single 3090.

2. **Better Quality**: Use full dataset combination: `pretrain_hq.jsonl` + `sft_512.jsonl` + `sft_2048.jsonl` + `dpo.jsonl` (~38-122 hours depending on model size).

3. **Model Size vs Depth**: For small models (<1B), "deep and narrow" architectures perform better than "wide and shallow". Prefer increasing `num_hidden_layers` over `hidden_size` when the same parameter count.

4. **Gradient Accumulation**: Use `--accumulation_steps` to simulate larger batch sizes with limited VRAM.

5. **Mixed Precision**: Default dtype is `bfloat16` (change with `--dtype`). Use autocast context for memory efficiency.

6. **Checkpoint Frequency**: Default save interval is 100 steps. Adjust with `--save_interval` based on dataset size.

7. **Long Context**: For sequences >512, use YaRN by setting `inference_rope_scaling=True` in config. Train with longer `max_seq_len` using `sft_1024.jsonl` or `sft_2048.jsonl`.

8. **LoRA for Domain Adaptation**: When adapting to specific domains (medical, legal, etc.), use LoRA to avoid catastrophic forgetting. Mix domain data with general SFT data.

9. **Reasoning Models**: When training R1-style models, increase loss weight on special tokens (`<think>`, `<answer>`) to enforce format compliance (see `train_distill_reason.py`).

10. **RLAIF Reward Sparsity**: For very small models, use continuous reward signals (reward models) rather than binary rule-based rewards to avoid sparse gradient problems.

## Compatibility Notes

- Models trained after 2025-04-26 use updated naming convention aligned with transformers
- Old models (minimind-v1 series) are no longer maintained
- The codebase supports both checkpoint formats (`.pth`) and transformers format
- Use `scripts/convert_model.py` to convert between formats

## API Server

Start OpenAI-compatible API server:
```bash
cd scripts
python serve_openai_api.py --load_from ../MiniMind2

# Test with client:
python chat_openai_api.py
```

Default endpoint: `http://localhost:8000/v1/chat/completions`

## Important File Locations

- **Model outputs**: `./out/` directory (create if not exists)
- **Checkpoints**: `./checkpoints/` directory (for resumption)
- **Datasets**: `./dataset/` directory (download from ModelScope/HuggingFace)
- **Tokenizer**: `./model/tokenizer.json`
- **Model implementations**: `./model/model_minimind.py` and `./model/model_lora.py`

## Recommended Training Sequence

For a complete model from scratch:

1. **Pretrain** → `pretrain_*.pth`
2. **SFT** (load pretrain weights) → `full_sft_*.pth`
3. **DPO** (load SFT weights) → `dpo_*.pth`
4. **GRPO/PPO** (load DPO weights) → `grpo_*.pth` / `ppo_*.pth`
5. (Optional) **Reasoning distillation** (load RLHF weights) → `reason_*.pth`

Alternative for domain-specific models:
1. Start from pretrained/SFT checkpoint
2. Apply **LoRA** with domain data → `lora_*_*.pth`
3. Inference with base + LoRA weights

## Common Issues

- **CUDA Out of Memory**: Reduce `batch_size`, increase `accumulation_steps`, or reduce `max_seq_len`
- **Model generates nonsense**: Ensure using correct tokenizer and chat template
- **Training loss not decreasing**: Check learning rate schedule, verify data format, ensure `from_weight` loads correct checkpoint
- **RLAIF training unstable**: Verify reward model path, check reward signal variance, ensure data quality
- **Long context performance**: Enable YaRN rope scaling and train with longer sequences

## External Resources

- **Dataset downloads**: [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) or [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset)
- **Model weights**: [HuggingFace Collection](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)
- **Reward model**: [internlm2-1_8b-reward](https://huggingface.co/internlm/internlm2-1_8b-reward)

## Development Philosophy

This codebase prioritizes **educational clarity** over abstraction:
- Core algorithms (attention, RoPE, LoRA, DPO, PPO, GRPO) are implemented from scratch in PyTorch
- Avoid black-box third-party wrappers when possible
- Code is heavily commented in Chinese (original documentation language)
- Designed for learning LLM internals, not production deployment
