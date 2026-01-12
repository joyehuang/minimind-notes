<div align="center">

# 🧠 MiniMind | LLM Training Principles Tutorial

<p>
  <strong>From 0 to 1: Understanding Large Models - Not a "copy-paste" manual, but a principle-first experimental lab</strong>
  <br>
  <em>From 0 to 1: Not a "copy-paste" manual, but a principle-first experimental lab for LLMs.</em>
</p>

<p>
  <a href="https://minimind.wiki">
    <img src="https://img.shields.io/badge/Documentation-Wiki-blue?style=for-the-badge&logo=read-the-docs" alt="Website">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
</p>

<h3>
  🎉 Full Interactive Documentation Live
  <br>
  <a href="https://minimind.wiki">👉 https://minimind.wiki 👈</a>
</h3>

<p>
  <a href="#-quick-start">⚡ Quick Start</a> • 
  <a href="#-learning-paths">🗺️ Learning Paths</a> • 
  <a href="#-module-navigation">📦 Module Navigation</a> • 
  <a href="README.md">🇨🇳 中文 README</a>
</p>

</div>

---

## 📖 Introduction

MiniMind aims to help developers deeply understand the training mechanisms of Large Language Models (LLMs) through practice, using extremely concise code and **comparative experiments**. It not only tells you "how to do it" but also shows you "why to do it this way" through experimental data.

> **Why this project?** Understand every design choice in LLM training through comparative experiments.

---

## 🎯 What is This?

This is a **modular LLM training tutorial** that helps you understand the training principles of modern large language models (such as Llama, GPT).

**Core Features**:
- ✅ **Principle-First**: Understand "why it's designed this way", not just "how to run it"
- ✅ **Comparative Experiments**: Each design choice answers "what happens if we don't do this" through experiments
- ✅ **Modular**: 6 independent modules, from basic components to complete architecture
- ✅ **Low Barrier**: Learning-stage experiments can run on CPU (minutes), full training requires GPU

**Based on Project**: [MiniMind](https://github.com/jingyaogong/minimind) - Complete tutorial for training ultra-small language models from scratch

---

## 👥 Who is This For?

### 🎯 **Perfect for Students Seeking LLM Internships/Jobs!**

This project is especially designed for students who want to enter the LLM field. By systematically learning LLM training principles, you will:
- ✅ **Ace Interviews**: Deep understanding of Transformer, Attention, RoPE, and other core mechanisms helps you excel in technical interviews
- ✅ **Stand Out**: Complete comparative experiments to showcase your deep understanding of LLM principles, making your resume more competitive
- ✅ **Quick Start**: Understand modern LLM (Llama, GPT) training pipelines from scratch, not just a "framework user"
- ✅ **Career Growth**: Master LLM training principles to build a solid foundation for future LLM-related work

---

### 🎓 Students and Researchers
- **🎯 Students Seeking LLM Internships/Jobs**: Systematically learn LLM training principles to improve technical interview success rates
- **📚 ML/DL Students**: Deeply understand the internal mechanisms of Transformers and LLMs, move beyond theory
- **🔬 Graduate Students/PhD Candidates**: Understand LLM training principles to provide a solid foundation for research and paper writing
- **💡 Researchers**: Understand design choices in modern LLM architectures and their underlying principles to inspire research directions

### 💻 Developers
- **🤖 AI/ML Engineers**: Progress from "knowing how to use frameworks" to "understanding principles" to solve real problems with confidence
- **🌐 Full-Stack Developers**: Interested in LLMs and want to systematically learn training mechanisms to expand your tech stack
- **⚙️ Algorithm Engineers**: Need to optimize or improve LLM training pipelines, understanding principles is key to making the right decisions

### 🚀 Learners
- **📖 With PyTorch Basics**: Familiar with basic deep learning concepts, want to dive deep into LLMs
- **🛠️ Hands-on Learners**: Prefer understanding through experiments and code, not just theory
- **🔍 Seek Deep Understanding**: Not satisfied with "getting code to work", want to know "why it's designed this way"

### ❌ Not For
- Complete beginners (suggest learning PyTorch basics first)
- Users who only want to quickly deploy models without understanding principles
- Users needing production-ready code and best practices (this project focuses on teaching)

**💪 If you're preparing for LLM job interviews or want to deeply understand LLM training principles, this project is for you!** 🚀

---

## ⚡ Quick Start

### 30-Minute Experience of Core Design

Run three key experiments to understand the core design choices of LLMs:

```bash
# 1. Clone the repository
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes

# 2. Activate virtual environment (if you have one)
source venv/bin/activate

# 3. Experiment 1: Why do we need normalization?
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py

# 4. Experiment 2: Why use RoPE positional encoding?
cd ../../02-position-encoding/experiments
python exp1_rope_basics.py

# 5. Experiment 3: How does Attention work?
cd ../../03-attention/experiments
python exp1_attention_basics.py
```

**What You'll See**:
- Visualization of gradient vanishing
- Principles of RoPE rotational encoding
- Calculation process of Attention weights

**Next Step**: Read [ROADMAP.md](./ROADMAP.md) to choose your learning path

---

## 📚 Learning Paths

Choose the appropriate path based on your time and goals:

| Path | Duration | Goal | Link |
|------|----------|------|------|
| ⚡ **Quick Experience** | 30 minutes | Understand core design choices | [Start](./ROADMAP.md#-quick-experience-30-minutes) |
| 📚 **Systematic Learning** | 6 hours | Master basic components | [Start](./ROADMAP.md#-systematic-learning-6-hours) |
| 🎓 **Deep Mastery** | 30+ hours | Train model from scratch | [Start](./ROADMAP.md#-deep-mastery-30-hours) |

Detailed roadmap: [ROADMAP.md](./ROADMAP.md)

---

## 🧱 Module Navigation

### Tier 1: Foundation (Basic Components)

| Module | Core Question | Experiments | Status |
|--------|---------------|-------------|--------|
| [01-normalization](modules/01-foundation/01-normalization) | Why normalization? Pre-LN vs Post-LN? | 2 | ✅ Complete |
| [02-position-encoding](modules/01-foundation/02-position-encoding) | Why choose RoPE? How to extrapolate length? | 4 | 🟡 Experiments Complete |
| [03-attention](modules/01-foundation/03-attention) | What's the intuition behind QKV? Why multi-head? | 3 | 🟡 Experiments Complete |
| [04-feedforward](modules/01-foundation/04-feedforward) | What knowledge does FFN store? Why expansion? | 1 | 🟡 Experiments Complete |

### Tier 2: Architecture (Architecture Assembly)

| Module | Core Question | Status |
|--------|---------------|--------|
| [01-residual-connection](modules/02-architecture/01-residual-connection) | Why residual connections? How to stabilize gradients? | 🔜 To Be Developed |
| [02-transformer-block](modules/02-architecture/02-transformer-block) | How to assemble components? Why this order? | 🔜 To Be Developed |

**Legend**:
- ✅ Complete: Includes teaching docs + experiment code + quizzes
- 🟡 Experiments Complete: Has experiment code, docs to be added
- 🔜 To Be Developed: Directory structure only

Detailed navigation: [modules/README.md](modules/README.md)

---

## 🔬 Experimental Features

### 1. Comparative Experiment Design

Each module answers core questions through experiments:

**Example**: Normalization Module

| Configuration | Converges? | NaN Appears at Step | Final Loss |
|---------------|------------|-------------------|------------|
| ❌ NoNorm | No | ~500 | NaN |
| ⚠️ Post-LN | Yes | - | 3.5 |
| ✅ Pre-LN + RMSNorm | Yes | - | 2.7 |

**Conclusion**: Pre-LN + RMSNorm is most stable → Standard choice for modern LLMs

---

### 2. Progressive Learning

```
Experiment → Intuition → Theory → Code
  ↓         ↓         ↓        ↓
10 min    20 min    30 min   10 min
```

Run experiments first to build intuition, then study theory to understand principles, finally read source code to master implementation.

---

### 3. Runs on Laptops

All experiments are based on **TinyShakespeare** (1MB) or synthetic data:
- ✅ No GPU required (CPU/MPS both work)
- ✅ Each experiment < 10 minutes
- ✅ Total data < 100 MB

---

## 📖 Document Structure

Each module contains:

```
01-normalization/
├── README.md           # Module navigation
├── teaching.md         # Teaching docs (Why/What/How)
├── code_guide.md       # Source code guide (links to MiniMind)
├── quiz.md            # Self-assessment questions
└── experiments/        # Comparative experiments
    ├── exp1_*.py
    ├── exp2_*.py
    └── results/        # Expected outputs
```

**Document Template** (teaching.md):
1. **Why**: Problem scenario + intuitive understanding
2. **What**: Mathematical definition + comparison table
3. **How**: Experimental design + expected results

---

## 🛠️ Tech Stack

- **Framework**: PyTorch 2.0+
- **Data**: TinyShakespeare, TinyStories
- **Visualization**: Matplotlib, Seaborn
- **Original Project**: [MiniMind](https://github.com/jingyaogong/minimind)

---

## 🤝 Contributing

Welcome contributions:
- ✨ New comparative experiments
- 📊 Better visualizations
- 🌍 English translations
- 🐛 Bug fixes

**Before submitting, please ensure**:
- [ ] Experiments can run independently
- [ ] Code has sufficient comments
- [ ] Results are reproducible (fixed random seed)
- [ ] Follows existing file structure

---

## 📂 Repository Structure

```
minimind-notes/
├── modules/                    # Modular teaching (new architecture)
│   ├── common/                # Common utilities
│   ├── 01-foundation/         # Basic components
│   └── 02-architecture/       # Architecture assembly
│
├── docs/                       # Personal learning records
│   ├── learning_log.md        # Learning log
│   ├── knowledge_base.md      # Knowledge base
│   └── notes.md              # Index
│
├── model/                      # MiniMind original code
├── trainer/                    # Training scripts
├── dataset/                    # Datasets
│
├── README.md                   # This file
├── ROADMAP.md                  # Learning roadmap
└── CLAUDE.md                   # AI assistant guide
```

---

## 📜 Acknowledgments

This repository is based on the following project:
- [MiniMind](https://github.com/jingyaogong/minimind) - Core code and training pipeline
- All modules link to MiniMind's real implementation

Special thanks to [@jingyaogong](https://github.com/jingyaogong) for open-sourcing the MiniMind project!

---

## 🔗 Related Resources

### Online Website
- **[minimind.wiki](https://minimind.wiki)** - Access full documentation and interactive content online

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [RoFormer: RoPE](https://arxiv.org/abs/2104.09864) - Rotary Position Embedding
- [RMSNorm](https://arxiv.org/abs/1910.07467) - Root Mean Square Layer Normalization

### Blogs
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

### Videos
- [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

## 📞 Contact

- Issue Feedback: [GitHub Issues](https://github.com/joyehuang/minimind-notes/issues)
- Original Project: [MiniMind](https://github.com/jingyaogong/minimind)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

<div align="center">

**⭐ If this project helps you, please give it a Star!**

**🌐 Visit Online Website:** [https://minimind.wiki](https://minimind.wiki)

**Ready to start?** [Begin your learning journey](./ROADMAP.md) 🚀

</div>
