<div align="center">

# 🧠 MiniMind | LLM 训练原理教案

<p>
  <strong>从 0 到 1 理解大模型：这不是"复制粘贴手册"，而是"原理优先"的实验场</strong>
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
  🎉 完整交互式文档已上线 / Full Documentation Live
  <br>
  <a href="https://minimind.wiki">👉 https://minimind.wiki 👈</a>
</h3>

<p>
  <a href="#-快速开始">⚡ 快速开始</a> • 
  <a href="#-学习路线">🗺️ 学习路线</a> • 
  <a href="#-模块导航">📦 模块导航</a> • 
  <a href="README_en.md">🇺🇸 English Readme</a>
</p>

</div>

---

## 📖 简介 (Introduction)

MiniMind 旨在通过极其精简的代码和**对照实验**，帮助开发者通过实践深入理解大语言模型（LLM）的训练机制。不仅告诉你“怎么做”，更通过实验数据告诉你“为什么要这么做”。

> **Why this project?** Understand every design choice in LLM training through comparative experiments.

---

## 🎯 这是什么？

这是一个**模块化的 LLM 训练教案**，帮助你理解现代大语言模型（如 Llama、GPT）的训练原理。

**核心特点**：
- ✅ **原理优先**：理解"为什么这样设计"，而不只是"怎么运行"
- ✅ **对照实验**：每个设计选择都通过实验回答"不这样做会怎样"
- ✅ **模块化**：6 个独立模块，从基础组件到完整架构
- ✅ **低门槛**：学习阶段实验可在 CPU 运行（几分钟），完整训练需 GPU

**基于项目**：[MiniMind](https://github.com/jingyaogong/minimind) - 从零训练超小语言模型的完整教程

---

## 👥 适合人群

### 🎯 **正在寻找大模型岗位实习/工作的同学必看！**

这个项目特别适合想要进入大模型领域的同学。通过系统学习 LLM 训练原理，你将：
- ✅ **面试加分**：深入理解 Transformer、Attention、RoPE 等核心机制，轻松应对技术面试
- ✅ **项目亮点**：完成对照实验，展示你对 LLM 原理的深度理解，简历更有竞争力
- ✅ **快速上手**：从零到一理解现代 LLM（Llama、GPT）的训练流程，不再只是"调包侠"
- ✅ **职业发展**：掌握 LLM 训练原理，为未来从事大模型相关工作打下坚实基础

---

### 🎓 学生和研究者
- **🎯 寻找大模型实习/工作的同学**：系统学习 LLM 训练原理，提升技术面试通过率
- **📚 机器学习/深度学习学生**：深入理解 Transformer 和 LLM 的内部机制，不再纸上谈兵
- **🔬 研究生/博士生**：理解 LLM 训练原理，为研究和论文写作提供扎实基础
- **💡 研究者**：了解现代 LLM 架构的设计选择及其背后的原理，启发研究方向

### 💻 开发者
- **🤖 AI/ML 工程师**：从"会用框架"提升到"理解原理"，解决实际问题更有底气
- **🌐 全栈开发者**：对 LLM 感兴趣，希望系统学习其训练机制，拓展技术栈
- **⚙️ 算法工程师**：需要优化或改进 LLM 训练流程，理解原理才能做出正确决策

### 🚀 学习者
- **📖 有 PyTorch 基础**：熟悉基本深度学习概念，想要深入 LLM 领域
- **🛠️ 喜欢动手实践**：通过实验和代码理解原理，而非只看理论
- **🔍 追求深度理解**：不满足于"跑通代码"，想知道"为什么这样设计"

### ❌ 不适合
- 完全零基础的初学者（建议先学习 PyTorch 基础）
- 只想快速部署模型，不关心原理的用户
- 需要生产级代码和最佳实践的用户（本项目聚焦教学）

**💪 如果你正在准备大模型岗位面试，或者想要深入理解 LLM 训练原理，这个项目就是为你准备的！** 🚀

---

## ⚡ 快速开始

### 30 分钟体验核心设计

运行三个关键实验，理解 LLM 的核心设计选择：

```bash
# 1. 克隆仓库
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes

# 2. 激活虚拟环境（如果已有）
source venv/bin/activate

# 3. 实验 1：为什么需要归一化？
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py

# 4. 实验 2：为什么用 RoPE 位置编码？
cd ../../02-position-encoding/experiments
python exp1_rope_basics.py

# 5. 实验 3：Attention 如何工作？
cd ../../03-attention/experiments
python exp1_attention_basics.py
```

**你将看到**：
- 梯度消失的可视化
- RoPE 旋转编码的原理
- Attention 权重的计算过程

**下一步**：阅读 [ROADMAP.md](./ROADMAP.md) 选择你的学习路径

---

## 📚 学习路线

根据你的时间和目标，选择合适的路径：

| 路径 | 时长 | 目标 | 链接 |
|------|------|------|------|
| ⚡ **快速体验** | 30 分钟 | 理解核心设计选择 | [开始](./ROADMAP.md#-快速体验-30-分钟) |
| 📚 **系统学习** | 6 小时 | 掌握基础组件 | [开始](./ROADMAP.md#-系统学习-6-小时) |
| 🎓 **深度掌握** | 30+ 小时 | 从零训练模型 | [开始](./ROADMAP.md#-深度掌握-30-小时) |

详细路线图：[ROADMAP.md](./ROADMAP.md)

---

## 🧱 模块导航

### Tier 1: Foundation（基础组件）

| 模块 | 核心问题 | 实验数 | 状态 |
|------|---------|--------|------|
| [01-normalization](modules/01-foundation/01-normalization) | 为什么要归一化？Pre-LN vs Post-LN？ | 2 | ✅ 完整 |
| [02-position-encoding](modules/01-foundation/02-position-encoding) | 为什么选择 RoPE？如何长度外推？ | 4 | 🟡 实验完成 |
| [03-attention](modules/01-foundation/03-attention) | QKV 的直觉是什么？为什么多头？ | 3 | 🟡 实验完成 |
| [04-feedforward](modules/01-foundation/04-feedforward) | FFN 存储什么知识？为什么扩张？ | 1 | 🟡 实验完成 |

### Tier 2: Architecture（架构组装）

| 模块 | 核心问题 | 状态 |
|------|---------|------|
| [01-residual-connection](modules/02-architecture/01-residual-connection) | 为什么需要残差？如何稳定梯度？ | 🔜 待开发 |
| [02-transformer-block](modules/02-architecture/02-transformer-block) | 如何组装组件？为什么这个顺序？ | 🔜 待开发 |

**图例**：
- ✅ 完整：包含教学文档 + 实验代码 + 自测题
- 🟡 实验完成：有实验代码，文档待补充
- 🔜 待开发：仅目录结构

详细导航：[modules/README.md](modules/README.md)

---

## 🔬 实验特色

### 1. 对照实验设计

每个模块通过实验回答核心问题：

**示例**：归一化模块

| 配置 | 是否收敛 | NaN 出现步数 | 最终 Loss |
|------|---------|-------------|-----------|
| ❌ NoNorm | 否 | ~500 | NaN |
| ⚠️ Post-LN | 是 | - | 3.5 |
| ✅ Pre-LN + RMSNorm | 是 | - | 2.7 |

**结论**：Pre-LN + RMSNorm 最稳定 → 现代 LLM 的标准选择

---

### 2. 渐进式学习

```
实验 → 直觉 → 理论 → 代码
  ↓      ↓      ↓      ↓
10分钟  20分钟  30分钟  10分钟
```

先跑实验建立直觉，再看理论理解原理，最后读源码掌握实现。

---

### 3. 可在笔记本运行

所有实验基于 **TinyShakespeare**（1MB）或合成数据：
- ✅ 无需 GPU（CPU/MPS 均可）
- ✅ 每个实验 < 10 分钟
- ✅ 总数据量 < 100 MB

---

## 📖 文档结构

每个模块包含：

```
01-normalization/
├── README.md           # 模块导航
├── teaching.md         # 教学文档（Why/What/How）
├── code_guide.md       # 源码导读（链接到 MiniMind）
├── quiz.md            # 自测题
└── experiments/        # 对照实验
    ├── exp1_*.py
    ├── exp2_*.py
    └── results/        # 预期输出
```

**文档模板**（teaching.md）：
1. **Why（为什么）**：问题场景 + 直觉理解
2. **What（是什么）**：数学定义 + 对比表格
3. **How（怎么验证）**：实验设计 + 预期结果

---

## 🛠️ 技术栈

- **框架**：PyTorch 2.0+
- **数据**：TinyShakespeare, TinyStories
- **可视化**：Matplotlib, Seaborn
- **原项目**：[MiniMind](https://github.com/jingyaogong/minimind)

---

## 🤝 贡献指南

欢迎补充：
- ✨ 新的对照实验
- 📊 更好的可视化
- 🌍 英文翻译
- 🐛 错误修正

**提交前请确保**：
- [ ] 实验可独立运行
- [ ] 代码有充分中文注释
- [ ] 结果可复现（固定随机种子）
- [ ] 遵循现有文件结构

---

## 📂 仓库结构

```
minimind-notes/
├── modules/                    # 模块化教学（新架构）
│   ├── common/                # 通用工具
│   ├── 01-foundation/         # 基础组件
│   └── 02-architecture/       # 架构组装
│
├── docs/                       # 个人学习记录
│   ├── learning_log.md        # 学习日志
│   ├── knowledge_base.md      # 知识库
│   └── notes.md              # 索引
│
├── model/                      # MiniMind 原始代码
├── trainer/                    # 训练脚本
├── dataset/                    # 数据集
│
├── README.md                   # 本文件
├── ROADMAP.md                  # 学习路线图
└── CLAUDE.md                   # AI 助手指南
```

---

## 📜 致谢

本仓库基于以下项目：
- [MiniMind](https://github.com/jingyaogong/minimind) - 核心代码和训练流程
- 所有模块链接到 MiniMind 的真实实现

特别感谢 [@jingyaogong](https://github.com/jingyaogong) 开源的 MiniMind 项目！

---

## 🔗 相关资源

### 在线网站
- **[minimind.wiki](https://minimind.wiki)** - 在线访问完整文档和交互式内容

### 论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [RoFormer: RoPE](https://arxiv.org/abs/2104.09864) - 旋转位置编码
- [RMSNorm](https://arxiv.org/abs/1910.07467) - 均方根归一化

### 博客
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

### 视频
- [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

## 📞 联系方式

- 问题反馈：[GitHub Issues](https://github.com/joyehuang/minimind-notes/issues)
- 原项目：[MiniMind](https://github.com/jingyaogong/minimind)

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star！**

**🌐 访问在线网站：** [https://minimind.wiki](https://minimind.wiki)

**准备好了吗？** [开始你的学习之旅](./ROADMAP.md) 🚀

</div>
