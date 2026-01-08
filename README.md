# MiniMind 训练原理教案

> 通过对照实验理解 LLM 训练的每个设计选择

<div align="center">

**这不是"命令复制手册"，而是"原理优先"的学习仓库**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)

**[🌐 在线访问网站](https://minimind.wiki)** • [快速开始](#-快速开始) • [学习路线](#-学习路线) • [模块导航](#-模块导航) • [贡献指南](#-贡献指南)

</div>

---

## 🌐 在线访问

**🎉 网站已上线！** 访问 **[https://minimind.wiki](https://minimind.wiki)** 在线浏览完整文档和交互式内容。

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
