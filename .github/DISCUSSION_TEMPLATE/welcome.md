# 👋 欢迎来到 MiniMind Notes！

欢迎加入 MiniMind Notes 社区！这是一个专注于通过**对照实验**深入理解 LLM 训练原理的教学项目。

## 🎯 这个项目是什么？

MiniMind Notes 是一个**模块化的 LLM 训练教案**，帮助你理解现代大语言模型（如 Llama、GPT）的训练原理。

**核心特点**：
- ✅ **原理优先**：理解"为什么这样设计"，而不只是"怎么运行"
- ✅ **对照实验**：每个设计选择都通过实验回答"不这样做会怎样"
- ✅ **模块化**：6 个独立模块，从基础组件到完整架构
- ✅ **低门槛**：学习阶段实验可在 CPU 运行（几分钟），完整训练需 GPU

## 🚀 快速开始

### 30 分钟体验核心设计

```bash
# 1. 克隆仓库
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes

# 2. 实验 1：为什么需要归一化？
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py

# 3. 实验 2：为什么用 RoPE 位置编码？
cd ../../02-position-encoding/experiments
python exp1_rope_basics.py

# 4. 实验 3：Attention 如何工作？
cd ../../03-attention/experiments
python exp1_attention_basics.py
```

**详细路线图**：查看 [ROADMAP.md](https://github.com/joyehuang/minimind-notes/blob/main/ROADMAP.md)

## 💬 如何参与？

### 提问和讨论
- ❓ **有问题？** 使用 [Question 模板](https://github.com/joyehuang/minimind-notes/issues/new?template=question.md) 创建 Issue
- 💡 **有想法？** 在 Discussions 中分享你的学习心得或实验想法
- 🎓 **学习心得？** 欢迎分享你的学习过程和收获

### 贡献代码
- 🐛 **发现 Bug？** 使用 [Bug 报告模板](https://github.com/joyehuang/minimind-notes/issues/new?template=bug_report.md)
- ✨ **有新功能？** 使用 [功能建议模板](https://github.com/joyehuang/minimind-notes/issues/new?template=feature_request.md)
- 🔬 **有新实验？** 使用 [实验建议模板](https://github.com/joyehuang/minimind-notes/issues/new?template=experiment_suggestion.md)

**完整贡献指南**：查看 [CONTRIBUTING.md](https://github.com/joyehuang/minimind-notes/blob/main/.github/CONTRIBUTING.md)

## 📚 资源链接

- 🌐 **在线文档**：[https://minimind.wiki](https://minimind.wiki)
- 📖 **README**：[项目主页](https://github.com/joyehuang/minimind-notes)
- 🗺️ **学习路线**：[ROADMAP.md](https://github.com/joyehuang/minimind-notes/blob/main/ROADMAP.md)
- 🔗 **原项目**：[MiniMind](https://github.com/jingyaogong/minimind)

## 🎯 适合人群

- 🎓 **学生和研究者**：深入理解 Transformer 和 LLM 的内部机制
- 💻 **开发者**：从"会用框架"提升到"理解原理"
- 🚀 **求职者**：准备大模型岗位面试，系统学习 LLM 训练原理
- 📖 **学习者**：通过实验和代码理解原理，而非只看理论

## 🤝 社区准则

我们遵循 [Contributor Covenant Code of Conduct](https://github.com/joyehuang/minimind-notes/blob/main/CODE_OF_CONDUCT.md)。请保持：
- ✅ 尊重和包容
- ✅ 建设性的反馈
- ✅ 帮助他人学习

## 💡 讨论分类

- 💬 **General** - 一般讨论和项目相关话题
- ❓ **Q&A** - 提问和回答
- 💡 **Ideas** - 功能建议和实验想法
- 🎓 **Learning** - 学习心得和经验分享
- 📊 **Show and Tell** - 展示你的实验成果

---

**准备好了吗？** 🚀

- 开始你的学习之旅：[ROADMAP.md](https://github.com/joyehuang/minimind-notes/blob/main/ROADMAP.md)
- 查看在线文档：[minimind.wiki](https://minimind.wiki)
- 参与讨论：在下方留言，介绍你自己或分享你的学习目标！

**⭐ 如果这个项目对你有帮助，请给个 Star！**
