# MiniMind Learning Skill

<div align="center">

**自动化学习笔记系统 - 让 AI 成为你的学习助手**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/joyehuang/minimind-notes)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![AI Agent](https://img.shields.io/badge/AI-Multi--Agent%20Support-purple.svg)](#-ai-agent-兼容性)

静默记录你的 MiniMind 学习历程 · 支持多种 AI Coding Agent

[快速开始](#-快速开始) • [功能特性](#-功能特性) • [使用指南](USER_GUIDE.md) • [详细文档](minimind-learning-skill/)

</div>

---

## 🎯 这是什么？

**MiniMind Learning Skill** 是一个专为 [MiniMind LLM 训练框架](https://github.com/jingyaogong/minimind) 学习者设计的 Claude Code Skill。它能在你与 AI 对话学习时，**自动在后台维护学习笔记**。

### 核心特性

- ✅ **完全自动化**：静默记录笔记，无需手动整理
- ✅ **智能触发**：识别 50+ MiniMind 专业术语，自动检测学习内容
- ✅ **三套笔记系统**：学习日志 + 知识库 + 代码示例
- ✅ **Git 集成**：自动生成简洁的 commit message 并推送
- ✅ **多平台支持**：Claude Code CLI / VS Code / Cursor / Windsurf

### 用户体验

```
你: 开始学习

AI: 👋 欢迎开始今天的 MiniMind 学习！
    你想学习哪个模块？

    基础组件：
    1. 归一化技术 - RMSNorm, LayerNorm
    2. 位置编码 - RoPE, YaRN
    3. 注意力机制 - Attention, GQA
    ...

你: 什么是 RMSNorm？

AI: RMSNorm 是一种比 LayerNorm 更简单高效的归一化方法...
    [详细讲解]

# 背后自动发生：
✅ docs/learning_log.md 添加今日学习条目
✅ docs/knowledge_base.md 添加 Q1: 什么是 RMSNorm？
✅ git commit -m "学习 RMSNorm 归一化原理"
✅ git push origin master

# 完全静默，零打扰！
```

---

## 🚀 快速开始

### 前置要求

- Git
- 任意支持 Claude Skills 的 AI Coding Agent（见[兼容性列表](#-ai-agent-兼容性)）

### 三步安装

#### 1. Clone 仓库

```bash
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes
```

#### 2. 运行初始化脚本

**Linux / macOS**:
```bash
chmod +x init-for-learning.sh
./init-for-learning.sh
```

**Windows**:
```cmd
init-for-learning.bat
```

脚本会自动：
- ✅ 备份作者的示例笔记到 `example-notes/`
- ✅ 安装 Skill 到 `~/.claude/skills/minimind-learning/`
- ✅ 配置 Git 忽略规则

#### 3. 开始学习

启动你的 AI Coding Agent：

```bash
# Claude Code CLI
claude code

# VS Code + Claude 扩展
# 打开 VS Code，启动 Claude

# Cursor / Windsurf / 其他 IDE
# 启动 AI 助手
```

然后说：

```
开始学习
```

或

```
开始今天的学习
```

就这么简单！🎉

---

## 📚 功能特性

### 智能触发系统

**三层检测机制**：

1. **Tier 1（即时触发）**：
   - 检测到 50+ MiniMind 术语（RMSNorm, RoPE, Attention, LoRA, DPO...）
   - 问题词（什么是、如何、为什么）
   - 问题指示（报错、错误、Bug）

2. **Tier 2（延迟批量）**：
   - 多轮深度对话（3+ 轮）
   - 包含代码块或数学公式
   - 长回复（>1000 字符）

3. **Tier 3（显式请求）**：
   - 用户说"记录一下"、"保存笔记"

### 三套笔记系统

```
docs/
├── notes.md                    # 总索引（快速导航）
├── learning_log.md             # 学习日志（按日期）
│   └── 2026-02-23: 理解 RoPE 多频率机制
│       ├── ✅ 完成事项
│       ├── 🐛 遇到的问题
│       ├── 💭 个人思考
│       └── 📝 相关学习材料
├── knowledge_base.md           # 知识库（按主题）
│   └── Q1: 什么是 RMSNorm？ ⭐️
│       ├── A: 简短回答
│       ├── 详细说明
│       └── 代码示例
└── learning_materials/         # 可执行代码
    ├── README.md
    ├── rmsnorm_explained.py
    └── rope_basics.py
```

### Git 自动化

**智能 Commit Message 生成**：

```bash
# 自动识别动作和主题
"学习 RMSNorm 归一化原理"
"理解 RoPE 多频率机制"
"添加 Attention 学习材料"
"解决 CUDA 内存溢出问题"

# 简洁、准确、中文
# 无"Generated with Claude Code"等通用短语
```

**完整 Git 流程**：
- 自动 `git add docs/`
- 自动 `git commit -m "..."`
- 自动 `git push origin master`
- 智能重试（网络失败时指数退避）

---

## 🎓 学习引导

### 模块化学习路径

Skill 会主动提供学习路径：

```markdown
📚 MiniMind 推荐学习路径:

Week 1: 基础组件
→ Day 1-2: 归一化技术 (RMSNorm)
→ Day 3-4: 位置编码 (RoPE)
→ Day 5-7: 注意力机制 (Attention, GQA)

Week 2: 完整架构
→ Day 8-10: Transformer Block
→ Day 11-14: 完整模型实现

Week 3-4: 训练技术
→ 预训练 → SFT → LoRA → RLHF

你想从哪里开始？
```

### 实践鼓励

检测到代码讨论时，主动建议：

```markdown
💡 要不要创建一个可运行的代码示例？

我可以帮你创建 `learning_materials/rope_basics.py`，
包含完整的 RoPE 实现和可视化。

这样你可以直接运行看效果！
```

### 定期总结

学习多个知识点后，自动总结：

```markdown
📊 今天学习总结:

✅ 完成事项:
- 理解了 RMSNorm 的原理
- 对比了 RMSNorm vs LayerNorm
- 运行了验证代码

🎯 建议:
明天可以学习 RoPE 位置编码，它和 RMSNorm
一起构成了现代 Transformer 的基础。
```

---

## 🛠️ AI Agent 兼容性

### 已测试环境

| AI Coding Agent | 状态 | 说明 |
|----------------|------|------|
| **Claude Code (CLI)** | ✅ 完全支持 | 官方 CLI 工具 |
| **VS Code + Claude** | ✅ 支持 | 需要 Claude 扩展 |
| **Cursor** | ✅ 支持 | 内置 Claude 支持 |
| **Windsurf** | ✅ 支持 | Codeium + Claude |

### 理论支持

任何支持 **Claude Skills** 规范的环境都应该能工作：
- Skill 文件位于 `~/.claude/skills/`
- 使用标准的 SKILL.md 格式
- 纯文本操作，无特殊依赖

---

## 📖 文档

### 核心文档

- **[USER_GUIDE.md](USER_GUIDE.md)** - 完整使用指南
- **[QUICKSTART.md](minimind-learning-skill/QUICKSTART.md)** - 5 分钟快速上手
- **[SKILL.md](minimind-learning-skill/SKILL.md)** - Skill 技术文档
- **[CONTRIBUTING.md](minimind-learning-skill/CONTRIBUTING.md)** - 贡献指南

### 查看示例

想看看最终笔记长什么样？查看作者的真实学习记录：

```bash
cd example-notes
cat learning_log.md          # 学习日志
cat knowledge_base.md         # 知识库（Q1-Q19）
ls learning_materials/        # 代码示例
```

---

## 🔧 高级配置

### 自定义配置

在仓库根目录创建 `.minimind-learning.json`：

```json
{
  "auto_commit": true,
  "auto_push": true,
  "batch_delay": 5,
  "git": {
    "remote": "origin",
    "branch": "master",
    "retry_count": 3,
    "timeout": 30
  },
  "notes_dir": "docs",
  "mark_important": true
}
```

### 验证工具

```bash
# 检查笔记一致性
python ~/.claude/skills/minimind-learning/scripts/validate_notes.py

# 自动修复 Q 编号
python ~/.claude/skills/minimind-learning/scripts/validate_notes.py --fix-numbering
```

---

## 🌟 使用场景

### 个人学习

```
适合：
- 系统学习 MiniMind 框架
- 深入理解 LLM 训练原理
- 积累个人知识库
```

### 团队培训

```
适合：
- 内部技术培训
- 新人入职学习
- 知识沉淀和传承
```

### 内容创作

```
适合：
- 撰写技术博客
- 制作教程视频
- 编写技术文档
```

---

## 📊 技术亮点

### 智能内容提取

- 正则表达式提取问题
- 概念定义自动识别
- 代码块和公式检测
- 问题-解决方案结构化

### 自动分类推断

```python
# 自动推断知识点所属主题
"RMSNorm" → 归一化技术
"RoPE" → 位置编码
"Attention" → 注意力机制
"LoRA" → 参数高效微调
```

### Q 编号管理

- 自动递增编号（Q1, Q2, Q3...）
- 连续性验证
- 自动修复工具

---

## 🐛 故障排查

### 常见问题

**问题：Skill 没有激活**
```bash
# 检查 Skill 是否安装
ls ~/.claude/skills/minimind-learning

# 显式请求触发
# 对 AI 说："记录这个知识点"
```

**问题：Git 推送失败**
```bash
# 检查 Git 配置
git config user.name
git config user.email

# 手动推送
git push origin master
```

**问题：Q 编号跳跃**
```bash
# 自动修复
python ~/.claude/skills/minimind-learning/scripts/validate_notes.py --fix-numbering
```

更多问题见：[USER_GUIDE.md - 故障排查](USER_GUIDE.md#-故障排查)

---

## 🤝 贡献

欢迎贡献！

### 报告问题

在 [GitHub Issues](https://github.com/joyehuang/minimind-notes/issues) 报告：
- Bug 和错误
- 功能建议
- 文档改进

### 提交代码

1. Fork 仓库
2. 创建特性分支
3. 提交 Pull Request

详见：[CONTRIBUTING.md](minimind-learning-skill/CONTRIBUTING.md)

---

## 📜 许可证

MIT License - 自由使用和修改，用于教育目的。

详见：[LICENSE](minimind-learning-skill/LICENSE)

---

## 🙏 致谢

- **MiniMind 项目**：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- **Claude Code**：[Anthropic Claude](https://claude.ai/code)
- **社群贡献者**：感谢所有测试和反馈的用户

---

## 📞 联系方式

- **作者**：Joye Huang ([@joyehuang](https://github.com/joyehuang))
- **项目主页**：https://github.com/joyehuang/minimind-notes
- **问题反馈**：[GitHub Issues](https://github.com/joyehuang/minimind-notes/issues)
- **MiniMind 社群**：参与讨论和交流

---

<div align="center">

**开始你的 MiniMind 学习之旅，让 AI 成为你的笔记助手！** 🚀

[立即开始](#-快速开始) • [查看文档](USER_GUIDE.md) • [查看示例](example-notes/)

</div>
