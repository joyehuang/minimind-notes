# MiniMind Learning Skill - 内测指南

> 感谢参与内测！这个 Skill 会在你学习 MiniMind 时自动记录笔记。

## 🚀 安装步骤

### 1. Clone 仓库

```bash
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes
```

### 2. 运行初始化脚本

**Linux / macOS**:
```bash
chmod +x minimind-learning-skill/scripts/init-for-learning.sh
./minimind-learning-skill/scripts/init-for-learning.sh
```

**Windows**:
```cmd
minimind-learning-skill\scripts\init-for-learning.bat
```

脚本会自动：
- ✅ 备份示例笔记到 `example-notes/`
- ✅ 安装 Skill 到 `~/.claude/skills/minimind-learning/`
- ✅ 配置好环境

### 3. 启动 Claude Code

```bash
claude code
# 或在 VS Code / Cursor / Windsurf 中启动 Claude
```

---

## 🧪 测试场景

### 场景 1：学习开始（必测）

**你说**：
```
开始学习
```

**预期**：
- Claude 应该显示学习模块菜单（8 个选项）
- 包括：归一化技术、位置编码、注意力机制等

---

### 场景 2：提问触发（必测）

**你说**：
```
什么是 RMSNorm？
```

**预期**：
- Claude 正常讲解 RMSNorm
- 背后**静默**更新笔记（不会提示你）
- 等 5-10 秒后检查：

```bash
# 检查是否创建了笔记
ls docs/

# 查看学习日志
cat docs/learning_log.md

# 查看知识库
cat docs/knowledge_base.md
```

**应该看到**：
- `docs/learning_log.md` 有今天的日期条目
- `docs/knowledge_base.md` 有 Q1: 什么是 RMSNorm？

---

### 场景 3：Git 自动化（必测）

**检查 Git 历史**：
```bash
git log --oneline -3
```

**应该看到**：
- 类似 `学习 RMSNorm 归一化原理` 的中文 commit
- 简洁、准确
- 无 "Generated with Claude Code" 等通用短语

---

### 场景 4：多轮对话（可选）

**你说**：
```
你: RoPE 是怎么工作的？
Claude: [回答]

你: 为什么需要多个频率？
Claude: [回答]

你: 能给个代码示例吗？
Claude: [回答]
```

**预期**：
- 5 秒后批量更新笔记
- 多个 Q&A 一起记录
- 一次 Git commit

---

### 场景 5：显式记录（可选）

**你说**：
```
这个知识点很重要，记录一下
```

**预期**：
- 立即触发更新
- Claude 可能简短确认（"已记录"）

---

## ✅ 验证清单

测试完成后，检查：

- [ ] `docs/` 目录已创建
- [ ] `docs/learning_log.md` 有今天的日期条目
- [ ] `docs/knowledge_base.md` 有 Q&A（Q1, Q2...）
- [ ] Git 历史有中文 commit
- [ ] commit message 简洁准确
- [ ] 学习过程中**没有打扰**（完全静默）

---

## 🐛 如果遇到问题

### Skill 没有激活

**检查**：
```bash
ls ~/.claude/skills/minimind-learning
```

**解决**：
- 重新运行初始化脚本
- 或手动复制：`cp -r minimind-learning-skill ~/.claude/skills/minimind-learning`

### 笔记没有更新

**尝试**：
- 显式触发："记录一下刚才的对话"
- 检查是否在正确的目录（minimind-notes）

### Git 没有自动提交

**检查 Git 配置**：
```bash
git config user.name
git config user.email
```

**如果为空，配置**：
```bash
git config user.name "Your Name"
git config user.email "your@email.com"
```

---

## 📝 反馈内容

请记录以下信息（可以截图或文字）：

### 基本信息
- 操作系统：Windows / macOS / Linux
- AI Agent：Claude Code CLI / VS Code / Cursor / Windsurf
- 测试时间：约 X 分钟

### 成功的场景
- [ ] 场景 1（学习开始）
- [ ] 场景 2（提问触发）
- [ ] 场景 3（Git 自动化）
- [ ] 场景 4（多轮对话）
- [ ] 场景 5（显式记录）

### 遇到的问题
1. 问题描述：
2. 操作步骤：
3. 错误信息：
4. 截图（如有）：

### 建议改进
- 有什么不方便的地方？
- 有什么希望增加的功能？
- 文档是否清晰？

---

## 📬 提交反馈

**方式 1：GitHub Issue**
https://github.com/joyehuang/minimind-notes/issues

**方式 2：社群讨论**
[在群里直接反馈]

**方式 3：私信作者**
[@joyehuang](https://github.com/joyehuang)

---

## 💡 测试建议

### 时间安排
- 最少：15 分钟（场景 1-3）
- 推荐：30 分钟（所有场景）

### 测试方式
1. 按顺序测试场景
2. 每个场景测试后验证结果
3. 记录问题和建议
4. 提交反馈

### 常见误区
- ❌ 不要在示例仓库中测试（会和示例笔记混淆）
- ✅ 运行初始化脚本后再测试
- ✅ 等待 5-10 秒让笔记更新
- ✅ 检查 Git 历史验证自动提交

---

## 🎉 测试完成

感谢你的参与！你的反馈对改进 Skill 非常重要。

**下一步**：
- 继续使用并观察效果
- 发现问题随时反馈
- 分享使用心得

---

**问题求助**：如果遇到任何问题，随时在群里提问或提 Issue！
