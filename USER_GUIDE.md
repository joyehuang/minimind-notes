# MiniMind Learning Skill - 使用指南

> 自动化学习笔记系统，支持多种 AI Coding Agent

## 🚀 快速开始

### 步骤 1：Clone 仓库

```bash
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes
```

### 步骤 2：运行初始化脚本

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
- ✅ 备份作者的学习记录到 `example-notes/`
- ✅ 安装 Skill 到 `~/.claude/skills/`
- ✅ 配置 Git 忽略用户笔记

### 步骤 3：启动学习

根据你使用的 AI Coding Agent：

#### 方式 A: Claude Code (CLI)

```bash
claude code
```

#### 方式 B: VS Code + Claude 扩展

1. 打开 VS Code
2. 确保安装了 Claude 扩展
3. 启动 Claude 对话

#### 方式 C: Cursor

1. 打开 Cursor
2. 启动 AI 助手 (Ctrl/Cmd + K)

#### 方式 D: Windsurf / 其他 IDE

1. 打开你的 IDE
2. 启动内置的 Claude AI 助手

### 步骤 4：开始对话

对 AI 说：

```
开始学习
```

或者：

```
开始今天的学习
```

Skill 会自动激活并显示学习模块菜单！

---

## 📚 工作原理

### 自动触发

Skill 在以下情况自动激活：

1. **学习开始时**
   - "开始学习"、"今天学什么"、"继续学习"

2. **讨论 MiniMind 内容时**
   - 提到：RMSNorm, LayerNorm, RoPE, Attention, LoRA, DPO, PPO, SFT...
   - 问题词：什么是、如何、为什么、解释、原理

3. **遇到问题时**
   - 提到：报错、错误、失败、Bug

### 笔记系统

Skill 会在 `docs/` 目录自动维护三套笔记：

```
minimind-notes/
├── docs/                          # 你的学习笔记（自动生成）
│   ├── notes.md                  # 总索引
│   ├── learning_log.md           # 学习日志（按日期）
│   ├── knowledge_base.md         # 知识库（Q&A，按主题）
│   └── learning_materials/       # 可执行代码示例
│       ├── README.md
│       └── *.py
└── example-notes/                # 作者的示例笔记（参考）
    ├── learning_log.md
    ├── knowledge_base.md
    └── learning_materials/
```

### 自动化流程

```
你提问 → AI 回答
    ↓
Skill 自动检测触发条件
    ↓
提取：问题、概念、代码
    ↓
更新三套笔记文件
    ↓
自动 Git commit + push
```

**完全静默，零打扰！**

---

## 💡 使用示例

### 示例 1：学习新概念

```
你: 什么是 RMSNorm？

AI: RMSNorm (Root Mean Square Normalization) 是...
    [详细讲解]

# 背后自动发生:
✅ docs/learning_log.md 添加今日条目
✅ docs/knowledge_base.md 添加 Q1
✅ git commit -m "学习 RMSNorm 归一化原理"
✅ git push origin master
```

### 示例 2：解决问题

```
你: 训练时报错 CUDA out of memory，怎么办？

AI: 这个错误通常是显存不足。解决方法:
    1. 减少 batch_size
    2. 使用 gradient accumulation
    [详细说明...]

# 背后自动发生:
✅ docs/learning_log.md 添加"遇到的问题"章节
✅ 提取：错误现象、原因、解决方案
✅ git commit -m "解决 CUDA 内存溢出问题"
```

### 示例 3：多轮深度对话

```
你: RoPE 是如何工作的？
AI: [解释 RoPE 基本原理...]

你: 为什么需要多个频率？
AI: [解释多频率机制...]

你: 能给个代码示例吗？
AI: [提供 Python 代码...]

# 5秒后批量更新:
✅ 多个 Q&A 一起记录
✅ 代码保存到 learning_materials/
✅ git commit -m "深入理解 RoPE 多频率机制"
```

---

## 🎯 学习路径

### 推荐顺序

**Week 1: 基础组件**
1. 归一化技术 - RMSNorm, LayerNorm
2. 位置编码 - RoPE, YaRN
3. 注意力机制 - Attention, GQA
4. 前馈网络 - FeedForward, SwiGLU

**Week 2: 完整架构**
5. Transformer Block
6. 完整模型实现

**Week 3-4: 训练技术**
7. 预训练 - Pretraining
8. 监督微调 - SFT
9. 参数高效微调 - LoRA
10. 强化学习 - DPO, PPO, GRPO

### 学习节奏

- 每天 1-2 个小时
- 深度理解优于快速覆盖
- 结合代码实践
- 定期回顾笔记

---

## 🔧 配置选项

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

### 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `auto_commit` | 自动提交到 Git | `true` |
| `auto_push` | 自动推送到远程 | `true` |
| `batch_delay` | 批量更新延迟（秒） | `5` |
| `notes_dir` | 笔记目录 | `"docs"` |
| `mark_important` | 自动标记重要问答 | `true` |

---

## 🛠️ 验证工具

### 检查笔记一致性

```bash
# 验证 Q 编号、日期格式、文件引用
python ~/.claude/skills/minimind-learning/scripts/validate_notes.py

# 自动修复 Q 编号
python ~/.claude/skills/minimind-learning/scripts/validate_notes.py --fix-numbering
```

### 查看学习进度

```bash
# 学习日志（时间线）
cat docs/learning_log.md

# 知识库（主题）
cat docs/knowledge_base.md

# 代码索引
cat docs/learning_materials/README.md

# Git 历史
git log --oneline --grep="学习\|理解\|添加\|解决"
```

---

## 📖 查看示例笔记

想看看最终笔记长什么样？

```bash
cd example-notes

# 查看作者的学习日志
cat learning_log.md

# 查看知识库（Q1-Q19）
cat knowledge_base.md

# 查看学习材料
ls learning_materials/
```

这些是真实的学习记录，可以作为参考！

---

## 🐛 故障排查

### 问题：Skill 没有激活

**检查**：
- 是否在 MiniMind 仓库目录？
- 是否提到了 MiniMind 相关术语？

**解决**：
- 显式请求："记录这个知识点"
- 验证 Skill 安装：`ls ~/.claude/skills/minimind-learning`

### 问题：笔记没有更新

**检查**：
- 查看 `docs/` 目录是否存在
- 检查文件权限

**解决**：
```bash
# 手动创建目录
mkdir -p docs/learning_materials

# 检查权限
ls -la docs/
```

### 问题：Git 推送失败

**检查**：
- 网络连接
- Git 凭据配置

**解决**：
```bash
# 检查 Git 配置
git config user.name
git config user.email

# 手动推送
cd docs/
git push origin master
```

### 问题：Q 编号不连续

**修复**：
```bash
python ~/.claude/skills/minimind-learning/scripts/validate_notes.py --fix-numbering
```

---

## 💬 AI Coding Agent 兼容性

### 已测试

| Agent | 状态 | 说明 |
|-------|------|------|
| Claude Code CLI | ✅ 完全支持 | 官方 CLI |
| VS Code + Claude | ✅ 支持 | 需要 Claude 扩展 |
| Cursor | ✅ 支持 | 内置 Claude |
| Windsurf | ✅ 支持 | Codeium + Claude |

### 理论支持

任何支持 Claude Skills 的环境都应该能工作：
- ✅ Skill 文件位于 `~/.claude/skills/`
- ✅ 使用标准的 Skill 格式
- ✅ 纯文本操作，无特殊依赖

### 报告兼容性问题

如果在你的环境遇到问题：
1. 检查 Skill 是否正确安装
2. 查看 AI 是否支持 Skills 功能
3. 在 GitHub Issues 报告问题

---

## 🎓 学习建议

### 最佳实践

1. **保持自然对话**
   - ✅ "为什么 RMSNorm 比 LayerNorm 快？"
   - ❌ "请记录 RMSNorm 的定义..."

2. **利用显式记录**
   - 遇到重要知识点说："记录一下"

3. **定期回顾**
   - 每周回顾 `learning_log.md`
   - 巩固 `knowledge_base.md` 的 Q&A

4. **实践验证**
   - 运行 `learning_materials/` 中的代码
   - 修改参数观察效果

### 进阶技巧

**自定义学习路径**：
```
你: 我对强化学习比较熟悉，想直接学 RLHF
AI: 好的，让我先确认你了解的前置知识...
```

**问题驱动学习**：
```
你: 为什么 LLaMA 不用 LayerNorm？
AI: 这涉及到 RMSNorm 的设计...
```

**代码优先**：
```
你: 能给我一个最小的 RoPE 实现吗？
AI: [提供代码 + 解释]
```

---

## 📊 学习统计

Skill 会自动维护统计信息：

- 学习天数
- 完成的问答数量
- 创建的代码示例数量
- 解决的问题数量

查看：
```bash
# 统计信息在 notes.md 中
cat docs/notes.md | grep "统计信息" -A 10
```

---

## 🤝 贡献和反馈

### 报告问题

在 GitHub Issues 报告：
- Bug 和错误
- 功能建议
- 文档改进

### 分享经验

欢迎分享：
- 学习心得
- 笔记组织技巧
- Skill 使用技巧

### 改进 Skill

Fork 仓库并提交 PR：
- 新功能
- Bug 修复
- 文档完善

---

## 📝 常见问题

### Q: 能否在多个仓库使用？

A: 可以！Skill 会检测当前目录是否是 MiniMind 仓库，在每个仓库维护独立的 `docs/` 笔记。

### Q: 笔记存储在哪里？

A: 在你当前仓库的 `docs/` 目录，通过 Git 管理。

### Q: 能否导出笔记？

A: 笔记就是标准的 Markdown 文件，可以直接：
- 复制到其他地方
- 导入到 Notion/Obsidian
- 转换为 PDF

### Q: 能否禁用 Git 自动推送？

A: 可以，创建配置文件设置 `"auto_push": false`。

### Q: 能否自定义笔记格式？

A: 模板在 `~/.claude/skills/minimind-learning/templates/`，可以修改。

---

## 📞 获取帮助

- **文档**: [README.md](README.md)
- **快速开始**: [QUICKSTART.md](QUICKSTART.md)
- **GitHub Issues**: https://github.com/joyehuang/minimind-notes/issues
- **作者**: [@joyehuang](https://github.com/joyehuang)

---

**祝学习愉快！** 🚀
