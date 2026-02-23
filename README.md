# MiniMind Learning Skill

> 自动化学习笔记系统 - 静默记录你的 MiniMind 学习历程

## 快速开始

```bash
# 1. Clone 仓库
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes

# 2. 运行初始化脚本
chmod +x init-for-learning.sh
./init-for-learning.sh              # Linux/macOS
# 或 init-for-learning.bat          # Windows

# 3. 启动 AI Coding Agent
claude code                         # Claude Code CLI
# 或在 VS Code / Cursor / Windsurf 中启动 Claude

# 4. 开始学习
# 对 AI 说："开始学习" 或 "开始今天的学习"
```

就这么简单！

## 它会做什么？

- ✅ **自动检测**：识别 MiniMind 术语（RMSNorm, RoPE, Attention, LoRA, DPO...）
- ✅ **自动记录**：在 `docs/` 维护三套笔记（学习日志、知识库、代码示例）
- ✅ **自动提交**：生成简洁的中文 commit message 并推送到 Git
- ✅ **完全静默**：专注学习对话，零打扰

## 示例

```
你: 什么是 RMSNorm？

AI: RMSNorm 是一种归一化方法...
    [详细讲解]

# 背后自动发生：
✅ docs/learning_log.md 添加今日条目
✅ docs/knowledge_base.md 添加 Q1
✅ git commit -m "学习 RMSNorm 归一化原理"
✅ git push
```

## 文件结构

```
minimind-notes/
├── minimind-learning-skill/    # Skill 本身
│   ├── SKILL.md               # 核心定义
│   ├── templates/             # 笔记模板
│   └── scripts/               # 验证工具
├── init-for-learning.sh       # 初始化脚本
├── init-for-learning.bat      # Windows 版本
└── example-notes/             # 示例笔记（参考）

# 使用后会创建：
└── docs/                      # 你的学习笔记
    ├── learning_log.md        # 按日期
    ├── knowledge_base.md      # 按主题 (Q&A)
    └── learning_materials/    # 代码示例
```

## 支持的 AI Agent

- Claude Code (CLI)
- VS Code + Claude 扩展
- Cursor
- Windsurf
- 任何支持 Claude Skills 的环境

## 配置（可选）

创建 `.minimind-learning.json`：

```json
{
  "auto_commit": true,
  "auto_push": true,
  "batch_delay": 5,
  "notes_dir": "docs"
}
```

## 问题排查

**Skill 没激活？**
```bash
ls ~/.claude/skills/minimind-learning    # 检查是否安装
# 对 AI 说："记录一下"                  # 显式触发
```

**Git 推送失败？**
```bash
git config user.name "Your Name"
git config user.email "your@email.com"
git push origin master                   # 手动推送
```

## 验证笔记

```bash
# 检查 Q 编号连续性
python ~/.claude/skills/minimind-learning/scripts/validate_notes.py

# 自动修复编号
python ~/.claude/skills/minimind-learning/scripts/validate_notes.py --fix-numbering
```

## 查看示例

```bash
cd example-notes
cat learning_log.md          # 作者的学习日志
cat knowledge_base.md         # 知识库 (Q1-Q19)
ls learning_materials/        # 代码示例
```

---

**License**: MIT | **Author**: [@joyehuang](https://github.com/joyehuang) | **Based on**: [MiniMind](https://github.com/jingyaogong/minimind)
