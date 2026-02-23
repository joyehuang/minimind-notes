# MiniMind Learning Skill 快速开始

> ⚡️ 5 分钟上手指南

## 📦 安装（3 步）

### 1. 复制 Skill 到 Claude Code

```bash
# macOS/Linux
cp -r minimind-learning-skill ~/.claude/skills/

# Windows
xcopy /E /I minimind-learning-skill %USERPROFILE%\.claude\skills\minimind-learning
```

### 2. 准备你的 MiniMind 仓库

```bash
# 如果还没有 MiniMind 仓库
git clone https://github.com/jingyaogong/minimind.git
cd minimind
```

### 3. 启动 Claude Code

```bash
cd minimind  # 在 MiniMind 目录中
claude code   # 或直接在 IDE 中使用 Claude Code
```

**就这么简单！** Skill 会自动创建 `docs/` 目录和笔记文件。

## 🎯 第一次使用

### 试试这些提问

**学习新概念**：
```
你: 什么是 RMSNorm？
```

**解决问题**：
```
你: 训练时显存不够怎么办？
```

**代码实践**：
```
你: 能给我一个 RoPE 的简单实现吗？
```

**深入讨论**：
```
你: 为什么 Transformer 需要位置编码？
你: RoPE 是怎么实现的？
你: 它和传统的 sinusoidal 编码有什么区别？
```

### 查看笔记

```bash
# 学习日志（按时间）
cat docs/learning_log.md

# 知识库（按主题）
cat docs/knowledge_base.md

# 代码索引
cat docs/learning_materials/README.md
```

### Git 历史

```bash
# 查看自动生成的 commit
git log --oneline

# 应该看到类似：
# abc1234 学习 RMSNorm 归一化原理
# def5678 理解 RoPE 位置编码机制
```

## ⚙️ 可选配置

如果想自定义行为，创建 `.minimind-learning.json`：

```bash
cd minimind  # 你的 MiniMind 仓库根目录
nano .minimind-learning.json
```

粘贴并修改：
```json
{
  "auto_commit": true,
  "auto_push": true,
  "batch_delay": 5,
  "notes_dir": "docs"
}
```

保存即可！Skill 会自动读取。

## 🔍 验证安装

运行验证脚本：

```bash
# 在 skill 目录中
cd ~/.claude/skills/minimind-learning
python scripts/validate_notes.py --docs-dir /path/to/your/minimind/docs

# 或在你的 MiniMind 目录中
cd /path/to/your/minimind
python ~/.claude/skills/minimind-learning/scripts/validate_notes.py
```

输出示例：
```
🔍 开始验证笔记系统...

✅ 目录结构完整
📅 验证 learning_log.md...
   找到 3 个日期条目
🧠 验证 knowledge_base.md...
   找到 5 个问答
   ✅ Q 编号连续 (Q1-Q5)
   找到 2 个重要问答 (⭐️)
🔬 验证 learning_materials/...
   找到 2 个 Python 文件
📖 验证 notes.md...

============================================================
✅ 所有检查通过！笔记系统状态良好。
============================================================
```

## 🐛 常见问题

### Q: 笔记没有自动更新？

**检查**：
1. 是否在 MiniMind 仓库目录中？
   ```bash
   pwd  # 应该显示你的 minimind 目录
   ```

2. 是否在讨论 MiniMind 相关内容？
   - 关键词：RMSNorm, RoPE, Attention, LoRA, DPO, PPO 等

3. 试试显式请求：
   ```
   你: 记录一下刚才的讨论
   ```

### Q: Git 推送失败？

**解决**：
1. 检查网络连接
2. 确认 Git 凭据配置：
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your@email.com"
   ```

3. 手动推送：
   ```bash
   cd docs
   git push origin master
   ```

### Q: Q 编号不连续？

**修复**：
```bash
cd ~/.claude/skills/minimind-learning
python scripts/validate_notes.py --fix-numbering --docs-dir /path/to/minimind/docs
```

## 📚 下一步

- 📖 阅读完整文档：[README.md](README.md)
- 🔧 查看高级配置：[SKILL.md](SKILL.md)
- 🤝 参与贡献：[CONTRIBUTING.md](CONTRIBUTING.md)

## 💡 使用技巧

### 1. 自然对话，无需刻意

**推荐** ✅：
```
你: 为什么 LLaMA 使用 RMSNorm 而不是 LayerNorm？
```

**不推荐** ❌：
```
你: 请在笔记中记录 RMSNorm 的定义，然后解释...
```

Skill 会自动检测和记录，保持自然对话即可。

### 2. 标记重要知识点

遇到核心概念时：
```
你: 这个很重要，记录一下
```

Skill 会用 ⭐️ 标记该问答。

### 3. 定期回顾笔记

每周查看学习进度：
```bash
# 查看本周学了什么
grep -A 5 "2026-02" docs/learning_log.md

# 查看所有重要知识点
grep "⭐️" docs/knowledge_base.md
```

### 4. 手动补充细节

Skill 提供结构，你可以添加：
- 个人批注
- 外部链接
- 更详细的图表

```bash
# 编辑知识库
vim docs/knowledge_base.md

# 在自动生成的 Q&A 下添加
**补充资料**:
- 论文：[链接]
- 博客：[链接]
```

## 🎉 开始学习吧！

现在你已经准备好了。打开 Claude Code，开始你的 MiniMind 学习之旅！

**记住**：
- ✅ 保持自然对话
- ✅ 遇到重要点说"记录"
- ✅ 定期回顾笔记
- ✅ 享受学习过程

祝学习愉快！🚀
