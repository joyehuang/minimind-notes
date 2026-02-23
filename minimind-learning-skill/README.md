# MiniMind Learning Assistant Skill

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/joyehuang/minimind-notes)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-Skill-purple.svg)](https://claude.ai/code)

> 🤖 **自动化学习笔记助手**：让 Claude Code 在你学习 MiniMind 时静默记录学习笔记，无需手动整理！

## 📖 简介

MiniMind Learning Assistant 是一个专为 [MiniMind](https://github.com/jingyaogong/minimind) 学习者设计的 Claude Code Skill。它能够：

- ✅ **自动检测学习内容**：识别 50+ MiniMind 专业术语（RMSNorm, RoPE, DPO, LoRA 等）
- ✅ **静默更新笔记**：在你与 Claude 对话时自动维护三套笔记系统
- ✅ **智能提交代码**：自动生成简洁的中文 Git commit message 并推送
- ✅ **零打扰工作流**：无需手动保存，无需确认，完全自动化

## 🎯 适用场景

这个 Skill 适合：

1. **MiniMind 学习者**：正在学习 MiniMind 框架的开发者
2. **内测社群成员**：参与 MiniMind 内测交流的学员
3. **知识积累者**：希望系统化记录学习过程的人

## 🚀 快速开始

### 1. 安装 Skill

```bash
# 克隆本仓库到 Claude Code skills 目录
cd ~/.claude/skills/  # macOS/Linux
# or
cd %USERPROFILE%\.claude\skills\  # Windows

git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes
cp -r minimind-learning-skill ~/.claude/skills/
```

### 2. 准备你的 MiniMind 仓库

Fork 或克隆 MiniMind 项目：

```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind
```

Skill 会自动在你的仓库中创建 `docs/` 目录结构：

```
your-minimind-fork/
├── model/                   # MiniMind 源码
├── trainer/
├── dataset/
├── docs/                    # ← Skill 自动创建
│   ├── notes.md            # 总索引
│   ├── learning_log.md     # 学习日志（按日期）
│   ├── knowledge_base.md   # 知识库（按主题）
│   └── learning_materials/ # 代码示例
│       ├── README.md
│       └── *.py
└── ...
```

### 3. 开始学习

就是这么简单！直接在你的 MiniMind 目录中使用 Claude Code 提问：

```bash
cd /path/to/your-minimind-fork
claude code  # 启动 Claude Code
```

**示例对话**：

```
你: 什么是 RMSNorm？

Claude: RMSNorm (Root Mean Square Normalization) 是一种归一化技术...
[详细解释...]

# 🎉 背后自动发生：
# ✅ docs/learning_log.md 添加今日学习条目
# ✅ docs/knowledge_base.md 添加 Q1: 什么是 RMSNorm？
# ✅ Git 提交: "学习 RMSNorm 归一化原理"
# ✅ Git 推送到远程仓库
```

**无需任何手动操作！**笔记已经自动保存并同步到 Git。

## 📚 工作原理

### 三层触发系统

#### Tier 1: 即时触发（2秒内更新）

检测到以下内容会立即更新笔记：

- **MiniMind 术语**：RMSNorm, LayerNorm, RoPE, YaRN, Attention, GQA, SwiGLU, Transformer, LoRA, DPO, PPO, GRPO, SFT, RLHF, RLAIF, MoE, distillation 等 50+ 术语
- **问题词**：什么是, 如何, 为什么, 怎样, 解释, 原理, 作用
- **问题指示**：报错, 错误, 问题, 失败, Bug

#### Tier 2: 延迟触发（5秒批量更新）

检测到深度对话场景：

- 多轮对话（3+ 轮交流）
- 包含代码块（```python）
- 包含数学公式（$...$）
- 长回复（>1000 字符）
- 引用源码文件（model/*.py, trainer/*.py）

#### Tier 3: 显式请求（总是触发）

直接告诉 Claude 记录：

```
你: 我刚理解了 RoPE 的多频率机制，记录一下
你: 把这个知识点写入笔记
你: 保存这段对话
```

### 三套笔记系统

#### 1. learning_log.md - 学习日志（按时间）

记录**何时学了什么**，按日期组织：

```markdown
### 2026-02-23: 理解 RoPE 多频率机制

#### ✅ 完成事项
- [x] 理解为什么需要多频率
- [x] 理解浮点数精度限制
- [x] 运行验证代码

#### 🐛 遇到的问题
**问题: CUDA out of memory**
- **错误现象**: 训练时显存不足
- **根本原因**: batch_size 设置过大
- **解决方案**: 减少 batch_size 并增加 accumulation_steps

#### 💭 个人思考
- **收获**: 理解了多频率是为了解决浮点数精度问题
- **疑问解答**: 为什么不直接提高 θ 值？因为会导致长距离位置难以区分

#### 📝 相关学习材料
- 新增代码: `learning_materials/rope_multi_freq.py`
```

#### 2. knowledge_base.md - 知识库（按主题）

记录**学了什么内容**，按技术主题组织：

```markdown
## 1. 归一化技术

**Q1: 什么是 RMSNorm？**

A: RMSNorm 是一种比 LayerNorm 更简单高效的归一化方法...

**详细说明**:
- 只使用 RMS（均方根）归一化，不使用均值中心化
- 计算量减少约 50%
- 在 LLaMA、Mistral 等模型中广泛使用

**代码示例**:
```python
def rmsnorm(x, eps=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms
```

参考代码: `learning_materials/rmsnorm_explained.py`

---

## 2. 位置编码

**Q2: 为什么 RoPE 需要多频率？** ⭐️

A: 因为单一低频率受浮点数精度限制，无法区分相邻位置。

[详细说明...]
```

#### 3. learning_materials/ - 代码示例（可执行）

记录**如何实践**，包含可运行的 Python 代码：

```python
# learning_materials/rope_multi_freq.py
"""
验证 RoPE 多频率机制的必要性

演示内容:
1. 单一低频率的浮点数精度问题
2. 多频率组合的优势
3. 可视化位置编码差异
"""

import torch
import matplotlib.pyplot as plt

# [可执行代码...]
```

### Git 自动化

Skill 会自动生成简洁的中文 commit message：

```bash
# 自动执行
git add docs/
git commit -m "学习 RMSNorm 归一化原理"
git push origin master
```

**Commit Message 规则**：
- ✅ 简洁（<30 字符）
- ✅ 中文
- ✅ 描述性强（学习 XXX / 解决 XXX / 添加 XXX）
- ❌ 无通用短语（如 "Generated with Claude Code"）

## ⚙️ 配置选项（可选）

在你的 MiniMind 仓库根目录创建 `.minimind-learning.json`：

```json
{
  "auto_commit": true,        // 自动提交（默认: true）
  "auto_push": true,          // 自动推送（默认: true）
  "batch_delay": 5,           // 批量更新延迟秒数（默认: 5）
  "git": {
    "remote": "origin",       // Git 远程名称（默认: origin）
    "branch": "master",       // 分支名称（默认: master）
    "retry_count": 3,         // 推送重试次数（默认: 3）
    "timeout": 30             // Git 操作超时秒数（默认: 30）
  },
  "notes_dir": "docs",        // 笔记目录（默认: docs）
  "mark_important": true      // 自动标记重要问答（默认: true）
}
```

## 📊 支持的 MiniMind 术语

Skill 深度理解 50+ MiniMind 专业术语：

### 架构组件（20个）
```
RMSNorm, LayerNorm, BatchNorm, GroupNorm,
RoPE, YaRN, ALiBi, SinusoidalPE,
Attention, MultiHeadAttention, GQA, MQA, FlashAttention,
FeedForward, SwiGLU, GELU, GLU,
Transformer, TransformerBlock, CausalLM
```

### 训练方法（20个）
```
pretrain, pretraining, SFT, supervised fine-tuning,
LoRA, LoRA-r, LoRA-alpha,
DPO, Direct Preference Optimization,
PPO, Proximal Policy Optimization,
GRPO, Group Relative Policy Optimization,
SPO, Simple Policy Optimization,
RLHF, RLAIF,
distillation, knowledge distillation
```

### 模型变体（10个）
```
MiniMind-Dense, MiniMind-MoE, MiniMind-Reason,
Mixture of Experts, MoE, shared experts, routed experts,
expert routing, load balancing loss
```

## 🔧 故障排查

### 问题：笔记没有更新

**检查项**：
- 是否在 MiniMind 相关对话中？
- 是否包含触发关键词？（见 Tier 1 列表）

**解决方案**：
- 使用显式请求："记录这个知识点"
- 检查是否在正确的 MiniMind 仓库目录中

### 问题：Git 推送失败

**检查项**：
- 网络连接是否正常？
- Git 凭据是否配置？

**解决方案**：
- 更改已在本地提交，稍后手动推送：
  ```bash
  cd docs/
  git push origin master
  ```

### 问题：Q 编号跳跃（Q1, Q2, Q5...）

**原因**：手动编辑 knowledge_base.md 时删除了问题

**解决方案**：
- 运行验证脚本修复编号：
  ```bash
  cd minimind-learning-skill
  python scripts/validate_notes.py --fix-numbering
  ```

### 问题：重复条目

**原因**：同一概念在不同时间讨论多次

**解决方案**：
- 手动合并重复的 Q&A
- Skill 未来会避免创建重复条目

## 📖 使用示例

### 示例 1：学习新概念

```
你: 什么是 RMSNorm？它和 LayerNorm 有什么区别？

Claude: RMSNorm (Root Mean Square Normalization) 是...

主要区别:
1. RMSNorm 不使用均值中心化
2. 计算量减少约 50%
3. 在大模型中表现相当或更好

# 自动更新:
# ✅ learning_log.md: "2026-02-23: 学习 RMSNorm"
# ✅ knowledge_base.md: "Q1: 什么是 RMSNorm？"
# ✅ Git: "学习 RMSNorm 归一化原理"
```

### 示例 2：解决问题

```
你: 训练时报错 CUDA out of memory，怎么办？

Claude: 这个错误通常是显存不足。解决方法:
1. 减少 batch_size
2. 增加 accumulation_steps
3. 使用 gradient checkpointing
[详细说明...]

# 自动更新:
# ✅ learning_log.md: 添加"🐛 遇到的问题"章节
# ✅ knowledge_base.md: "Q2: 如何解决 CUDA OOM？"
# ✅ Git: "解决 CUDA 内存溢出问题"
```

### 示例 3：深度探讨（多轮对话）

```
你: RoPE 是如何工作的？
Claude: [解释 RoPE 基本原理...]

你: 为什么需要多个频率？
Claude: [解释多频率机制...]

你: 能给个代码示例吗？
Claude: [提供 Python 代码...]

# 自动更新（批量，5秒后）:
# ✅ learning_log.md: 完整对话记录
# ✅ knowledge_base.md: Q3, Q4（多个问答）
# ✅ learning_materials/: rope_multi_freq.py（新代码）
# ✅ Git: "深入理解 RoPE 多频率机制"
```

### 示例 4：显式请求

```
你: 我刚理解了 Attention 机制的 Q、K、V 三个矩阵的作用，记录下来

Claude: 好的，我已经记录到你的学习笔记中。

# 自动更新:
# ✅ learning_log.md: 添加个人思考章节
# ✅ knowledge_base.md: Q5: Attention 中 Q/K/V 的作用
# ✅ Git: "理解 Attention Q/K/V 机制"
```

## 🎓 最佳实践

### 1. 保持自然对话

**推荐**：
```
你: 为什么 MiniMind 使用 RMSNorm 而不是 LayerNorm？
```

**不推荐**：
```
你: 请在笔记中记录 RMSNorm 的定义...（过于生硬）
```

Skill 会自动检测并记录，无需特意"喂"给它。

### 2. 利用显式请求

遇到重要知识点时：
```
你: 这个概念很重要，记录一下
你: 把刚才的讨论保存到笔记
```

### 3. 定期查看笔记

虽然 Skill 自动维护笔记，但定期查看可以：
- 巩固学习内容
- 发现知识盲区
- 调整学习节奏

```bash
cd docs/
cat learning_log.md       # 查看学习历程
cat knowledge_base.md     # 查看知识体系
```

### 4. 手动补充细节

Skill 提供结构，你可以手动添加：
- 个人批注
- 更详细的图表
- 外部资源链接

```markdown
# 在 knowledge_base.md 中手动添加
**Q1: 什么是 RMSNorm？**
A: ...

**补充资料**:  ← 手动添加
- 原论文: [Root Mean Square Layer Normalization](https://arxiv.org)
- 可视化工具: ...
```

## 🤝 贡献指南

欢迎贡献！这个 Skill 是为 MiniMind 学习社群设计的。

### 贡献方式

1. **报告 Bug**
   ```bash
   # 在 GitHub 创建 Issue
   https://github.com/joyehuang/minimind-notes/issues
   ```

2. **提出改进建议**
   - 新的触发场景
   - 更好的笔记格式
   - 其他语言支持（英文、日文）

3. **提交代码**
   ```bash
   git clone https://github.com/joyehuang/minimind-notes.git
   cd minimind-notes/minimind-learning-skill
   git checkout -b feature/your-feature
   # 进行修改
   git commit -m "feat: add new feature"
   git push origin feature/your-feature
   # 创建 Pull Request
   ```

### 开发建议

- 遵循现有代码风格
- 添加测试场景
- 更新文档

## 📝 版本历史

### v1.0.0 (2026-02-23)
- ✅ 初始版本发布
- ✅ 三层触发系统
- ✅ 完整 Git 自动化
- ✅ 50+ MiniMind 术语识别
- ✅ 三套笔记系统

### 未来计划
- [ ] 支持英文笔记
- [ ] Anki/Obsidian 集成
- [ ] 语音输入记录
- [ ] 自动生成概念图

## 📄 许可证

MIT License - 自由使用和修改，用于教育目的。

## 🙏 致谢

- **作者**: Joye Huang ([@joyehuang](https://github.com/joyehuang))
- **灵感来源**: [MiniMind](https://github.com/jingyaogong/minimind) by jingyaogong
- **社群**: MiniMind 学习群成员

## 📧 联系方式

- **GitHub Issues**: https://github.com/joyehuang/minimind-notes/issues
- **讨论区**: [MiniMind 社群]

---

**开始你的 MiniMind 学习之旅，让 AI 成为你的笔记助手！** 🚀

