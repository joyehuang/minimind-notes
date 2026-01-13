# 🤝 贡献指南

感谢你对 MiniMind Notes 项目的兴趣！我们欢迎各种形式的贡献。

## 📋 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
  - [报告 Bug](#报告-bug)
  - [建议新功能](#建议新功能)
  - [提交代码](#提交代码)
  - [改进文档](#改进文档)
  - [建议新实验](#建议新实验)
- [开发流程](#开发流程)
- [代码规范](#代码规范)
- [提交规范](#提交规范)

## 行为准则

本项目遵循 [Contributor Covenant Code of Conduct](https://github.com/joyehuang/minimind-notes/blob/main/CODE_OF_CONDUCT.md)。参与项目时，请保持尊重和包容。

## 如何贡献

### 报告 Bug

如果你发现了 bug：

1. **搜索现有 Issues**：确认问题尚未被报告
2. **创建 Issue**：使用 [Bug 报告模板](https://github.com/joyehuang/minimind-notes/issues/new?template=bug_report.md)
3. **提供详细信息**：
   - 清晰的 bug 描述
   - 复现步骤
   - 预期行为 vs 实际行为
   - 环境信息（OS、Python 版本、PyTorch 版本等）
   - 错误信息或截图

### 建议新功能

如果你有功能建议：

1. **搜索现有 Issues**：确认建议尚未被提出
2. **创建 Issue**：使用 [功能建议模板](https://github.com/joyehuang/minimind-notes/issues/new?template=feature_request.md)
3. **说明使用场景**：解释这个功能解决了什么问题
4. **提供实现思路**：如果有的话，分享你的想法

### 提交代码

#### 准备工作

1. **Fork 仓库**
2. **克隆你的 Fork**：
   ```bash
   git clone https://github.com/YOUR_USERNAME/minimind-notes.git
   cd minimind-notes
   ```
3. **创建分支**：
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

#### 开发流程

1. **遵循代码规范**（见下方）
2. **编写清晰的代码**：添加必要的中文注释
3. **确保可运行**：如果是实验代码，确保可以独立运行
4. **固定随机种子**：确保实验结果可复现
5. **更新文档**：如果修改了功能，记得更新相关文档

#### 提交 Pull Request

1. **提交前检查**：
   - [ ] 代码遵循项目风格
   - [ ] 实验可以独立运行
   - [ ] 结果可复现（固定随机种子）
   - [ ] 更新了相关文档
   - [ ] 没有引入新的错误

2. **推送并创建 PR**：
   ```bash
   git push origin feature/your-feature-name
   ```
   然后在 GitHub 上创建 Pull Request，使用 [PR 模板](.github/PULL_REQUEST_TEMPLATE.md)

3. **等待审查**：维护者会审查你的 PR，可能会提出修改建议

### 改进文档

文档改进包括：
- 修正错误或不清楚的描述
- 添加缺失的信息
- 改进示例代码
- 翻译（中英文）

**流程**：
1. 使用 [文档改进模板](https://github.com/joyehuang/minimind-notes/issues/new?template=documentation.md) 创建 Issue
2. 或直接提交 PR 修改文档

### 建议新实验

我们特别欢迎新的对照实验建议！

**好的实验应该**：
- 回答一个明确的问题（例如："为什么选择 X 而不是 Y？"）
- 有清晰的对照组和实验组
- 可以在合理时间内运行（学习阶段实验 < 10 分钟）
- 有可量化的评估指标

**流程**：
1. 使用 [实验建议模板](https://github.com/joyehuang/minimind-notes/issues/new?template=experiment_suggestion.md) 创建 Issue
2. 讨论实验设计
3. 实现实验并提交 PR

## 开发流程

### 项目结构

```
minimind-notes/
├── modules/                    # 模块化教学
│   ├── 01-foundation/         # 基础组件
│   │   ├── 01-normalization/
│   │   │   ├── README.md
│   │   │   ├── teaching.md    # 教学文档
│   │   │   ├── code_guide.md  # 代码指南
│   │   │   ├── quiz.md        # 自测题
│   │   │   └── experiments/   # 实验代码
│   │   └── ...
│   └── 02-architecture/       # 架构组装
├── docs/                       # 文档
└── ...
```

### 添加新模块

如果你想添加新模块：

1. **创建模块目录**：遵循命名规范 `XX-module-name`
2. **添加必要文件**：
   - `README.md` - 模块导航
   - `teaching.md` - 教学文档（Why/What/How）
   - `code_guide.md` - 代码指南
   - `quiz.md` - 自测题
   - `experiments/` - 实验代码
3. **更新主 README**：在模块导航部分添加新模块

### 添加新实验

实验代码应该：

1. **独立可运行**：不依赖其他实验
2. **固定随机种子**：确保可复现
   ```python
   import torch
   torch.manual_seed(42)
   ```
3. **清晰的注释**：解释关键步骤
4. **预期输出**：在 `results/` 目录保存预期结果
5. **命名规范**：`exp{N}_{description}.py`

## 代码规范

### Python 代码

- **风格**：遵循 PEP 8（可以使用 `black` 格式化）
- **注释**：关键步骤添加中文注释
- **类型提示**：鼓励但不强制
- **文档字符串**：函数和类添加 docstring

### 实验代码示例

```python
"""
实验 1: 梯度消失问题演示

这个实验展示了为什么需要归一化层。
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 固定随机种子，确保可复现
torch.manual_seed(42)

# 你的实验代码...
```

### 文档规范

- **Markdown 格式**：使用标准 Markdown
- **中文为主**：教学文档使用中文，代码注释也使用中文
- **结构清晰**：使用标题、列表、代码块等
- **示例代码**：提供可运行的示例

## 提交规范

### Commit Message

使用清晰的 commit message：

```
类型: 简短描述

详细说明（可选）
```

**类型**：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具相关
- `experiment`: 新实验

**示例**：
```
feat: 添加 RMSNorm vs LayerNorm 对比实验

- 添加 exp2_layernorm_vs_rmsnorm.py
- 更新 teaching.md 添加实验结果
- 添加可视化图表
```

### Pull Request

- **标题清晰**：说明 PR 做了什么
- **描述详细**：使用 PR 模板
- **关联 Issue**：如果相关，使用 `Closes #123`
- **小步提交**：一个 PR 专注于一个功能/修复

## 🎯 贡献类型

我们特别欢迎以下类型的贡献：

### 🌟 高优先级

- ✨ **新的对照实验**：回答"为什么这样设计"的问题
- 📖 **文档改进**：让内容更清晰易懂
- 🌍 **英文翻译**：帮助国际用户

### 💡 也欢迎

- 🐛 Bug 修复
- 🎨 更好的可视化
- ⚡ 性能优化
- 📊 实验结果分享

## ❓ 需要帮助？

- **提问**：使用 [Question 模板](https://github.com/joyehuang/minimind-notes/issues/new?template=question.md) 创建 Issue
- **讨论**：在 [GitHub Discussions](https://github.com/joyehuang/minimind-notes/discussions) 发起讨论
- **查看文档**：访问 [在线文档](https://minimind.wiki)

## 🙏 致谢

感谢所有贡献者！你的参与让这个项目变得更好。

---

**准备好贡献了吗？** 🚀

1. Fork 仓库
2. 创建分支
3. 开始贡献！
