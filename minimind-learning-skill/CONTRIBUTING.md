# 贡献指南 (Contributing Guide)

感谢你有兴趣为 MiniMind Learning Assistant Skill 做贡献！

## 📋 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发指南](#开发指南)
- [测试指南](#测试指南)
- [提交规范](#提交规范)

## 行为准则

本项目遵循 [Contributor Covenant](https://www.contributor-covenant.org/) 行为准则。参与者应当：

- ✅ 尊重他人观点
- ✅ 接受建设性批评
- ✅ 关注对社群最有利的事项
- ✅ 对其他社群成员表示同理心

## 如何贡献

### 报告 Bug

发现问题？请在 [GitHub Issues](https://github.com/joyehuang/minimind-notes/issues) 创建 Issue，包含：

1. **Bug 描述**：清晰简洁的描述
2. **重现步骤**：
   ```
   1. 进入 '...'
   2. 点击 '...'
   3. 看到错误
   ```
3. **期望行为**：应该发生什么
4. **实际行为**：实际发生了什么
5. **环境信息**：
   - OS：Windows/macOS/Linux
   - Python 版本
   - Claude Code 版本
   - MiniMind 版本

### 提出功能建议

想要新功能？请创建 Feature Request Issue，说明：

1. **问题背景**：这个功能解决什么问题？
2. **建议方案**：你希望如何实现？
3. **替代方案**：考虑过哪些其他方案？
4. **使用场景**：谁会用到这个功能？

### 提交代码

1. **Fork 仓库**
   ```bash
   git clone https://github.com/joyehuang/minimind-notes.git
   cd minimind-notes/minimind-learning-skill
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **进行修改**
   - 遵循[开发指南](#开发指南)
   - 添加必要的测试
   - 更新相关文档

4. **提交更改**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   # 遵循 Conventional Commits 规范
   ```

5. **推送并创建 PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   然后在 GitHub 创建 Pull Request

## 开发指南

### 环境设置

```bash
# 克隆仓库
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes/minimind-learning-skill

# 安装开发依赖
pip install -r requirements-dev.txt  # 如果有

# 设置 Claude Code Skill
ln -s $(pwd) ~/.claude/skills/minimind-learning
```

### 项目结构

```
minimind-learning-skill/
├── SKILL.md                 # Skill 核心定义（YAML + Markdown）
├── README.md                # 用户文档
├── CONTRIBUTING.md          # 本文件
├── templates/               # 笔记模板
│   ├── notes.md.template
│   ├── learning_log.md.template
│   ├── knowledge_base.md.template
│   └── learning_materials_readme.md.template
├── scripts/                 # 工具脚本
│   └── validate_notes.py   # 验证脚本
└── .gitignore
```

### 代码风格

**Markdown 文件**：
- 使用中文（用户文档）
- 英文仅用于代码和技术术语
- 使用 emoji 提高可读性（但不滥用）
- 代码块使用语法高亮

**Python 代码**：
- 遵循 PEP 8
- 使用类型提示（Python 3.10+）
- 详细的文档字符串（中文）

**SKILL.md**：
- 遵循 Claude Code Skill 规范
- 清晰的触发条件
- 详细的实现算法
- 丰富的示例

### 修改 SKILL.md

SKILL.md 是核心文件，修改时注意：

1. **触发系统**：
   - 添加新术语到 `MINIMIND_TERMS`
   - 更新触发关键词列表
   - 确保触发逻辑准确

2. **更新算法**：
   - 提供伪代码或 Python 示例
   - 解释算法意图
   - 考虑边界情况

3. **Git 自动化**：
   - Commit message 生成规则
   - 错误处理策略
   - 重试机制

### 修改模板

模板文件在 `templates/` 目录：

1. **保持一致性**：
   - 统一的 Markdown 格式
   - 一致的 emoji 使用
   - 清晰的章节结构

2. **占位符**：
   - 使用 `{TODAY}` 表示日期
   - 使用 `<!-- 注释 -->` 标记插入点

3. **示例内容**：
   - 提供清晰的示例
   - 展示期望格式
   - 帮助用户理解结构

## 测试指南

### 运行验证脚本

```bash
# 验证笔记一致性
python scripts/validate_notes.py

# 自动修复 Q 编号
python scripts/validate_notes.py --fix-numbering

# 指定笔记目录
python scripts/validate_notes.py --docs-dir /path/to/docs
```

### 手动测试场景

**场景 1：学习新概念**
```
1. 在 MiniMind 仓库中启动 Claude Code
2. 提问："什么是 RMSNorm？"
3. 验证：
   - learning_log.md 添加今日条目
   - knowledge_base.md 添加 Q&A
   - Git commit 生成正确
   - Git push 成功
```

**场景 2：解决问题**
```
1. 提问："训练时报错 CUDA OOM 怎么办？"
2. 验证：
   - learning_log.md 添加"遇到的问题"章节
   - 提取：错误现象、原因、解决方案
   - Commit message 包含"解决"
```

**场景 3：多轮对话**
```
1. 进行 3+ 轮深度对话
2. 包含代码块和公式
3. 验证：
   - 延迟 5 秒批量更新
   - 所有 Q&A 都被记录
   - 代码示例被创建（如适用）
```

**场景 4：显式请求**
```
1. 对话后说："记录这个知识点"
2. 验证：
   - 立即触发更新
   - 所有文件更新正确
```

### 验证清单

- [ ] Tier 1 触发（关键词检测）
- [ ] Tier 2 触发（深度对话）
- [ ] Tier 3 触发（显式请求）
- [ ] learning_log.md 日期章节正确
- [ ] knowledge_base.md Q 编号连续
- [ ] learning_materials/README.md 同步
- [ ] Git commit message 简洁准确
- [ ] Git push 成功
- [ ] 并发更新无冲突
- [ ] 错误处理正确（网络超时、权限等）

## 提交规范

### Commit Message 格式

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type**：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 格式调整（不影响代码逻辑）
- `refactor`: 重构（不是新功能也不是修复）
- `test`: 添加测试
- `chore`: 构建/工具链更新

**Scope**（可选）：
- `skill`: SKILL.md 相关
- `template`: 模板文件相关
- `script`: 脚本相关
- `docs`: 文档相关

**示例**：
```bash
feat(skill): add support for English notes
fix(template): correct date format in learning_log
docs(readme): update installation instructions
refactor(skill): simplify Q numbering algorithm
```

### Pull Request 指南

**PR 标题**：使用 Conventional Commits 格式

**PR 描述**：
```markdown
## 改动说明
简要描述你的改动

## 改动类型
- [ ] Bug 修复
- [ ] 新功能
- [ ] 文档更新
- [ ] 重构

## 测试
- [ ] 已在本地测试
- [ ] 添加了新测试（如适用）
- [ ] 所有测试通过

## 相关 Issue
Closes #123

## 截图（如适用）
[截图]
```

**代码审查**：
- 耐心等待维护者审查
- 积极响应反馈
- 及时更新 PR

## 发布流程

### 版本号规范

遵循 [Semantic Versioning](https://semver.org/)：

- **Major (1.0.0)**: 不兼容的 API 变更
- **Minor (0.1.0)**: 向后兼容的功能新增
- **Patch (0.0.1)**: 向后兼容的 Bug 修复

### 发布清单

- [ ] 更新版本号（SKILL.md, README.md）
- [ ] 更新 CHANGELOG.md
- [ ] 运行所有测试
- [ ] 创建 Git tag
- [ ] 发布到 GitHub Releases
- [ ] 公告（社群、博客）

## 社群

### 交流渠道

- **GitHub Issues**: Bug 报告和功能建议
- **GitHub Discussions**: 一般讨论
- **MiniMind 社群**: 内测和交流

### 认可贡献者

- 所有贡献者将被列入 CONTRIBUTORS.md
- 重大贡献将在 README.md 中特别感谢

## 许可证

通过贡献代码，你同意你的贡献将在 [MIT License](LICENSE) 下发布。

## 问题？

有疑问？请：
- 查看 [README.md](README.md)
- 查看 [GitHub Issues](https://github.com/joyehuang/minimind-notes/issues)
- 联系维护者：[@joyehuang](https://github.com/joyehuang)

---

**感谢你的贡献！** 🎉

让我们一起让 MiniMind 学习更轻松！
