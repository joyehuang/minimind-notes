# VitePress 推荐结构 (方案 A+)

## 📋 为什么选择方案 A+？

基于你的项目特点:
1. ✅ **学习笔记持续更新** - learning_log.md, knowledge_base.md 还在写
2. ✅ **Claude Code 集成** - CLAUDE.md 依赖固定路径
3. ✅ **可执行代码** - learning_materials/ 需要直接运行
4. ✅ **双轨系统** - 个人笔记 + 模块化教学需要共存

**结论**: 保持所有文件在原位置,VitePress 作为"展示层"

---

## 🏗️ 目录结构设计

```
minimind-notes/                    # 项目根目录
├── .vitepress/                    # VitePress 配置 (新增)
│   └── config.ts
├── docs/                          # VitePress 文档根目录 (新增)
│   ├── .vitepress -> ../.vitepress  # 符号链接
│   ├── public/                    # 静态资源
│   │   ├── logo.svg
│   │   └── images/
│   ├── index.md                   # 首页
│   ├── guide/                     # 学习指南 (新建内容)
│   │   ├── quick-start.md
│   │   ├── systematic.md
│   │   └── mastery.md
│   └── reference/                 # 参考文档 (新建内容)
│       └── architecture.md
│
├── learning_log.md                # 保持原位置 ⭐
├── knowledge_base.md              # 保持原位置 ⭐
├── notes.md                       # 保持原位置 ⭐
├── CLAUDE.md                      # 保持原位置 ⭐
├── ROADMAP.md                     # 保持原位置 ⭐
├── modules/                       # 保持原位置 ⭐
│   ├── 01-foundation/
│   └── 02-architecture/
├── learning_materials/            # 保持原位置 ⭐
│   ├── rope_basics.py
│   └── ...
│
├── model/                         # MiniMind 原始代码
├── trainer/
└── dataset/
```

---

## 🔗 VitePress 如何引用现有文件？

### 方法 1: 直接引用 (推荐)

VitePress 可以直接读取根目录的文件!

**配置 `docs/.vitepress/config.ts`**:

```typescript
export default defineConfig({
  srcDir: '..',  // 源目录指向上一级 (项目根目录)

  // 这样 VitePress 可以访问根目录的所有 .md 文件
})
```

**导航配置**:

```typescript
sidebar: {
  '/': [
    {
      text: '📝 我的学习笔记',
      items: [
        { text: '学习日志', link: '/learning_log' },      // → ../learning_log.md
        { text: '知识库', link: '/knowledge_base' },      // → ../knowledge_base.md
        { text: '总索引', link: '/notes' },               // → ../notes.md
      ]
    },
    {
      text: '🧱 模块教学',
      items: [
        {
          text: '01 归一化',
          link: '/modules/01-foundation/01-normalization/'  // → ../modules/...
        },
        // ...
      ]
    }
  ]
}
```

### 方法 2: 符号链接 (备选)

如果方法 1 遇到问题,使用符号链接:

```bash
cd docs

# 链接学习笔记
ln -s ../learning_log.md ./learning-log.md
ln -s ../knowledge_base.md ./knowledge-base.md

# 链接模块
ln -s ../modules ./modules

# 链接学习材料
ln -s ../learning_materials ./learning-materials
```

---

## 📐 具体实施步骤

### Step 1: 创建 VitePress 配置

```bash
# 在项目根目录创建 .vitepress/
mkdir -p .vitepress
touch .vitepress/config.ts
```

### Step 2: 创建 docs/ 目录

```bash
mkdir -p docs/public
mkdir -p docs/guide
mkdir -p docs/reference
```

### Step 3: 配置 srcDir

**`.vitepress/config.ts`**:

```typescript
import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'MiniMind 学习笔记',
  description: '深入理解 LLM 训练的每个设计选择',

  // 🔑 关键配置: 源目录指向根目录
  srcDir: '.',  // 在根目录运行 VitePress
  outDir: '.vitepress/dist',

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '📝 我的笔记', link: '/learning_log' },
      { text: '🧱 模块', link: '/modules/' },
    ],

    sidebar: {
      '/': [
        {
          text: '🚀 学习指南',
          items: [
            { text: '学习路线图', link: '/ROADMAP' },
            { text: '快速开始', link: '/docs/guide/quick-start' },
          ]
        },
        {
          text: '📝 我的学习笔记',
          items: [
            { text: '📅 学习日志', link: '/learning_log' },
            { text: '📚 知识库', link: '/knowledge_base' },
            { text: '🗂️ 总索引', link: '/notes' },
          ]
        },
        {
          text: '🧱 模块化教学',
          collapsed: false,
          items: [
            { text: '模块总览', link: '/modules/' },
            {
              text: 'Foundation (基础)',
              collapsed: false,
              items: [
                {
                  text: '01 归一化',
                  link: '/modules/01-foundation/01-normalization/'
                },
                {
                  text: '02 位置编码',
                  link: '/modules/01-foundation/02-position-encoding/'
                },
                {
                  text: '03 注意力机制',
                  link: '/modules/01-foundation/03-attention/'
                },
                {
                  text: '04 前馈网络',
                  link: '/modules/01-foundation/04-feedforward/'
                },
              ]
            },
            {
              text: 'Architecture (架构)',
              items: [
                { text: '架构总览', link: '/modules/02-architecture/' },
              ]
            }
          ]
        },
        {
          text: '💻 代码示例',
          items: [
            { text: '示例总览', link: '/learning_materials/README' },
          ]
        },
        {
          text: '📖 参考文档',
          items: [
            { text: 'Claude 使用指南', link: '/CLAUDE' },
            { text: '笔记更新指南', link: '/NOTE_UPDATE_GUIDE' },
          ]
        }
      ]
    }
  },

  markdown: {
    math: true,
    lineNumbers: true,
  }
})
```

### Step 4: 创建首页

**`docs/index.md`** (这个文件在 docs/ 下):

```markdown
---
layout: home

hero:
  name: "MiniMind 学习笔记"
  text: "深入理解 LLM 训练的每个设计选择"
  tagline: 从零开始训练语言模型 | 理论+实验+实践
  actions:
    - theme: brand
      text: 📅 学习日志
      link: /learning_log
    - theme: alt
      text: 📚 知识库
      link: /knowledge_base
    - theme: alt
      text: 🧱 模块教学
      link: /modules/

features:
  - icon: 📝
    title: 学习日志
    details: 记录每日学习进度、问题和思考
    link: /learning_log

  - icon: 📚
    title: 知识库
    details: 系统化整理的技术知识和问答
    link: /knowledge_base

  - icon: 🧱
    title: 模块化教学
    details: 4个基础组件 + 2个架构模块
    link: /modules/

  - icon: 💻
    title: 代码示例
    details: 可执行的学习材料
    link: /learning_materials/README
---

## 🎯 当前学习进度

**阶段**: 第一阶段 - Transformer 核心组件学习中

- ✅ RMSNorm (归一化)
- ✅ RoPE (位置编码)
- ⏳ Attention (注意力机制)
- ⏳ FeedForward (前馈网络)

## 🚀 快速开始

::: code-group

```bash [激活环境]
source venv/bin/activate
```

```bash [运行示例]
python learning_materials/rope_basics.py
```

:::

## 📖 浏览方式

- **按时间**: [学习日志](/learning_log) - 看我的学习历程
- **按主题**: [知识库](/knowledge_base) - 查技术概念
- **按模块**: [模块教学](/modules/) - 系统学习
```

### Step 5: package.json 配置

```json
{
  "name": "minimind-notes",
  "scripts": {
    "docs:dev": "vitepress dev",
    "docs:build": "vitepress build",
    "docs:preview": "vitepress preview"
  },
  "devDependencies": {
    "vitepress": "^1.0.0",
    "vue": "^3.4.0"
  }
}
```

注意: 不需要 `vitepress dev docs`,直接 `vitepress dev` 即可!

---

## ✅ 这个方案的优势

### 1. 不破坏现有工作流 ✨

```bash
# 学习笔记继续在原位置更新
learning_log.md       # Claude Code 会更新这个文件
knowledge_base.md     # Claude Code 会更新这个文件

# CLAUDE.md 的路径指令继续有效
"更新 learning_log.md"  ✅ 路径正确
"添加到 knowledge_base.md" ✅ 路径正确
```

### 2. 代码可以直接运行 💻

```bash
# 示例代码路径不变
python learning_materials/rope_basics.py  ✅

# 实验代码路径不变
cd modules/01-foundation/01-normalization/experiments
python exp1_gradient_vanishing.py  ✅
```

### 3. VitePress 提供更好的浏览体验 🎨

```
访问 http://localhost:5173

- 漂亮的首页
- 强大的搜索
- 清晰的导航
- 数学公式渲染
- 代码高亮
```

### 4. 两套系统和平共处 🤝

```
学习时:
- 用 Claude Code 更新笔记
- 运行示例代码
- 提交 Git

复习/分享时:
- 用 VitePress 浏览
- 搜索知识点
- 部署到 GitHub Pages
```

---

## 🚫 不推荐方案 B 的原因

如果采用方案 B (把文件移到 docs/):

### 问题 1: 破坏 Claude Code 集成

```markdown
# CLAUDE.md 中的指令
"更新 learning_log.md"

# 如果文件移到 docs/learning_log.md
❌ 需要更新所有路径
❌ CLAUDE.md 变得复杂
❌ 半自动化学习流程被打断
```

### 问题 2: 代码路径混乱

```python
# learning_materials/rope_basics.py
from modules.common import ...  # ❌ 找不到 modules/

# 需要改成
import sys
sys.path.append('../..')
from modules.common import ...  # ✅ 但很丑
```

### 问题 3: Git 历史混乱

```bash
# 移动文件会丢失 Git 历史
git mv learning_log.md docs/learning_log.md

# Git blame 会断裂
# 贡献统计会错乱
```

---

## 📝 总结

### 对于你的项目

**推荐: 方案 A+**

| 特性 | 方案 A+ | 方案 B |
|------|---------|--------|
| 保持原有文件位置 | ✅ | ❌ |
| Claude Code 正常工作 | ✅ | ❌ (需大量修改) |
| 代码可以直接运行 | ✅ | ❌ (需修改路径) |
| VitePress 展示 | ✅ | ✅ |
| 学习流程不受影响 | ✅ | ❌ |
| 实施难度 | 低 | 高 |

### 实施建议

1. **第一步**: 创建 `.vitepress/config.ts` 和 `docs/index.md`
2. **第二步**: 配置 `srcDir: '.'` 让 VitePress 读取根目录
3. **第三步**: 本地测试 `npm run docs:dev`
4. **第四步**: 逐步添加新内容到 `docs/guide/`
5. **第五步**: 部署到 GitHub Pages

### 长期演进

```
阶段 1: 保持现状 + 添加 VitePress 展示层
   ↓
阶段 2: 学习完成后,考虑是否重组
   ↓
阶段 3: 如果需要,再迁移到完整的方案 B
```

---

## 🎯 下一步

准备好了吗? 我可以帮你:

1. ✅ 创建 `.vitepress/config.ts` 配置文件
2. ✅ 创建 `docs/index.md` 首页
3. ✅ 设置 `package.json`
4. ✅ 本地测试运行

想要我直接帮你实施吗?
