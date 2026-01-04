# VitePress 模块化迁移开发计划

> 按照模块化方式逐步迁移，每个模块独立开发、测试、提交

---

## 🎯 开发原则

1. **模块化开发** - 每个模块独立完成，可单独测试
2. **增量提交** - 每个模块完成后立即 commit
3. **持续验证** - 每次提交前本地测试
4. **不影响主分支** - 在新分支开发，不影响学习工作流

---

## 📋 模块列表 (共9个模块)

| 模块 | 任务 | 预计时间 | Commit 信息 |
|------|------|---------|------------|
| **M1** | 项目初始化 | 10分钟 | `初始化 VitePress 项目配置` |
| **M2** | 核心配置 | 20分钟 | `添加 VitePress 核心配置和导航` |
| **M3** | 首页开发 | 20分钟 | `创建 VitePress 首页` |
| **M4** | 学习指南 | 30分钟 | `添加学习指南页面` |
| **M5** | 静态资源 | 15分钟 | `添加 logo 和 favicon` |
| **M6** | 自定义样式 | 20分钟 | `添加自定义样式主题` |
| **M7** | 内容优化 | 30分钟 | `优化现有内容的 frontmatter` |
| **M8** | 部署配置 | 15分钟 | `添加 Vercel 部署配置` |
| **M9** | 测试验收 | 20分钟 | `完成迁移并验收` |

**总计**: 约 3 小时

---

## 📦 详细开发计划

### Module 1: 项目初始化 ⏱️ 10分钟

**目标**: 安装 VitePress 和必要依赖

#### 任务清单

- [ ] 创建 `package.json`
  ```json
  {
    "name": "minimind-notes",
    "version": "1.0.0",
    "description": "MiniMind 学习笔记 - VitePress 文档站点",
    "type": "module",
    "scripts": {
      "docs:dev": "vitepress dev",
      "docs:build": "vitepress build",
      "docs:preview": "vitepress preview"
    },
    "keywords": ["LLM", "MiniMind", "Learning Notes"],
    "author": "joyehuang",
    "license": "MIT"
  }
  ```

- [ ] 安装依赖
  ```bash
  npm install -D vitepress vue
  ```

- [ ] 更新 `.gitignore`
  ```
  # VitePress
  node_modules/
  .vitepress/dist/
  .vitepress/cache/
  package-lock.json
  ```

- [ ] 测试基础环境
  ```bash
  npx vitepress --version
  ```

#### 验收标准
- ✅ `package.json` 创建完成
- ✅ 依赖安装成功
- ✅ VitePress 可执行

#### Git Commit
```bash
git add package.json .gitignore
git commit -m "初始化 VitePress 项目配置

- 创建 package.json 配置 npm 脚本
- 安装 VitePress 和 Vue 依赖
- 更新 .gitignore 忽略 node_modules 和构建产物"
```

---

### Module 2: 核心配置 ⏱️ 20分钟

**目标**: 创建 VitePress 核心配置文件

#### 任务清单

- [ ] 创建目录结构
  ```bash
  mkdir -p .vitepress/theme
  mkdir -p docs/public
  ```

- [ ] 创建 `.vitepress/config.ts`
  - srcDir: '.'
  - 基础站点信息
  - 顶部导航配置
  - 侧边栏配置 (链接现有文件)
  - 搜索配置
  - Markdown 增强 (数学公式、行号)

- [ ] 创建 `.vitepress/theme/index.ts`
  ```typescript
  import DefaultTheme from 'vitepress/theme'
  import './style.css'

  export default DefaultTheme
  ```

#### 文件内容

**`.vitepress/config.ts`**: (完整配置，参考 `config-example.ts`)

关键配置:
```typescript
export default defineConfig({
  title: 'MiniMind 学习笔记',
  description: '深入理解 LLM 训练的每个设计选择',
  lang: 'zh-CN',

  srcDir: '.',  // 🔑 关键: 读取根目录
  outDir: '.vitepress/dist',

  themeConfig: {
    nav: [...],
    sidebar: {
      '/': [
        {
          text: '我的学习笔记',
          items: [
            { text: '学习日志', link: '/learning_log' },
            { text: '知识库', link: '/knowledge_base' },
          ]
        },
        // ... 更多
      ]
    }
  },

  markdown: {
    math: true,
    lineNumbers: true
  }
})
```

#### 验收标准
- ✅ `.vitepress/config.ts` 创建完成
- ✅ 导航和侧边栏配置正确
- ✅ 可以访问现有 .md 文件

#### Git Commit
```bash
git add .vitepress/
git commit -m "添加 VitePress 核心配置和导航

- 创建 .vitepress/config.ts 主配置文件
- 配置顶部导航和侧边栏
- 启用搜索、数学公式、代码行号
- 配置 srcDir 指向根目录以读取现有文件"
```

---

### Module 3: 首页开发 ⏱️ 20分钟

**目标**: 创建美观的首页

#### 任务清单

- [ ] 创建 `docs/index.md`
  - Hero 区域 (标题、介绍、快速入口按钮)
  - Features 卡片 (6个特色功能)
  - 学习进度展示
  - 学习路径卡片
  - 模块概览表格
  - 快速开始代码示例

- [ ] 添加自定义样式 (在 Module 3 中简单添加)

#### 文件内容

**`docs/index.md`**: (参考 `docs-index-example.md`)

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
  # ... 更多 features
---

## 🎯 当前学习进度
...
```

#### 验收标准
- ✅ 首页美观，布局合理
- ✅ Hero 区域显示正确
- ✅ Features 卡片完整
- ✅ 所有链接有效

#### Git Commit
```bash
git add docs/index.md
git commit -m "创建 VitePress 首页

- 添加 Hero 区域展示项目信息
- 添加 6 个 Features 卡片
- 显示当前学习进度
- 添加快速开始指南和学习路径展示"
```

---

### Module 4: 学习指南页面 ⏱️ 30分钟

**目标**: 创建三个学习路径的详细页面

#### 任务清单

- [ ] 创建 `docs/guide/` 目录
  ```bash
  mkdir -p docs/guide
  ```

- [ ] 创建 `docs/guide/quick-start.md` (从 ROADMAP.md 提取)
  - 30 分钟快速体验路径
  - 3 个关键实验
  - 环境准备步骤

- [ ] 创建 `docs/guide/systematic.md`
  - 6 小时系统学习路径
  - 4 个基础模块详细说明
  - 学习流程图

- [ ] 创建 `docs/guide/mastery.md`
  - 30 小时深度掌握路径
  - 完整训练流程
  - 进阶主题

#### 文件结构

```
docs/guide/
├── quick-start.md     # ⚡ 快速体验 (30分钟)
├── systematic.md      # 📚 系统学习 (6小时)
└── mastery.md         # 🎓 深度掌握 (30小时)
```

#### 验收标准
- ✅ 三个页面都创建完成
- ✅ 内容从 ROADMAP.md 提取并优化
- ✅ 链接到相关模块正确

#### Git Commit
```bash
git add docs/guide/
git commit -m "添加学习指南页面

- 创建快速体验路径 (30分钟)
- 创建系统学习路径 (6小时)
- 创建深度掌握路径 (30小时)
- 从 ROADMAP.md 提取并优化内容"
```

---

### Module 5: 静态资源 ⏱️ 15分钟

**目标**: 添加 logo 和 favicon

#### 任务清单

- [ ] 创建或获取 logo
  - 如果有现成的，复制到 `docs/public/logo.svg`
  - 如果没有，创建简单的 SVG logo

- [ ] 创建或获取 favicon
  - 放到 `docs/public/favicon.ico`

- [ ] 在 config.ts 中引用
  ```typescript
  themeConfig: {
    logo: '/logo.svg'
  },
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }]
  ]
  ```

- [ ] 创建图片目录结构
  ```bash
  mkdir -p docs/public/images/visualizations
  mkdir -p docs/public/images/screenshots
  mkdir -p docs/public/images/animations
  ```

#### 验收标准
- ✅ Logo 显示在导航栏
- ✅ Favicon 显示在浏览器标签
- ✅ 图片目录结构创建完成

#### Git Commit
```bash
git add docs/public/
git commit -m "添加 logo 和 favicon

- 添加网站 logo 到导航栏
- 添加 favicon 到浏览器标签
- 创建图片资源目录结构 (visualizations/screenshots/animations)"
```

---

### Module 6: 自定义样式 ⏱️ 20分钟

**目标**: 优化网站外观和样式

#### 任务清单

- [ ] 创建 `.vitepress/theme/style.css`
  - 自定义颜色主题
  - 优化表格样式
  - 优化代码块样式
  - 添加进度条样式
  - 优化卡片样式

- [ ] 更新 `.vitepress/theme/index.ts` 引用样式

#### 文件内容

**`.vitepress/theme/style.css`**:

```css
/**
 * VitePress 自定义样式
 */

/* 自定义颜色主题 */
:root {
  --vp-c-brand-1: #3b82f6;
  --vp-c-brand-2: #2563eb;
  --vp-c-brand-3: #1d4ed8;
}

/* 进度容器样式 */
.progress-container {
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  margin: 1rem 0;
}

/* 路径卡片样式 */
.path-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}

.path-cards > div {
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
}

/* 优化表格样式 */
table {
  width: 100%;
  border-collapse: collapse;
}

table th,
table td {
  padding: 0.75rem;
  border: 1px solid var(--vp-c-divider);
}

/* 优化代码块 */
div[class*='language-'] {
  margin: 1rem 0;
  border-radius: 8px;
}

/* 响应式优化 */
@media (max-width: 768px) {
  .path-cards {
    grid-template-columns: 1fr;
  }
}
```

#### 验收标准
- ✅ 颜色主题统一
- ✅ 表格和代码块美观
- ✅ 移动端适配良好

#### Git Commit
```bash
git add .vitepress/theme/
git commit -m "添加自定义样式主题

- 配置品牌颜色主题
- 优化表格和代码块样式
- 添加进度容器和卡片样式
- 优化移动端响应式布局"
```

---

### Module 7: 内容优化 ⏱️ 30分钟

**目标**: 为现有内容添加 frontmatter 和优化链接

#### 任务清单

- [ ] 为关键文件添加 frontmatter

  **`learning_log.md`**:
  ```yaml
  ---
  title: 学习日志
  description: 记录 MiniMind 学习历程中的每日进度、问题和思考
  ---
  ```

  **`knowledge_base.md`**:
  ```yaml
  ---
  title: 知识库
  description: 系统化整理的技术知识、概念解释和问答记录
  ---
  ```

  **`notes.md`**:
  ```yaml
  ---
  title: 学习笔记总索引
  description: MiniMind 学习笔记系统的导航入口
  ---
  ```

  **`ROADMAP.md`**:
  ```yaml
  ---
  title: 学习路线图
  description: 三条学习路径 - 快速体验/系统学习/深度掌握
  ---
  ```

- [ ] 检查并优化内部链接
  - 确保使用相对路径
  - 检查链接有效性

- [ ] 优化 modules/README.md 和子模块 README

#### 验收标准
- ✅ 关键文件都有 frontmatter
- ✅ 所有内部链接正确
- ✅ 页面标题和描述完整

#### Git Commit
```bash
git add learning_log.md knowledge_base.md notes.md ROADMAP.md modules/
git commit -m "优化现有内容的 frontmatter

- 为 learning_log.md 添加 frontmatter
- 为 knowledge_base.md 添加 frontmatter
- 为 notes.md 添加 frontmatter
- 为 ROADMAP.md 添加 frontmatter
- 优化内部链接格式"
```

---

### Module 8: 部署配置 ⏱️ 15分钟

**目标**: 配置 Vercel 部署

#### 任务清单

- [ ] 创建 `vercel.json` (可选，Vercel 通常自动识别)
  ```json
  {
    "buildCommand": "npm run docs:build",
    "outputDirectory": ".vitepress/dist",
    "installCommand": "npm ci",
    "framework": "vitepress"
  }
  ```

- [ ] 创建 `.github/workflows/deploy-pages.yml` (可选，GitHub Pages 备份)

- [ ] 更新 README.md 添加部署说明

#### 验收标准
- ✅ Vercel 配置文件创建
- ✅ (可选) GitHub Actions 配置

#### Git Commit
```bash
git add vercel.json
git commit -m "添加 Vercel 部署配置

- 创建 vercel.json 配置文件
- 指定构建命令和输出目录
- 准备部署到 Vercel"
```

---

### Module 9: 测试验收 ⏱️ 20分钟

**目标**: 完整测试所有功能

#### 任务清单

- [ ] 本地完整测试
  ```bash
  npm run docs:dev
  ```
  - 测试首页
  - 测试所有导航链接
  - 测试所有侧边栏链接
  - 测试搜索功能
  - 测试学习指南页面
  - 测试现有 .md 文件访问
  - 测试 modules/ 内容访问

- [ ] 构建测试
  ```bash
  npm run docs:build
  npm run docs:preview
  ```

- [ ] 移动端测试
  - 响应式布局
  - 导航菜单
  - 搜索功能

- [ ] 创建测试报告

#### 验收标准
- ✅ 所有链接有效
- ✅ 搜索功能正常
- ✅ 移动端适配良好
- ✅ 构建无错误

#### Git Commit
```bash
git commit -m "完成 VitePress 迁移并验收

- 完成本地功能测试
- 完成构建测试
- 完成移动端适配测试
- 所有功能验收通过"
```

---

## 🔄 开发流程

### 每个模块的开发流程

```
1. 切换到新分支
   ↓
2. 开发模块功能
   ↓
3. 本地测试
   ↓
4. Git commit (规范化)
   ↓
5. Push 到远程
   ↓
6. 继续下一个模块
```

### Git Commit 规范

**格式**:
```
<type>: <subject>

<body>
```

**示例**:
```bash
git commit -m "添加 VitePress 核心配置和导航

- 创建 .vitepress/config.ts 主配置文件
- 配置顶部导航和侧边栏
- 启用搜索、数学公式、代码行号
- 配置 srcDir 指向根目录以读取现有文件"
```

---

## ✅ 最终验收清单

### 功能性
- [ ] 首页美观且功能完整
- [ ] 所有导航链接有效
- [ ] 所有侧边栏链接有效
- [ ] 搜索功能可用
- [ ] 能访问所有现有 .md 文件
- [ ] 能访问所有 modules/ 内容
- [ ] 学习指南页面完整

### 样式和体验
- [ ] 颜色主题统一
- [ ] 表格和代码块美观
- [ ] 移动端适配良好
- [ ] 加载速度快

### 部署准备
- [ ] Vercel 配置完成
- [ ] 构建无错误
- [ ] 准备好部署

### 工作流兼容
- [ ] learning_log.md 可访问
- [ ] knowledge_base.md 可访问
- [ ] CLAUDE.md 路径未改变
- [ ] Python 脚本可运行

---

## 📝 开发注意事项

1. **不移动现有文件** - 所有现有 .md 保持在原位置
2. **测试后再提交** - 每次提交前确保本地测试通过
3. **规范化 commit** - 使用清晰的 commit 信息
4. **增量开发** - 一个模块一个模块完成
5. **保持简洁** - 不过度设计，先完成基础功能

---

## 🚀 准备开始

**开始命令**:
```bash
# (已在当前分支，准备开始 Module 1)
npm init -y
```

准备好开始 Module 1 了吗？
