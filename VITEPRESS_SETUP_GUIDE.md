# VitePress 设置指南

## 📋 实施步骤

### 方案选择

**推荐: 方案 A (最小改动方案)**
- ✅ 保留现有文件结构
- ✅ 通过符号链接或直接引用现有文件
- ✅ 快速上线,后续可以逐步优化

### Step 1: 安装 VitePress

```bash
# 在项目根目录
npm init -y
npm install -D vitepress vue
```

### Step 2: 创建 docs 目录结构

```bash
mkdir -p docs/.vitepress
mkdir -p docs/public

# 创建配置文件
touch docs/.vitepress/config.ts

# 创建首页
touch docs/index.md
```

### Step 3: 组织文档结构

**选项 A: 使用符号链接 (推荐,保持文件同步)**

```bash
cd docs

# 创建符号链接指向现有文件
ln -s ../modules ./modules
ln -s ../learning_log.md ./learning-log.md
ln -s ../knowledge_base.md ./knowledge-base.md
ln -s ../learning_materials ./learning-materials
ln -s ../ROADMAP.md ./guide/roadmap.md
```

**优点**:
- 保持单一数据源,修改自动同步
- 不需要复制文件
- Git 仍然追踪原始文件

**选项 B: 重组文件 (更清晰,但需要迁移)**

```bash
# 移动文件到 docs 目录
mv learning_log.md docs/notes/learning-log.md
mv knowledge_base.md docs/notes/knowledge-base.md
mv learning_materials docs/notes/materials
mv modules docs/modules

# 更新 Git
git add docs/
git commit -m "重组文档结构用于 VitePress"
```

### Step 4: 配置 VitePress

将 `.vitepress-config-example.ts` 的内容复制到 `docs/.vitepress/config.ts`

```bash
cp .vitepress-config-example.ts docs/.vitepress/config.ts
```

根据你选择的方案(A 或 B)调整配置文件中的路径。

### Step 5: 创建首页

将 `docs-index-example.md` 的内容复制到 `docs/index.md`

```bash
cp docs-index-example.md docs/index.md
```

### Step 6: 添加 npm 脚本

在 `package.json` 中添加:

```json
{
  "scripts": {
    "docs:dev": "vitepress dev docs",
    "docs:build": "vitepress build docs",
    "docs:preview": "vitepress preview docs"
  }
}
```

### Step 7: 本地预览

```bash
npm run docs:dev
```

浏览器访问 `http://localhost:5173`

### Step 8: 调整现有 Markdown 文件

VitePress 需要一些小调整:

**1. 添加 Frontmatter (可选但推荐)**

在每个 markdown 文件顶部添加:

```yaml
---
title: 页面标题
description: 页面描述
---
```

**2. 修复相对路径链接**

确保链接使用正确的相对路径:
- `[文本](./file.md)` - 同目录
- `[文本](../file.md)` - 上级目录
- `[文本](/path/to/file)` - 绝对路径(从 docs 根目录开始)

**3. 图片路径**

将图片放在 `docs/public/` 下:
```markdown
![alt](/images/demo.png)
```

### Step 9: 部署到 GitHub Pages

**配置 GitHub Actions**

创建 `.github/workflows/deploy.yml`:

```yaml
name: Deploy VitePress site to Pages

on:
  push:
    branches: [main]  # 或你的主分支名称
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - name: Install dependencies
        run: npm ci

      - name: Build with VitePress
        run: npm run docs:build

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: docs/.vitepress/dist

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    needs: build
    runs-on: ubuntu-latest
    name: Deploy
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

**启用 GitHub Pages**

1. 进入 GitHub 仓库 Settings → Pages
2. Source 选择 "GitHub Actions"
3. 保存

推送后会自动部署到: `https://joyehuang.github.io/minimind-notes`

---

## 🎨 自定义建议

### 1. Logo 和 Favicon

```bash
# 添加 logo
docs/public/logo.svg
docs/public/favicon.ico
```

在 `config.ts` 中引用:
```ts
themeConfig: {
  logo: '/logo.svg'
}
```

在 `docs/.vitepress/config.ts` 中添加:
```ts
head: [
  ['link', { rel: 'icon', href: '/favicon.ico' }]
]
```

### 2. 自定义样式

创建 `docs/.vitepress/theme/style.css`:

```css
/* 自定义颜色 */
:root {
  --vp-c-brand-1: #3b82f6;
  --vp-c-brand-2: #2563eb;
}

/* 自定义容器样式 */
.progress-container {
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  margin: 1rem 0;
}
```

创建 `docs/.vitepress/theme/index.ts`:

```ts
import DefaultTheme from 'vitepress/theme'
import './style.css'

export default DefaultTheme
```

### 3. 自定义组件

可以创建 Vue 组件来增强功能:

**学习进度组件** (`docs/.vitepress/components/LearningProgress.vue`):

```vue
<template>
  <div class="learning-progress">
    <h3>学习进度</h3>
    <div class="progress-bar">
      <div class="progress-fill" :style="{ width: progress + '%' }"></div>
    </div>
    <p>{{ completed }} / {{ total }} 模块完成</p>
  </div>
</template>

<script setup>
defineProps({
  completed: Number,
  total: Number
})

const progress = computed(() => (completed / total) * 100)
</script>

<style scoped>
.progress-bar {
  height: 20px;
  background: var(--vp-c-bg-soft);
  border-radius: 10px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--vp-c-brand-1);
  transition: width 0.3s ease;
}
</style>
```

在 markdown 中使用:

```markdown
<LearningProgress :completed="2" :total="4" />
```

### 4. 代码组高亮

VitePress 支持代码组:

```markdown
::: code-group

```python [RMSNorm 实现]
def rmsnorm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
\```

```python [LayerNorm 实现]
def layernorm(x, eps=1e-6):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)
\```

:::
```

### 5. 数学公式支持

已在配置中启用 KaTeX,可以直接使用:

```markdown
行内公式: $E = mc^2$

块级公式:
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma
$$
```

---

## 📂 最终目录结构

```
minimind-notes/
├── docs/                          # VitePress 根目录
│   ├── .vitepress/
│   │   ├── config.ts             # 主配置
│   │   ├── theme/
│   │   │   ├── index.ts          # 主题配置
│   │   │   ├── style.css         # 自定义样式
│   │   │   └── components/       # 自定义组件
│   │   └── dist/                 # 构建输出
│   ├── public/                   # 静态资源
│   │   ├── logo.svg
│   │   └── images/
│   ├── index.md                  # 首页
│   ├── guide/
│   │   ├── quick-start.md
│   │   ├── systematic.md
│   │   └── mastery.md
│   ├── modules/                  → 符号链接到 ../modules
│   ├── notes/
│   │   ├── learning-log.md       → 符号链接到 ../learning_log.md
│   │   ├── knowledge-base.md     → 符号链接到 ../knowledge_base.md
│   │   └── materials/            → 符号链接到 ../learning_materials
│   └── reference/
│       └── claude-guide.md       → 符号链接到 ../CLAUDE.md
├── .github/
│   └── workflows/
│       └── deploy.yml            # GitHub Actions 部署
├── package.json
├── [其他现有文件保持不变]
└── README.md
```

---

## 🔧 常见问题

### Q1: 符号链接在 Windows 上不工作?

使用硬拷贝或在 `config.ts` 中配置 vite 别名:

```ts
export default defineConfig({
  vite: {
    resolve: {
      alias: {
        '@modules': path.resolve(__dirname, '../../modules'),
        '@notes': path.resolve(__dirname, '../..')
      }
    }
  }
})
```

### Q2: 本地链接跳转不工作?

确保使用相对路径或绝对路径(从 docs 根目录开始):
- `[链接](./file)` - 同目录,推荐
- `[链接](/path/to/file)` - 绝对路径,推荐
- `[链接](file.md)` - 不推荐

### Q3: 如何添加评论功能?

可以集成 Giscus (基于 GitHub Discussions):

```ts
// docs/.vitepress/theme/index.ts
import Giscus from '@giscus/vue'

export default {
  ...DefaultTheme,
  enhanceApp({ app }) {
    app.component('Giscus', Giscus)
  }
}
```

### Q4: 如何优化搜索?

本地搜索已配置。如需更强大的搜索,可以集成 Algolia DocSearch (免费):

1. 申请: https://docsearch.algolia.com/apply/
2. 配置:
```ts
themeConfig: {
  search: {
    provider: 'algolia',
    options: {
      appId: 'YOUR_APP_ID',
      apiKey: 'YOUR_API_KEY',
      indexName: 'YOUR_INDEX_NAME'
    }
  }
}
```

---

## 📊 下一步

1. **本地测试**: `npm run docs:dev` 确保一切正常
2. **调整样式**: 根据个人喜好自定义主题
3. **添加内容**: 完善各个模块的文档
4. **部署上线**: 推送到 GitHub,自动部署
5. **分享链接**: 分享你的学习笔记网站!

---

## 🎯 优化建议

- **SEO**: 为每个页面添加 frontmatter (title, description)
- **性能**: 使用图片压缩和懒加载
- **可访问性**: 添加 alt 文本和 ARIA 标签
- **Analytics**: 集成 Google Analytics 或其他分析工具
- **进度追踪**: 使用自定义组件显示学习进度
- **互动性**: 添加可折叠的代码示例、可交互的图表

祝你搭建成功! 🚀
