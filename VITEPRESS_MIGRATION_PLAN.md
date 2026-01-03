# VitePress 迁移方案与计划

> 将 MiniMind 学习笔记迁移到 VitePress，同时保持现有学习工作流

---

## 🎯 迁移目标

1. ✅ 使用 VitePress 构建静态文档站点
2. ✅ 保持所有现有文件在原位置 (不破坏 Claude Code 工作流)
3. ✅ 部署到 Vercel (快速访问)
4. ✅ 可选部署到 GitHub Pages (备份)

---

## 📐 迁移方案: 方案 A+ (混合方案)

### 核心原则

| 原则 | 说明 | 好处 |
|------|------|------|
| **不移动现有文件** | 所有 .md 和代码保持原位 | Claude Code 继续工作 |
| **VitePress 作为展示层** | 通过 `srcDir: '.'` 读取根目录 | 代码可以直接运行 |
| **渐进式迁移** | 分步实施，每步都可以工作 | 风险低，易回滚 |
| **图片后置** | 先搭建框架，图片后续补充 | 快速上线 |

### 最终目录结构

```
minimind-notes/
├── .vitepress/                    # 新增: VitePress 配置
│   ├── config.ts                  # 主配置文件
│   ├── theme/
│   │   ├── index.ts               # 主题入口
│   │   ├── style.css              # 自定义样式
│   │   └── components/            # 自定义组件 (可选)
│   └── dist/                      # 构建输出 (git ignore)
│
├── docs/                          # 新增: VitePress 新内容
│   ├── index.md                   # 🏠 首页
│   ├── guide/                     # 📚 学习指南
│   │   ├── quick-start.md         # 快速开始
│   │   ├── systematic.md          # 系统学习
│   │   └── mastery.md             # 深度掌握
│   ├── reference/                 # 📖 参考文档
│   │   └── architecture.md        # 架构说明
│   └── public/                    # 静态资源
│       ├── logo.svg               # Logo
│       ├── favicon.ico            # 网站图标
│       └── images/                # 图片资源
│           ├── visualizations/    # Python 脚本输出图表
│           ├── screenshots/       # 截图
│           └── animations/        # GIF/视频
│
├── learning_log.md                # ✅ 保持原位
├── knowledge_base.md              # ✅ 保持原位
├── notes.md                       # ✅ 保持原位
├── CLAUDE.md                      # ✅ 保持原位
├── ROADMAP.md                     # ✅ 保持原位
├── NOTE_UPDATE_GUIDE.md           # ✅ 保持原位
│
├── modules/                       # ✅ 保持原位
│   ├── 01-foundation/
│   └── 02-architecture/
│
├── learning_materials/            # ✅ 保持原位
│   ├── rope_basics.py
│   └── ...
│
├── model/                         # ✅ MiniMind 原始代码
├── trainer/
├── dataset/
├── scripts/
│
├── package.json                   # 新增: npm 配置
├── package-lock.json              # 自动生成
├── vercel.json                    # 新增: Vercel 配置 (可选)
├── .gitignore                     # 更新: 忽略 node_modules, dist
└── .github/workflows/
    └── deploy-pages.yml           # 可选: GitHub Pages 部署
```

### VitePress 如何读取现有文件

**关键配置** (`.vitepress/config.ts`):

```typescript
export default defineConfig({
  // 源目录指向项目根目录
  srcDir: '.',

  // 输出目录
  outDir: '.vitepress/dist',

  // 侧边栏配置
  themeConfig: {
    sidebar: {
      '/': [
        {
          text: '我的学习笔记',
          items: [
            { text: '学习日志', link: '/learning_log' },      // → learning_log.md
            { text: '知识库', link: '/knowledge_base' },      // → knowledge_base.md
            { text: '总索引', link: '/notes' },               // → notes.md
          ]
        },
        {
          text: '模块化教学',
          items: [
            { text: '模块总览', link: '/modules/' },          // → modules/README.md
            {
              text: '基础组件',
              link: '/modules/01-foundation/',                // → modules/01-foundation/README.md
            }
          ]
        }
      ]
    }
  }
})
```

**工作原理**:
- VitePress 读取根目录的所有 `.md` 文件
- 链接路径 `/learning_log` → `learning_log.md`
- 链接路径 `/modules/` → `modules/README.md`
- 所有文件保持原位置，无需移动

---

## 📋 迁移计划 (分5个阶段)

### Phase 1: 基础设置 ⏱️ 30分钟

**目标**: 安装 VitePress，创建基础配置，确保本地可以运行

#### 任务清单

- [ ] **1.1** 初始化 npm 项目
  ```bash
  npm init -y
  npm install -D vitepress vue
  ```

- [ ] **1.2** 创建 `.vitepress/config.ts` 基础配置
  - srcDir: '.'
  - 基础导航和侧边栏
  - 中文本地化

- [ ] **1.3** 创建 `package.json` 脚本
  ```json
  {
    "scripts": {
      "docs:dev": "vitepress dev",
      "docs:build": "vitepress build",
      "docs:preview": "vitepress preview"
    }
  }
  ```

- [ ] **1.4** 更新 `.gitignore`
  ```
  node_modules/
  .vitepress/dist/
  .vitepress/cache/
  package-lock.json
  ```

- [ ] **1.5** 测试本地运行
  ```bash
  npm run docs:dev
  ```
  预期: 能访问 http://localhost:5173

#### 验收标准
✅ 本地可以运行 `npm run docs:dev`
✅ 能看到基本页面 (即使样式简陋)
✅ 侧边栏能显示 (即使链接还不完整)

---

### Phase 2: 首页和导航 ⏱️ 1小时

**目标**: 创建首页，配置完整的导航和侧边栏

#### 任务清单

- [ ] **2.1** 创建 `docs/index.md` 首页
  - Hero 区域 (标题、介绍、快速入口)
  - Features 卡片 (6个特色)
  - 学习进度展示
  - 快速开始命令

- [ ] **2.2** 配置顶部导航
  ```typescript
  nav: [
    { text: '首页', link: '/' },
    { text: '📚 学习指南', link: '/guide/' },
    { text: '🧱 模块教学', link: '/modules/' },
    { text: '📝 我的笔记', link: '/learning_log' },
  ]
  ```

- [ ] **2.3** 配置完整侧边栏
  - 学习指南 (3个页面)
  - 我的学习笔记 (链接现有文件)
  - 模块化教学 (链接 modules/)
  - 代码示例 (链接 learning_materials/)
  - 参考文档 (链接 CLAUDE.md 等)

- [ ] **2.4** 创建学习指南占位页面
  - `docs/guide/quick-start.md` (从 ROADMAP.md 提取)
  - `docs/guide/systematic.md` (从 ROADMAP.md 提取)
  - `docs/guide/mastery.md` (从 ROADMAP.md 提取)

- [ ] **2.5** 测试所有链接
  - 点击每个导航项
  - 确保没有 404
  - 确保能访问现有 .md 文件

#### 验收标准
✅ 首页美观，有完整的 Hero 和 Features
✅ 顶部导航和侧边栏完整
✅ 所有链接都能正常工作
✅ 能访问 learning_log.md, knowledge_base.md, modules/ 等现有文件

---

### Phase 3: 内容优化 ⏱️ 1-2小时

**目标**: 优化现有内容的展示，添加元数据和样式

#### 任务清单

- [ ] **3.1** 为关键 .md 文件添加 frontmatter

  示例: `learning_log.md`
  ```yaml
  ---
  title: 学习日志
  description: 记录 MiniMind 学习历程中的每日进度、问题和思考
  prev:
    text: '总索引'
    link: '/notes'
  next:
    text: '知识库'
    link: '/knowledge_base'
  ---
  ```

- [ ] **3.2** 优化 markdown 格式
  - 检查内部链接 (改为相对路径)
  - 检查图片链接 (暂时可以保持原样)
  - 优化代码块 (添加语言标识)

- [ ] **3.3** 创建自定义样式
  - `.vitepress/theme/style.css`
  - 自定义颜色主题
  - 优化表格、代码块样式
  - 添加进度条样式

- [ ] **3.4** 添加 Logo 和 Favicon
  - 创建或使用现有 logo
  - 放到 `docs/public/logo.svg`
  - 放到 `docs/public/favicon.ico`

- [ ] **3.5** 配置搜索功能
  ```typescript
  search: {
    provider: 'local',
    options: {
      locales: {
        root: {
          translations: {
            button: { buttonText: '搜索文档' }
          }
        }
      }
    }
  }
  ```

- [ ] **3.6** 完善学习指南内容
  - 从 ROADMAP.md 提取并扩展内容
  - 添加更多说明和示例
  - 优化排版

#### 验收标准
✅ 所有关键页面有 frontmatter
✅ 链接都正确
✅ 搜索功能可用
✅ 网站有统一的视觉风格

---

### Phase 4: 可视化资源 ⏱️ 后续补充

**目标**: 为 Python 脚本生成可视化图表，丰富文档内容

> ⚠️ 这个阶段可以后续补充，不阻塞部署

#### 任务清单

- [ ] **4.1** 创建图片生成脚本

  创建 `scripts/generate_visualizations.py`:
  ```python
  """
  运行所有 learning_materials 脚本，保存图表到 docs/public/images/
  """
  import os
  import subprocess

  scripts = [
      'learning_materials/rope_basics.py',
      'learning_materials/rope_multi_frequency.py',
      'learning_materials/attention_explained.py',
      # ... 更多
  ]

  output_dir = 'docs/public/images/visualizations'
  os.makedirs(output_dir, exist_ok=True)

  for script in scripts:
      print(f"Running {script}...")
      # 修改脚本保存路径，或复制输出图片
      subprocess.run(['python', script])
  ```

- [ ] **4.2** 修改 learning_materials 脚本
  - 添加参数: `--output-dir` 指定图片保存位置
  - 或: 运行后手动复制图片到 `docs/public/images/`

- [ ] **4.3** 运行脚本生成图表
  ```bash
  python scripts/generate_visualizations.py
  ```

- [ ] **4.4** 组织图片文件
  ```
  docs/public/images/
  ├── visualizations/
  │   ├── rope_basics_output.png
  │   ├── rope_multi_frequency.png
  │   ├── attention_heatmap.png
  │   ├── normalization_comparison.png
  │   └── ...
  ├── screenshots/
  │   └── ...
  └── animations/
      └── ...
  ```

- [ ] **4.5** 在文档中引用图片

  示例: `modules/01-foundation/02-position-encoding/teaching.md`
  ```markdown
  ## RoPE 可视化

  ![RoPE 旋转模式](/images/visualizations/rope_basics_output.png)

  *图: RoPE 在不同频率下的旋转模式*
  ```

- [ ] **4.6** 为每个可视化添加说明
  - 图片标题
  - 关键观察点
  - 运行命令 (供读者本地复现)

#### 验收标准
✅ 关键可视化都有对应图片
✅ 图片在文档中正确显示
✅ 每个图片都有说明

---

### Phase 5: 部署 ⏱️ 30分钟

**目标**: 部署到 Vercel 和 (可选) GitHub Pages

#### 5A. Vercel 部署 (推荐)

- [ ] **5A.1** 创建 `vercel.json` (可选)
  ```json
  {
    "buildCommand": "npm run docs:build",
    "outputDirectory": ".vitepress/dist",
    "framework": "vitepress"
  }
  ```

- [ ] **5A.2** 在 Vercel 导入项目
  1. 访问 https://vercel.com
  2. "New Project" → 选择 GitHub 仓库
  3. Vercel 自动检测 VitePress
  4. 点击 "Deploy"

- [ ] **5A.3** 配置自定义域名 (可选)
  - Vercel Dashboard → Settings → Domains
  - 添加自定义域名
  - 配置 DNS

- [ ] **5A.4** 测试部署
  - 访问 Vercel 提供的 URL
  - 检查所有页面
  - 检查搜索功能

#### 5B. GitHub Pages 部署 (可选)

- [ ] **5B.1** 创建 GitHub Actions 配置

  `.github/workflows/deploy-pages.yml`:
  ```yaml
  name: Deploy to GitHub Pages

  on:
    push:
      branches: [main]

  permissions:
    contents: read
    pages: write
    id-token: write

  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-node@v4
          with:
            node-version: 20
            cache: npm
        - run: npm ci
        - run: npm run docs:build
        - uses: actions/upload-pages-artifact@v3
          with:
            path: .vitepress/dist

    deploy:
      needs: build
      runs-on: ubuntu-latest
      steps:
        - uses: actions/deploy-pages@v4
  ```

- [ ] **5B.2** 启用 GitHub Pages
  - GitHub 仓库 → Settings → Pages
  - Source: GitHub Actions

- [ ] **5B.3** 推送触发部署
  ```bash
  git push origin main
  ```

#### 验收标准
✅ Vercel 部署成功，可以访问
✅ (可选) GitHub Pages 部署成功
✅ 所有页面正常显示
✅ 搜索功能可用
✅ 图片资源正确加载

---

## 📊 整体时间估算

| 阶段 | 预计时间 | 可并行 |
|------|---------|--------|
| Phase 1: 基础设置 | 30分钟 | - |
| Phase 2: 首页和导航 | 1小时 | - |
| Phase 3: 内容优化 | 1-2小时 | - |
| Phase 4: 可视化资源 | 后续补充 | ✅ 可后置 |
| Phase 5: 部署 | 30分钟 | - |
| **总计** | **3-4小时** (不含 Phase 4) | |

---

## 🎯 里程碑

### Milestone 1: 本地运行 ✅
- Phase 1 完成
- 能在本地看到基本页面

### Milestone 2: 内容完整 ✅
- Phase 2-3 完成
- 所有现有内容都能正确展示
- 导航、搜索等功能完善

### Milestone 3: 上线部署 🚀
- Phase 5 完成
- 网站公开可访问

### Milestone 4: 完整体验 🎨
- Phase 4 完成
- 所有可视化图表都已添加
- 文档内容完整丰富

---

## 🚨 风险控制

### 风险1: 链接失效
**预防**:
- 每完成一个 Phase 都测试所有链接
- 使用相对路径

**应对**:
- 检查 VitePress 的链接重写规则
- 必要时添加 rewrites 配置

### 风险2: 构建失败
**预防**:
- 本地充分测试再推送
- 先在分支测试部署

**应对**:
- 检查 Vercel 构建日志
- 确保 Node.js 版本一致

### 风险3: 样式混乱
**预防**:
- 先使用默认主题
- 逐步添加自定义样式

**应对**:
- 使用浏览器开发者工具调试
- 参考 VitePress 官方文档

### 风险4: 现有工作流被破坏
**预防**:
- 不移动任何现有文件
- 使用 srcDir: '.' 读取根目录
- 充分测试 Claude Code 工作流

**应对**:
- 如果有问题，立即回滚
- 检查 CLAUDE.md 中的路径配置

---

## ✅ 验收清单 (最终检查)

### 功能性

- [ ] 首页美观，信息完整
- [ ] 所有导航链接有效
- [ ] 所有侧边栏链接有效
- [ ] 搜索功能可用
- [ ] 能访问所有现有 .md 文件
- [ ] 能访问所有 modules/ 内容
- [ ] 图片资源正确显示 (如果已添加)
- [ ] 移动端适配良好

### 性能

- [ ] 首次加载 < 3秒
- [ ] 页面切换流畅
- [ ] 搜索响应快速

### SEO

- [ ] 每个页面都有 title
- [ ] 每个页面都有 description (frontmatter)
- [ ] 生成 sitemap

### 兼容性

- [ ] Chrome 正常
- [ ] Firefox 正常
- [ ] Safari 正常
- [ ] 移动浏览器正常

### 工作流

- [ ] Claude Code 可以正常更新 learning_log.md
- [ ] Claude Code 可以正常更新 knowledge_base.md
- [ ] Python 脚本可以直接运行
- [ ] Git 提交历史完整

---

## 📝 附录: 关键文件清单

### 需要创建的文件

```
新文件:
├── .vitepress/config.ts           ⭐ 核心配置
├── .vitepress/theme/index.ts
├── .vitepress/theme/style.css
├── docs/index.md                  ⭐ 首页
├── docs/guide/quick-start.md
├── docs/guide/systematic.md
├── docs/guide/mastery.md
├── docs/public/logo.svg
├── docs/public/favicon.ico
├── package.json                   ⭐ npm 配置
├── vercel.json                    (可选)
└── .github/workflows/deploy-pages.yml  (可选)
```

### 需要修改的文件

```
修改:
├── .gitignore                     添加 node_modules, dist
├── learning_log.md                添加 frontmatter (可选)
├── knowledge_base.md              添加 frontmatter (可选)
└── modules/*/README.md            添加 frontmatter (可选)
```

### 保持不变的文件

```
不变:
├── CLAUDE.md                      ✅ 保持原样
├── learning_materials/            ✅ 保持原样
├── model/                         ✅ 保持原样
├── trainer/                       ✅ 保持原样
└── ...其他所有文件                ✅ 保持原样
```

---

## 🚀 准备开始

**下一步**:
1. 创建新的 Git 分支: `feature/vitepress-migration`
2. 开始 Phase 1: 基础设置
3. 每完成一个 Phase，提交一次
4. 所有 Phase 完成后，创建 Pull Request

**开始命令**:
```bash
# 创建新分支
git checkout -b feature/vitepress-migration

# 开始 Phase 1
# (按照上面的任务清单执行)
```

准备好了吗？🎯
