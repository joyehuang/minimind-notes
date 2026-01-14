# Paint Metrics 优化实施方案

本文档提供具体的代码修改方案和实施步骤。

## Phase 1: 紧急修复（立即执行）

### 1.1 优化 OG Image（最高优先级）

#### 问题
当前 `/public/og-image.png` 大小为 **7.1MB**，严重影响 LCP。

#### 解决方案 A: 使用在线工具优化

1. **使用 Squoosh.app**（推荐）
   - 访问 https://squoosh.app/
   - 上传 `public/og-image.png`
   - 选择 WebP 格式
   - 质量设置为 80-85
   - 下载优化后的图片

2. **使用 TinyPNG**
   - 访问 https://tinypng.com/
   - 上传图片
   - 下载压缩后的 PNG

#### 解决方案 B: 使用命令行工具

```bash
# 安装 sharp（Node.js 图片处理库）
npm install --save-dev sharp-cli

# 转换为 WebP 格式（推荐）
npx sharp-cli \
  --input public/og-image.png \
  --output public/og-image.webp \
  --format webp \
  --quality 85

# 同时生成 PNG fallback（压缩版）
npx sharp-cli \
  --input public/og-image.png \
  --output public/og-image-optimized.png \
  --format png \
  --compressionLevel 9

# 检查文件大小
ls -lh public/og-image*
```

#### 解决方案 C: 创建自动化脚本

创建 `scripts/optimize-images.js`:

```javascript
import sharp from 'sharp'
import { readdir, stat } from 'fs/promises'
import { join } from 'path'

async function optimizeImage(inputPath, outputPath, format = 'webp') {
  const image = sharp(inputPath)
  const metadata = await image.metadata()

  console.log(`Optimizing ${inputPath}...`)
  console.log(`Original size: ${metadata.size} bytes`)

  if (format === 'webp') {
    await image
      .webp({ quality: 85, effort: 6 })
      .toFile(outputPath)
  } else if (format === 'png') {
    await image
      .png({ compressionLevel: 9, effort: 10 })
      .toFile(outputPath)
  }

  const outputStat = await stat(outputPath)
  const reduction = ((1 - outputStat.size / metadata.size) * 100).toFixed(2)
  console.log(`Optimized size: ${outputStat.size} bytes (-${reduction}%)`)
}

// 优化 OG image
await optimizeImage(
  'public/og-image.png',
  'public/og-image.webp',
  'webp'
)

// 生成 PNG fallback
await optimizeImage(
  'public/og-image.png',
  'public/og-image-optimized.png',
  'png'
)
```

```bash
# 添加到 package.json
{
  "scripts": {
    "optimize:images": "node scripts/optimize-images.js"
  }
}

# 运行
npm run optimize:images
```

#### 修改配置文件使用优化后的图片

`.vitepress/config.ts`:

```typescript
// 修改前
['meta', { property: 'og:image', content: `${siteUrl}/og-image.png` }],

// 修改后 - 使用 WebP
['meta', { property: 'og:image', content: `${siteUrl}/og-image.webp` }],
// 添加 PNG fallback
['meta', { property: 'og:image', content: `${siteUrl}/og-image-optimized.png` }],
['meta', { property: 'og:image:type', content: 'image/webp' }],
['meta', { property: 'og:image:width', content: '1200' }],
['meta', { property: 'og:image:height', content: '630' }],
```

#### 预期效果
- 文件大小：7.1MB → 150-250KB（减少 95%+）
- LCP 改善：-2000 到 -3000ms

---

### 1.2 延迟加载 Google Analytics

#### 当前代码
`.vitepress/config.ts` 第 14-20 行：

```typescript
['script', { async: true, src: 'https://www.googletagmanager.com/gtag/js?id=G-7B7HTLDJ65' }],
['script', {}, `
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-7B7HTLDJ65');
`]
```

#### 修改方案

**选项 1: 简单延迟加载**（推荐）

```typescript
// 删除上面的代码，替换为：
['script', { type: 'text/javascript' }, `
  window.addEventListener('load', function() {
    (function() {
      var script = document.createElement('script');
      script.src = 'https://www.googletagmanager.com/gtag/js?id=G-7B7HTLDJ65';
      script.async = true;
      document.head.appendChild(script);

      script.onload = function() {
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-7B7HTLDJ65');
      };
    })();
  });
`]
```

**选项 2: 使用 requestIdleCallback**（最佳性能）

```typescript
['script', { type: 'text/javascript' }, `
  (function() {
    function loadGA() {
      var script = document.createElement('script');
      script.src = 'https://www.googletagmanager.com/gtag/js?id=G-7B7HTLDJ65';
      script.async = true;
      document.head.appendChild(script);

      script.onload = function() {
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-7B7HTLDJ65');
      };
    }

    if ('requestIdleCallback' in window) {
      requestIdleCallback(loadGA, { timeout: 2000 });
    } else {
      setTimeout(loadGA, 2000);
    }
  })();
`]
```

#### 预期效果
- FCP 改善：-200 到 -400ms
- 不影响数据收集准确性

---

### 1.3 延迟加载 Vercel Analytics

#### 当前代码
`.vitepress/theme/index.ts` 第 14-18 行：

```typescript
// 注入 Vercel Analytics 和 Speed Insights
if (typeof window !== 'undefined') {
  inject()
  injectSpeedInsights()
}
```

#### 修改后

```typescript
// 延迟注入 Vercel Analytics 和 Speed Insights
if (typeof window !== 'undefined') {
  // 选项 1: 使用 window.load 事件
  window.addEventListener('load', () => {
    inject()
    injectSpeedInsights()
  })

  // 选项 2: 使用 requestIdleCallback（推荐）
  /*
  if ('requestIdleCallback' in window) {
    requestIdleCallback(() => {
      inject()
      injectSpeedInsights()
    }, { timeout: 2000 })
  } else {
    setTimeout(() => {
      inject()
      injectSpeedInsights()
    }, 2000)
  }
  */
}
```

#### 预期效果
- FCP 改善：-150 到 -250ms
- 不影响分析功能

---

## Phase 2: 重要优化

### 2.1 添加资源预加载

#### 修改 `.vitepress/config.ts`

在 `sharedHead` 数组最前面添加：

```typescript
const sharedHead = [
  // === 资源预加载优化 ===
  // DNS Prefetch - 提前解析域名
  ['link', { rel: 'dns-prefetch', href: 'https://www.googletagmanager.com' }],
  ['link', { rel: 'dns-prefetch', href: 'https://vercel.live' }],
  ['link', { rel: 'dns-prefetch', href: 'https://va.vercel-scripts.com' }],

  // Preconnect - 建立连接（包括 DNS、TCP、TLS）
  ['link', { rel: 'preconnect', href: 'https://www.googletagmanager.com', crossorigin: '' }],
  ['link', { rel: 'preconnect', href: 'https://vercel.live', crossorigin: '' }],

  // 预加载关键资源
  ['link', { rel: 'preload', href: '/logo.svg', as: 'image', type: 'image/svg+xml' }],

  // 如果使用了自定义字体，预加载字体
  // ['link', {
  //   rel: 'preload',
  //   href: '/fonts/custom-font.woff2',
  //   as: 'font',
  //   type: 'font/woff2',
  //   crossorigin: 'anonymous'
  // }],

  // 原有配置...
  ['link', { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }],
  // ...
]
```

#### 预期效果
- 减少第三方资源加载时间：-100 到 -200ms
- FCP 改善：-50 到 -150ms

---

### 2.2 懒加载非首屏组件

#### 方案 A: 使用 Vue 的 defineAsyncComponent

修改 `index.md`:

```vue
---
layout: home
# ... 其他配置
---

<script setup>
import { defineAsyncComponent } from 'vue'

// 首屏组件保持同步加载
import FeaturesCards from './.vitepress/theme/components/FeaturesCards.vue'
import LearningPathCards from './.vitepress/theme/components/LearningPathCards.vue'

// 非首屏组件懒加载
const ModulesGrid = defineAsyncComponent(() =>
  import('./.vitepress/theme/components/ModulesGrid.vue')
)
const TerminalCode = defineAsyncComponent(() =>
  import('./.vitepress/theme/components/TerminalCode.vue')
)
</script>

<FeaturesCards />
<LearningPathCards />
<ModulesGrid />
<TerminalCode />

<!-- 其他内容 -->
```

#### 方案 B: 使用 Intersection Observer（推荐）

创建新组件 `.vitepress/theme/components/LazyComponent.vue`:

```vue
<template>
  <div ref="root">
    <component v-if="isVisible" :is="component" v-bind="$attrs" />
    <div v-else class="lazy-placeholder" :style="{ minHeight: minHeight }">
      <div class="loading-spinner"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, defineAsyncComponent } from 'vue'

const props = defineProps({
  loader: {
    type: Function,
    required: true
  },
  minHeight: {
    type: String,
    default: '400px'
  },
  rootMargin: {
    type: String,
    default: '200px'
  }
})

const root = ref(null)
const isVisible = ref(false)
const component = defineAsyncComponent(props.loader)

onMounted(() => {
  const observer = new IntersectionObserver(
    (entries) => {
      if (entries[0].isIntersecting) {
        isVisible.value = true
        observer.disconnect()
      }
    },
    {
      rootMargin: props.rootMargin
    }
  )

  if (root.value) {
    observer.observe(root.value)
  }
})
</script>

<style scoped>
.lazy-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--vp-c-divider);
  border-top-color: var(--vp-c-brand-1);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
```

注册全局组件 `.vitepress/theme/index.ts`:

```typescript
import LazyComponent from './components/LazyComponent.vue'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    // 注册懒加载包装器
    app.component('LazyComponent', LazyComponent)

    // 其他组件...
  }
}
```

修改 `index.md` 使用懒加载：

```vue
<FeaturesCards />
<LearningPathCards />

<LazyComponent
  :loader="() => import('./.vitepress/theme/components/ModulesGrid.vue')"
  min-height="500px"
  root-margin="300px"
/>

<LazyComponent
  :loader="() => import('./.vitepress/theme/components/TerminalCode.vue')"
  min-height="400px"
  root-margin="300px"
/>
```

#### 预期效果
- JS bundle 减少：-50 到 -100KB
- LCP 改善：-300 到 -600ms
- TTI 改善：-400 到 -800ms

---

### 2.3 优化首页组件

#### 减少 ModulesGrid 复杂度

如果 `ModulesGrid.vue` 过于复杂（717 行），考虑简化：

1. **移除复杂动画**
2. **使用 CSS Grid 代替 JS 计算**
3. **延迟加载图片**

示例优化（修改 `ModulesGrid.vue`）:

```vue
<template>
  <section class="modules-section">
    <div class="modules-grid">
      <div
        v-for="(module, index) in modules"
        :key="module.title"
        class="module-card"
      >
        <!-- 移除复杂的 :style 计算 -->
        <img
          v-if="module.image"
          :src="module.image"
          :alt="module.title"
          loading="lazy"
          decoding="async"
        >
        <h3>{{ module.title }}</h3>
        <p>{{ module.description }}</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
/* 使用 CSS Grid，移除 JS 计算 */
.modules-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
}

/* 使用 CSS 动画代替 JS */
.module-card {
  animation: fadeInUp 0.6s ease-out backwards;
  animation-delay: calc(var(--index, 0) * 0.1s);
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
```

#### 预期效果
- 减少组件渲染时间
- 减少 JS 执行时间

---

## Phase 3: 持续优化

### 3.1 CSS 优化

#### 方案 1: 启用 CSS 代码分割

`.vitepress/config.ts`:

```typescript
export default defineConfig({
  // ... 其他配置

  vite: {
    build: {
      // 启用 CSS 代码分割
      cssCodeSplit: true,

      // 压缩配置
      minify: 'terser',
      terserOptions: {
        compress: {
          drop_console: true,
          drop_debugger: true,
          pure_funcs: ['console.log', 'console.info']
        }
      },

      // Rollup 优化
      rollupOptions: {
        output: {
          manualChunks: {
            // 将大型组件分离到单独的 chunk
            'vue-components': [
              '.vitepress/theme/components/ModulesGrid.vue',
              '.vitepress/theme/components/InteractiveQuiz.vue',
              '.vitepress/theme/components/TerminalCode.vue'
            ]
          }
        }
      }
    },

    // 优化依赖预构建
    optimizeDeps: {
      include: ['vue', '@vercel/analytics', '@vercel/speed-insights']
    }
  }
})
```

#### 方案 2: 拆分关键 CSS

创建 `.vitepress/theme/critical.css`:

```css
/* 只包含首屏必需的样式 */
:root {
  --vp-c-brand-1: #3b82f6;
  --vp-c-brand-2: #2563eb;
  /* ... 关键变量 */
}

/* Hero 区域样式 */
.VPHero {
  /* ... */
}

/* 基础布局 */
.VPNav, .VPContent {
  /* ... */
}
```

修改 `.vitepress/theme/index.ts`:

```typescript
import './critical.css'  // 关键 CSS 同步加载

if (typeof window !== 'undefined') {
  // 非关键 CSS 异步加载
  window.addEventListener('load', () => {
    import('./style.css')
  })
}
```

---

### 3.2 静态化结构化数据

#### 问题
当前在 `Layout.vue` 中动态注入 Breadcrumb Schema。

#### 解决方案
使用 VitePress 的 `transformHead` API：

修改 `.vitepress/config.ts`:

```typescript
import { generateBreadcrumbSchema } from './theme/utils/schema'

export default defineConfig({
  // ... 其他配置

  transformHead: ({ pageData }) => {
    const route = toRoute(pageData.relativePath)
    const isEn = route.startsWith('/en/')

    // 生成面包屑结构化数据
    const breadcrumbSchema = generateBreadcrumbSchema(route, isEn)

    return [
      // 原有的 canonical 和 alternate links
      ['link', { rel: 'canonical', href: `${siteUrl}${route}` }],
      // ...

      // 添加面包屑 Schema
      ...(breadcrumbSchema ? [
        ['script', {
          type: 'application/ld+json',
          'data-schema': 'breadcrumb'
        }, JSON.stringify(breadcrumbSchema)]
      ] : [])
    ]
  }
})
```

创建 `.vitepress/theme/utils/schema.ts`:

```typescript
export function generateBreadcrumbSchema(route: string, isEn: boolean) {
  // 如果是首页，不需要面包屑
  if (route === '/' || route === '/en/') {
    return null
  }

  const paths = route.split('/').filter(Boolean)
  const baseUrl = 'https://minimind.wiki'

  const items = paths.map((path, index) => {
    const position = index + 1
    const url = `${baseUrl}/${paths.slice(0, position).join('/')}`

    return {
      '@type': 'ListItem',
      position,
      name: getPageTitle(path, isEn),
      item: url
    }
  })

  // 添加首页
  items.unshift({
    '@type': 'ListItem',
    position: 1,
    name: isEn ? 'Home' : '首页',
    item: baseUrl
  })

  return {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: items
  }
}

function getPageTitle(path: string, isEn: boolean): string {
  // 根据路径返回页面标题
  const titleMap = {
    'docs': isEn ? 'Docs' : '文档',
    'modules': isEn ? 'Modules' : '模块',
    'guide': isEn ? 'Guide' : '指南',
    // ... 添加更多映射
  }

  return titleMap[path] || path
}
```

然后在 `.vitepress/theme/Layout.vue` 中删除动态注入逻辑：

```vue
<template>
  <Layout>
    <template #doc-before>
      <Breadcrumbs />
    </template>
    <template #doc-after>
      <GitHubFooter />
    </template>
  </Layout>
</template>

<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import GitHubFooter from './components/GitHubFooter.vue'
import Breadcrumbs from './components/Breadcrumbs.vue'

const { Layout } = DefaultTheme

// 删除所有 Schema 注入逻辑
</script>
```

---

### 3.3 HTTP 缓存优化

修改 `vercel.json`:

```json
{
  "buildCommand": "npm run docs:build",
  "outputDirectory": ".vitepress/dist",
  "installCommand": "npm ci",
  "framework": "vitepress",
  "cleanUrls": true,

  "headers": [
    {
      "source": "/assets/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        }
      ]
    },
    {
      "source": "/(.*\\.(jpg|jpeg|png|webp|avif|gif|svg|ico))",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        },
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        }
      ]
    },
    {
      "source": "/(.*\\.(woff|woff2|ttf|otf|eot))",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        },
        {
          "key": "Access-Control-Allow-Origin",
          "value": "*"
        }
      ]
    },
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "X-DNS-Prefetch-Control",
          "value": "on"
        },
        {
          "key": "X-Frame-Options",
          "value": "SAMEORIGIN"
        }
      ]
    }
  ],

  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/$1"
    }
  ]
}
```

---

## 验证和测试

### 本地测试

```bash
# 构建生产版本
npm run docs:build

# 预览
npm run docs:preview

# 使用 Lighthouse CLI 测试
npm install -g lighthouse
lighthouse http://localhost:4173 --view --preset=desktop
lighthouse http://localhost:4173 --view --preset=mobile
```

### 在线测试

1. **部署到 Vercel**
2. **使用 PageSpeed Insights**
   ```
   https://pagespeed.web.dev/analysis?url=https://minimind.wiki
   ```

3. **使用 WebPageTest**
   ```
   https://www.webpagetest.org/
   设置：
   - Location: China (如果目标用户在中国)
   - Connection: 4G
   ```

### 性能指标对比

在优化前后记录以下指标：

| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| FCP | ? ms | ? ms | ? ms |
| LCP | ? ms | ? ms | ? ms |
| TTI | ? ms | ? ms | ? ms |
| TBT | ? ms | ? ms | ? ms |
| CLS | ? | ? | ? |
| Lighthouse Score | ? | ? | +? |

---

## 实施检查清单

### Phase 1（必须完成）
- [ ] 优化 og-image.png（7.1MB → 200KB）
- [ ] 延迟加载 Google Analytics
- [ ] 延迟加载 Vercel Analytics/Speed Insights
- [ ] 添加 DNS prefetch 和 preconnect
- [ ] 构建并测试

### Phase 2（强烈建议）
- [ ] 实现组件懒加载（Intersection Observer）
- [ ] 注册 LazyComponent 全局组件
- [ ] 修改 index.md 使用懒加载
- [ ] 简化大型组件（ModulesGrid、TerminalCode）
- [ ] 测试性能改善

### Phase 3（可选）
- [ ] 启用 CSS 代码分割
- [ ] 拆分关键 CSS
- [ ] 静态化 Breadcrumb Schema
- [ ] 配置 HTTP 缓存头
- [ ] 配置 Vite 优化选项

### 最终验证
- [ ] 本地 Lighthouse 测试（Desktop & Mobile）
- [ ] PageSpeed Insights 测试
- [ ] WebPageTest 测试（真实网络环境）
- [ ] 验证 Core Web Vitals 达标
  - [ ] FCP < 1.8s
  - [ ] LCP < 2.5s
  - [ ] CLS < 0.1

---

## 遇到问题？

### 常见问题

**Q: 优化后 Google Analytics 数据会丢失吗？**
A: 不会。延迟加载只是推迟初始化时间，不影响数据收集。

**Q: 组件懒加载会影响 SEO 吗？**
A: 不会。VitePress 是 SSG（静态站点生成），HTML 内容在构建时就已生成。

**Q: WebP 图片在旧浏览器不支持怎么办？**
A: 可以使用 `<picture>` 标签提供 PNG fallback：

```html
<picture>
  <source srcset="/og-image.webp" type="image/webp">
  <img src="/og-image-optimized.png" alt="OG Image">
</picture>
```

**Q: 如何确认优化有效？**
A: 使用 Chrome DevTools Performance 面板录制页面加载，对比优化前后的 FCP、LCP 时间线。

---

## 下一步

完成以上优化后，建议：

1. **持续监控**：使用 Vercel Analytics 和 Google Analytics 监控 Core Web Vitals
2. **A/B 测试**：对比不同优化方案的效果
3. **渐进式增强**：根据用户反馈逐步优化
4. **定期审查**：每季度审查一次性能指标

---

## 参考资源

- [Web.dev - Optimize LCP](https://web.dev/optimize-lcp/)
- [VitePress Build Performance](https://vitepress.dev/guide/performance)
- [sharp 图片优化](https://sharp.pixelplumbing.com/)
- [Intersection Observer API](https://developer.mozilla.org/en-US/docs/Web/API/Intersection_Observer_API)
