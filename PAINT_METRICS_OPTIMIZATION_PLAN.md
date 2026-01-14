# Paint Metrics 优化计划

## 问题概述

当前网站的 **First Contentful Paint (FCP)** 和 **Largest Contentful Paint (LCP)** 指标较差，严重影响用户体验。根据代码分析，发现了多个关键性能瓶颈。

## 问题根因分析

### 🔴 严重问题（Critical - 影响最大）

#### 1. OG Image 文件过大（7.1MB）
**位置**: `/public/og-image.png`
**问题**:
- 当前大小：7.1MB
- 这是 LCP 的主要瓶颈
- 即使不在首屏显示，也会被预加载（meta 标签中引用）

**影响**:
- 延迟 LCP 指标 3-5 秒
- 占用大量带宽
- 移动端用户体验极差

**解决方案**:
1. 使用 image optimization 工具压缩图片
2. 转换为 WebP 格式（减少 70-80% 体积）
3. 提供多尺寸版本（响应式图片）
4. 目标大小：< 200KB

```bash
# 使用 sharp 或 ImageMagick 优化
npm install sharp
# 或使用在线工具如 TinyPNG, Squoosh.app
```

---

#### 2. Google Analytics 同步加载
**位置**: `.vitepress/config.ts` 第 14-20 行

**问题**:
```typescript
['script', { async: true, src: 'https://www.googletagmanager.com/gtag/js?id=G-7B7HTLDJ65' }],
['script', {}, `
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-7B7HTLDJ65');
`]
```

虽然第一个 script 标签有 `async: true`，但第二个内联脚本会立即执行，阻塞渲染。

**影响**:
- 阻塞首屏渲染
- 增加 FCP 时间 200-500ms

**解决方案**:
1. 延迟到页面加载完成后再初始化
2. 使用 Partytown 将分析脚本移到 Web Worker

```typescript
// 推荐方案：延迟加载
['script', { type: 'text/partytown' }, `
  window.addEventListener('load', () => {
    const script = document.createElement('script');
    script.src = 'https://www.googletagmanager.com/gtag/js?id=G-7B7HTLDJ65';
    script.async = true;
    document.head.appendChild(script);

    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-7B7HTLDJ65');
  });
`]
```

---

### 🟠 重要问题（High - 需要优化）

#### 3. 首页加载大量 Vue 组件
**位置**: `index.md`

**问题**:
```vue
<FeaturesCards />        <!-- 355 行代码 -->
<LearningPathCards />    <!-- 506 行代码 -->
<ModulesGrid />          <!-- 717 行代码 -->
<TerminalCode />         <!-- 421 行代码 -->
```

总计 **~2000 行 Vue 组件代码** 在首屏加载。

**影响**:
- 增加 JS bundle 大小
- 延迟交互时间（TTI）
- 增加 FCP 和 LCP 时间

**解决方案**:
1. **懒加载非首屏组件**：
```vue
<script setup>
import { defineAsyncComponent } from 'vue'

const ModulesGrid = defineAsyncComponent(() =>
  import('./components/ModulesGrid.vue')
)
const TerminalCode = defineAsyncComponent(() =>
  import('./components/TerminalCode.vue')
)
</script>
```

2. **使用 Intersection Observer 延迟加载**：
只在组件进入视口时才加载

3. **代码分割**：
确保 VitePress 正确进行代码分割

---

#### 4. 缺少关键资源预加载
**位置**: `.vitepress/config.ts` head 配置

**问题**:
- 没有 preconnect 到第三方域名
- 没有 DNS prefetch
- 没有预加载关键 CSS/JS

**影响**:
- 增加第三方资源加载时间
- 延迟 FCP 100-300ms

**解决方案**:
在 `sharedHead` 数组中添加：

```typescript
const sharedHead = [
  // DNS prefetch 和 preconnect
  ['link', { rel: 'dns-prefetch', href: 'https://www.googletagmanager.com' }],
  ['link', { rel: 'preconnect', href: 'https://www.googletagmanager.com', crossorigin: '' }],
  ['link', { rel: 'dns-prefetch', href: 'https://vercel.live' }],
  ['link', { rel: 'preconnect', href: 'https://vercel.live', crossorigin: '' }],

  // 预加载关键字体（如果有）
  // ['link', { rel: 'preload', as: 'font', type: 'font/woff2', href: '/fonts/...' }],

  // 现有配置...
]
```

---

#### 5. 第三方脚本未优化
**位置**: `.vitepress/theme/index.ts` 第 15-17 行

**问题**:
```typescript
if (typeof window !== 'undefined') {
  inject()                    // Vercel Analytics
  injectSpeedInsights()       // Vercel Speed Insights
}
```

这些在页面加载时立即执行，占用主线程。

**影响**:
- 延迟首屏渲染
- 增加 FCP 时间

**解决方案**:
延迟到页面加载完成后：

```typescript
if (typeof window !== 'undefined') {
  window.addEventListener('load', () => {
    inject()
    injectSpeedInsights()
  })
}
```

或使用 `requestIdleCallback`：

```typescript
if (typeof window !== 'undefined') {
  if ('requestIdleCallback' in window) {
    requestIdleCallback(() => {
      inject()
      injectSpeedInsights()
    })
  } else {
    setTimeout(() => {
      inject()
      injectSpeedInsights()
    }, 2000)
  }
}
```

---

### 🟡 中等问题（Medium - 可以改进）

#### 6. CSS 文件较大
**位置**: `.vitepress/theme/style.css`

**问题**:
- 404 行自定义 CSS
- 包含大量暗黑模式样式
- 可能有未使用的样式

**影响**:
- 增加首屏渲染时间
- 增加 CSS 解析时间

**解决方案**:
1. 使用 PurgeCSS 删除未使用的样式
2. 关键 CSS 内联，非关键 CSS 异步加载
3. 拆分 CSS 文件（基础样式 + 主题样式）

```typescript
// vitepress config
export default defineConfig({
  vite: {
    build: {
      cssCodeSplit: true
    }
  }
})
```

---

#### 7. 动态注入结构化数据
**位置**: `.vitepress/theme/Layout.vue` 第 24-40 行

**问题**:
```typescript
const injectBreadcrumbSchema = () => {
  const existingSchema = document.querySelector('script[data-schema="breadcrumb"]')
  if (existingSchema) {
    existingSchema.remove()
  }
  // 动态创建和插入 script 标签
}
```

每次路由变化都操作 DOM，可能影响性能。

**影响**:
- 轻微影响页面切换性能

**解决方案**:
在构建时生成静态结构化数据，使用 `transformHead`：

```typescript
export default defineConfig({
  transformHead: ({ pageData }) => {
    const breadcrumbSchema = generateBreadcrumbSchema(pageData)
    return [
      ['script', { type: 'application/ld+json' }, JSON.stringify(breadcrumbSchema)]
    ]
  }
})
```

---

## 优化实施优先级

### Phase 1: 紧急修复（预计提升 40-50%）
1. ✅ **优化 og-image.png**（7.1MB → 200KB）
   - 预计 LCP 改善：-3000ms
2. ✅ **延迟 Google Analytics**
   - 预计 FCP 改善：-300ms
3. ✅ **延迟 Vercel 脚本**
   - 预计 FCP 改善：-200ms

### Phase 2: 重要优化（预计提升 20-30%）
4. ⏳ **添加资源预加载**
   - 预计改善：-200ms
5. ⏳ **懒加载非首屏组件**
   - 预计 LCP 改善：-500ms
   - 减少 JS bundle 大小：-50KB

### Phase 3: 持续优化（预计提升 10-20%）
6. ⏳ **优化 CSS**
7. ⏳ **静态化结构化数据**
8. ⏳ **启用 HTTP/2 Server Push**（Vercel 已支持）

---

## 性能目标

### 当前估计（未优化）
- FCP: ~2.5-3.5s
- LCP: ~4.5-6.0s
- TTI: ~4.0-5.0s

### 优化后目标
- FCP: < 1.5s ✅（Good）
- LCP: < 2.5s ✅（Good）
- TTI: < 3.0s ✅（Good）

### 参考标准（Google Core Web Vitals）
- FCP: Good < 1.8s, Needs Improvement 1.8-3.0s, Poor > 3.0s
- LCP: Good < 2.5s, Needs Improvement 2.5-4.0s, Poor > 4.0s

---

## 监控和验证

### 工具
1. **Lighthouse**（Chrome DevTools）
2. **PageSpeed Insights**（https://pagespeed.web.dev/）
3. **WebPageTest**（https://www.webpagetest.org/）
4. **Vercel Analytics**（已集成）

### 测试环境
- Desktop: Fast 3G
- Mobile: 4G
- 测试地区：中国大陆、美国

---

## 额外建议

### 1. 启用 VitePress 内置优化
```typescript
export default defineConfig({
  vite: {
    build: {
      minify: 'terser',
      terserOptions: {
        compress: {
          drop_console: true,
          drop_debugger: true
        }
      }
    }
  }
})
```

### 2. 启用 CDN 和缓存
```json
// vercel.json
{
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
      "source": "/(.*\\.(?:jpg|jpeg|png|gif|svg|webp|avif))",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        }
      ]
    }
  ]
}
```

### 3. 使用 WebP/AVIF 图片格式
- 自动生成多格式图片
- 使用 `<picture>` 标签提供降级方案

### 4. 考虑使用 Partytown
如果第三方脚本较多，可以使用 Partytown 将它们移到 Web Worker：

```bash
npm install @builder.io/partytown
```

---

## 实施计划时间表

| 任务 | 优先级 | 预计时间 | 负责人 | 状态 |
|------|--------|----------|--------|------|
| 优化 og-image.png | P0 | 30 分钟 | - | ⏳ Pending |
| 延迟 Google Analytics | P0 | 15 分钟 | - | ⏳ Pending |
| 延迟 Vercel 脚本 | P0 | 10 分钟 | - | ⏳ Pending |
| 添加资源预加载 | P1 | 20 分钟 | - | ⏳ Pending |
| 懒加载组件 | P1 | 1 小时 | - | ⏳ Pending |
| 优化 CSS | P2 | 1 小时 | - | ⏳ Pending |
| 静态化 Schema | P2 | 30 分钟 | - | ⏳ Pending |
| 性能测试验证 | P0 | 30 分钟 | - | ⏳ Pending |

**总预计时间**: 约 4 小时

---

## 参考资源

- [Web.dev - First Contentful Paint](https://web.dev/fcp/)
- [Web.dev - Largest Contentful Paint](https://web.dev/lcp/)
- [VitePress Performance Guide](https://vitepress.dev/guide/performance)
- [Google Core Web Vitals](https://web.dev/vitals/)
- [Partytown Documentation](https://partytown.builder.io/)
