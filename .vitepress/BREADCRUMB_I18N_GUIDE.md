# 面包屑导航国际化使用指南
# Breadcrumb Navigation i18n Guide

本文档说明如何测试和使用面包屑导航的国际化功能。

---

## 🌍 已实现的功能

### ✅ 支持语言
- **中文**（zh-CN）- 默认
- **英文**（en-US / en）

### ✅ 国际化内容
1. **面包屑文本** - 所有路径标签根据语言显示
2. **aria-label** - 无障碍标签跟随语言切换
3. **结构化数据** - BreadcrumbList JSON-LD 使用对应语言

---

## 🧪 如何测试

### 方法 1：在 VitePress 配置中启用多语言

如果你的项目计划支持多语言，需要在 `.vitepress/config.ts` 中添加 `locales` 配置：

```typescript
// .vitepress/config.ts
export default defineConfig({
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      title: 'minimind从零理解llm训练',
      description: '深入理解 LLM 训练的每个设计选择',
      themeConfig: {
        nav: [
          { text: '首页', link: '/' },
          { text: '学习指南', link: '/docs/guide/quick-start' },
          // ... 中文导航
        ],
      }
    },
    en: {
      label: 'English',
      lang: 'en-US',
      title: 'MiniMind - Understanding LLM Training from Scratch',
      description: 'Deep dive into every design choice in LLM training',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/' },
          { text: 'Guide', link: '/en/docs/guide/quick-start' },
          // ... 英文导航
        ],
      }
    }
  }
})
```

然后创建对应的英文内容文件：
```
en/
├── index.md
├── docs/
│   └── guide/
│       ├── quick-start.md
│       └── systematic.md
└── modules/
    └── ...
```

### 方法 2：手动模拟语言切换（快速测试）

如果暂时不想配置完整的多语言支持，可以手动测试：

**步骤 1：** 打开浏览器开发者工具（F12）

**步骤 2：** 在 Console 中运行以下代码来模拟语言切换：

```javascript
// 切换到英文
document.documentElement.lang = 'en-US'

// 或切换回中文
document.documentElement.lang = 'zh-CN'

// 然后刷新页面查看效果
location.reload()
```

**步骤 3：** 观察面包屑导航的变化

---

## 📋 翻译对照表

以下是当前支持的路径翻译：

| 路径标识符 | 中文 | 英文 |
|-----------|------|------|
| home | 首页 | Home |
| docs | 学习指南 | Learning Guide |
| guide | 学习指南 | Guide |
| quick-start | 快速体验 | Quick Start |
| systematic | 系统学习 | Systematic Learning |
| mastery | 深度掌握 | Deep Mastery |
| modules | 模块教学 | Modules |
| 01-foundation | 基础组件 | Foundation |
| 02-architecture | 架构组装 | Architecture |
| 01-normalization | Normalization（归一化） | Normalization |
| 02-position-encoding | Position Encoding（位置编码） | Position Encoding |
| 03-attention | Attention（注意力机制） | Attention Mechanism |
| 04-feedforward | FeedForward（前馈网络） | FeedForward Network |
| teaching | 教学文档 | Teaching Doc |
| code_guide | 代码导读 | Code Guide |
| quiz | 自测题 | Quiz |
| learning_log | 学习日志 | Learning Log |
| knowledge_base | 知识库 | Knowledge Base |
| notes | 笔记索引 | Notes Index |
| learning_materials | 学习材料 | Learning Materials |
| ROADMAP | 学习路线图 | Roadmap |

---

## 🔧 如何添加新的翻译

如果需要添加新的页面路径翻译，编辑 `.vitepress/theme/i18n/breadcrumbs.ts`：

```typescript
// 中文映射
export const breadcrumbMappingsZh: BreadcrumbTranslations = {
  // ... 现有翻译
  'new-page': '新页面',  // 添加你的翻译
}

// 英文映射
export const breadcrumbMappingsEn: BreadcrumbTranslations = {
  // ... 现有翻译
  'new-page': 'New Page',  // 添加对应的英文翻译
}
```

---

## 🌐 添加更多语言支持

如果需要支持更多语言（如日语、韩语等），可以：

### 1. 在 `breadcrumbs.ts` 中添加新语言映射

```typescript
export type Locale = 'zh-CN' | 'en-US' | 'en' | 'ja-JP' | 'ko-KR' // 添加新语言

// 添加日语映射
export const breadcrumbMappingsJa: BreadcrumbTranslations = {
  'home': 'ホーム',
  'docs': '学習ガイド',
  'modules': 'モジュール',
  // ... 完整翻译
}

// 添加韩语映射
export const breadcrumbMappingsKo: BreadcrumbTranslations = {
  'home': '홈',
  'docs': '학습 가이드',
  'modules': '모듈',
  // ... 完整翻译
}
```

### 2. 更新 `getBreadcrumbMappings` 函数

```typescript
export function getBreadcrumbMappings(locale: string): BreadcrumbTranslations {
  const normalizedLocale = normalizeLocale(locale)

  switch (normalizedLocale) {
    case 'en-US':
    case 'en':
      return breadcrumbMappingsEn
    case 'ja-JP':
      return breadcrumbMappingsJa
    case 'ko-KR':
      return breadcrumbMappingsKo
    case 'zh-CN':
    default:
      return breadcrumbMappingsZh
  }
}
```

### 3. 更新 `normalizeLocale` 函数

```typescript
export function normalizeLocale(locale: string): Locale {
  if (locale.startsWith('en')) return 'en'
  if (locale.startsWith('ja')) return 'ja-JP'
  if (locale.startsWith('ko')) return 'ko-KR'
  return 'zh-CN'
}
```

### 4. 更新 `getHomeLabel` 函数

```typescript
export function getHomeLabel(locale: string): string {
  const normalizedLocale = normalizeLocale(locale)
  switch (normalizedLocale) {
    case 'en': return 'Home'
    case 'ja-JP': return 'ホーム'
    case 'ko-KR': return '홈'
    default: return '首页'
  }
}
```

---

## 🎯 预期效果

### 中文环境（默认）
```
首页 / 模块教学 / 基础组件 / Normalization（归一化） / 教学文档
```

### 英文环境
```
Home / Modules / Foundation / Normalization / Teaching Doc
```

---

## 🐛 常见问题

### Q1: 面包屑仍然显示中文，但我已经切换到英文了

**A:** 检查以下几点：
1. 确认 VitePress 的 `lang` 配置正确
2. 检查浏览器开发者工具中 `<html lang="...">` 的值
3. 尝试清除缓存并刷新页面

### Q2: 如何知道当前使用的是哪个语言？

**A:** 在浏览器控制台运行：
```javascript
console.log(document.documentElement.lang)
```

### Q3: 某些路径没有翻译，显示原始路径标识符

**A:** 这是正常的。如果某个路径在 `breadcrumbs.ts` 中没有映射，会显示原始路径名。你可以按照上面的步骤添加新的翻译。

---

## 📚 相关文件

- **配置文件**: `.vitepress/theme/i18n/breadcrumbs.ts`
- **组件**: `.vitepress/theme/components/Breadcrumbs.vue`
- **Composable**: `.vitepress/theme/composables/useBreadcrumbSchema.ts`
- **Layout**: `.vitepress/theme/Layout.vue`

---

## 🚀 下一步

面包屑导航的国际化已经完成！接下来可以：

1. **添加完整的多语言内容** - 为每种语言创建对应的 markdown 文件
2. **配置语言切换器** - 在导航栏添加语言切换按钮
3. **优化 SEO** - 为每种语言版本添加 `hreflang` 标签

---

有任何问题，请查阅 VitePress 官方文档：
https://vitepress.dev/guide/i18n
