<template>
  <nav v-if="breadcrumbs.length > 1" class="breadcrumbs" :aria-label="ariaLabel">
    <ol itemscope itemtype="https://schema.org/BreadcrumbList">
      <li
        v-for="(crumb, index) in breadcrumbs"
        :key="crumb.url"
        itemprop="itemListElement"
        itemscope
        itemtype="https://schema.org/ListItem"
      >
        <a
          v-if="index < breadcrumbs.length - 1"
          :href="crumb.url"
          itemprop="item"
        >
          <span itemprop="name">{{ crumb.name }}</span>
        </a>
        <span v-else itemprop="name" class="current">{{ crumb.name }}</span>
        <meta itemprop="position" :content="String(index + 1)" />
        <span v-if="index < breadcrumbs.length - 1" class="separator">/</span>
      </li>
    </ol>
  </nav>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'
import { getBreadcrumbMappings, getHomeLabel, normalizeLocale } from '../i18n/breadcrumbs'

interface Breadcrumb {
  name: string
  url: string
}

const { page, lang } = useData()

// 获取当前语言的面包屑映射
const pathMappings = computed(() => getBreadcrumbMappings(lang.value))

// 获取 aria-label 的翻译
const ariaLabel = computed(() => {
  const locale = normalizeLocale(lang.value)
  return locale === 'en' ? 'Breadcrumb navigation' : '面包屑导航'
})

const breadcrumbs = computed<Breadcrumb[]>(() => {
  const path = page.value.relativePath
  const parts = path.replace(/\.md$/, '').split('/')
  const homeLabel = getHomeLabel(lang.value)
  const crumbs: Breadcrumb[] = [{ name: homeLabel, url: '/' }]

  let currentPath = ''

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i]

    // 跳过 index 和空字符串
    if (!part || part === 'index') continue

    currentPath += (currentPath ? '/' : '') + part

    // 使用 i18n 映射生成面包屑名称
    const name = pathMappings.value[part] || part

    // 构建URL
    let url = '/' + currentPath

    // 对于模块的子页面，需要确保URL正确
    if (i === parts.length - 1) {
      // 最后一个部分，使用完整URL
      url = '/' + path.replace(/\.md$/, '')
    } else if (part === 'docs' && parts[i + 1] !== 'guide') {
      // docs/index.md 的情况
      url = '/docs/'
    } else if (part === 'guide') {
      // guide 不单独作为面包屑
      continue
    }

    crumbs.push({ name, url })
  }

  return crumbs
})
</script>

<style scoped>
.breadcrumbs {
  padding: 1rem 0;
  margin-bottom: 1rem;
  border-bottom: 1px solid var(--vp-c-divider);
  font-size: 0.9rem;
}

.breadcrumbs ol {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
}

.breadcrumbs li {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.breadcrumbs a {
  color: var(--vp-c-brand-1);
  text-decoration: none;
  transition: color 0.2s;
}

.breadcrumbs a:hover {
  color: var(--vp-c-brand-2);
  text-decoration: underline;
}

.breadcrumbs .current {
  color: var(--vp-c-text-1);
  font-weight: 500;
}

.breadcrumbs .separator {
  color: var(--vp-c-text-3);
  user-select: none;
}

/* 暗色模式优化 */
:global(.dark) .breadcrumbs {
  border-bottom-color: var(--vp-c-divider);
}

:global(.dark) .breadcrumbs a {
  color: var(--vp-c-brand-1);
}

:global(.dark) .breadcrumbs a:hover {
  color: var(--vp-c-brand-2);
}

/* 移动端优化 */
@media (max-width: 768px) {
  .breadcrumbs {
    padding: 0.75rem 0;
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
  }

  .breadcrumbs ol {
    gap: 0.4rem;
  }

  .breadcrumbs li {
    gap: 0.4rem;
  }
}
</style>
