<template>
  <nav v-if="breadcrumbs.length > 1" class="breadcrumbs" aria-label="面包屑导航">
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

interface Breadcrumb {
  name: string
  url: string
}

const { page } = useData()

const breadcrumbs = computed<Breadcrumb[]>(() => {
  const path = page.value.relativePath
  const parts = path.replace(/\.md$/, '').split('/')
  const crumbs: Breadcrumb[] = [{ name: '首页', url: '/' }]

  let currentPath = ''

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i]

    // 跳过 index 和空字符串
    if (!part || part === 'index') continue

    currentPath += (currentPath ? '/' : '') + part

    // 生成面包屑名称
    let name = part

    // 特殊路径映射
    const pathMappings: Record<string, string> = {
      'docs': '学习指南',
      'guide': '学习指南',
      'quick-start': '快速体验',
      'systematic': '系统学习',
      'mastery': '深度掌握',
      'modules': '模块教学',
      '01-foundation': '基础组件',
      '02-architecture': '架构组装',
      '01-normalization': 'Normalization（归一化）',
      '02-position-encoding': 'Position Encoding（位置编码）',
      '03-attention': 'Attention（注意力机制）',
      '04-feedforward': 'FeedForward（前馈网络）',
      'teaching': '教学文档',
      'code_guide': '代码导读',
      'quiz': '自测题',
      'learning_log': '学习日志',
      'knowledge_base': '知识库',
      'notes': '笔记索引',
      'learning_materials': '学习材料',
      'ROADMAP': '学习路线图',
    }

    name = pathMappings[part] || part

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
