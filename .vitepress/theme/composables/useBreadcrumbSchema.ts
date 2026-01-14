import { computed } from 'vue'
import { useData } from 'vitepress'
import { getBreadcrumbMappings, getHomeLabel } from '../i18n/breadcrumbs'

interface BreadcrumbItem {
  '@type': 'ListItem'
  position: number
  name: string
  item: string
}

interface BreadcrumbListSchema {
  '@context': 'https://schema.org'
  '@type': 'BreadcrumbList'
  itemListElement: BreadcrumbItem[]
}

export function useBreadcrumbSchema() {
  const { page, lang } = useData()

  const breadcrumbSchema = computed<BreadcrumbListSchema | null>(() => {
    const path = page.value.relativePath
    const parts = path.replace(/\.md$/, '').split('/')
    const items: BreadcrumbItem[] = []

    // 获取当前语言的映射和首页标签
    const pathMappings = getBreadcrumbMappings(lang.value)
    const homeLabel = getHomeLabel(lang.value)

    // 添加首页
    items.push({
      '@type': 'ListItem',
      position: 1,
      name: homeLabel,
      item: 'https://minimind.wiki/',
    })

    let currentPath = ''
    let position = 2

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i]

      // 跳过 index 和空字符串
      if (!part || part === 'index') continue

      currentPath += (currentPath ? '/' : '') + part

      // 使用 i18n 映射生成面包屑名称
      const name = pathMappings[part] || part

      // 跳过 guide 作为单独的面包屑
      if (part === 'guide') continue

      // 构建完整URL
      let url = 'https://minimind.wiki/' + currentPath

      // 对于最后一个部分，使用完整路径
      if (i === parts.length - 1) {
        url = 'https://minimind.wiki/' + path.replace(/\.md$/, '')
      } else if (part === 'docs' && parts[i + 1] !== 'guide') {
        url = 'https://minimind.wiki/docs/'
      }

      items.push({
        '@type': 'ListItem',
        position: position++,
        name,
        item: url,
      })
    }

    // 只有超过1个面包屑时才返回schema
    if (items.length <= 1) {
      return null
    }

    return {
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: items,
    }
  })

  return { breadcrumbSchema }
}
