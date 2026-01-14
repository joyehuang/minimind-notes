import { computed } from 'vue'
import { useData } from 'vitepress'

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
  const { page } = useData()

  const breadcrumbSchema = computed<BreadcrumbListSchema | null>(() => {
    const path = page.value.relativePath
    const parts = path.replace(/\.md$/, '').split('/')
    const items: BreadcrumbItem[] = []

    // 添加首页
    items.push({
      '@type': 'ListItem',
      position: 1,
      name: '首页',
      item: 'https://minimind.wiki/',
    })

    let currentPath = ''
    let position = 2

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
