/**
 * Utility to generate BreadcrumbList structured data for SEO
 * Used in VitePress transformHead hook (server-side)
 */

import type { PageData } from 'vitepress'
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

const SITE_URL = 'https://minimind.wiki'

/**
 * Generate BreadcrumbList schema.org structured data for a page
 * Returns null if breadcrumbs would only contain home (single item)
 */
export function generateBreadcrumbSchema(
  pageData: PageData
): BreadcrumbListSchema | null {
  const path = pageData.relativePath
  const lang = pageData.lang || 'zh-CN'
  const parts = path.replace(/\.md$/, '').split('/')
  const items: BreadcrumbItem[] = []

  // Get i18n mappings for current language
  const pathMappings = getBreadcrumbMappings(lang)
  const homeLabel = getHomeLabel(lang)

  // Add home as first item
  items.push({
    '@type': 'ListItem',
    position: 1,
    name: homeLabel,
    item: `${SITE_URL}/`,
  })

  let currentPath = ''
  let position = 2

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i]

    // Skip index and empty strings
    if (!part || part === 'index') continue

    currentPath += (currentPath ? '/' : '') + part

    // Use i18n mapping for breadcrumb name
    const name = pathMappings[part] || part

    // Skip 'guide' as standalone breadcrumb
    if (part === 'guide') continue

    // Build full URL
    let url = `${SITE_URL}/${currentPath}`

    // For last part, use complete path
    if (i === parts.length - 1) {
      url = `${SITE_URL}/${path.replace(/\.md$/, '')}`
    } else if (part === 'docs' && parts[i + 1] !== 'guide') {
      url = `${SITE_URL}/docs/`
    }

    items.push({
      '@type': 'ListItem',
      position: position++,
      name,
      item: url,
    })
  }

  // Only return schema if more than just home
  if (items.length <= 1) {
    return null
  }

  return {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: items,
  }
}
