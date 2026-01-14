/**
 * 面包屑导航国际化配置
 * Breadcrumb navigation i18n configuration
 */

export type Locale = 'zh-CN' | 'en-US' | 'en'

export interface BreadcrumbTranslations {
  [key: string]: string
}

/**
 * 面包屑路径映射 - 中文
 */
export const breadcrumbMappingsZh: BreadcrumbTranslations = {
  // 核心页面
  'home': '首页',

  // 学习指南
  'docs': '学习指南',
  'guide': '学习指南',
  'quick-start': '快速体验',
  'systematic': '系统学习',
  'mastery': '深度掌握',

  // 模块教学
  'modules': '模块教学',
  '01-foundation': '基础组件',
  '02-architecture': '架构组装',

  // 基础组件模块
  '01-normalization': 'Normalization（归一化）',
  '02-position-encoding': 'Position Encoding（位置编码）',
  '03-attention': 'Attention（注意力机制）',
  '04-feedforward': 'FeedForward（前馈网络）',

  // 模块子页面
  'teaching': '教学文档',
  'code_guide': '代码导读',
  'quiz': '自测题',
  'index': '概览',

  // 学习笔记
  'learning_log': '学习日志',
  'knowledge_base': '知识库',
  'notes': '笔记索引',
  'learning_materials': '学习材料',

  // 其他页面
  'ROADMAP': '学习路线图',
  'README': '项目介绍',
}

/**
 * 面包屑路径映射 - 英文
 */
export const breadcrumbMappingsEn: BreadcrumbTranslations = {
  // Core pages
  'home': 'Home',

  // Learning guides
  'docs': 'Learning Guide',
  'guide': 'Guide',
  'quick-start': 'Quick Start',
  'systematic': 'Systematic Learning',
  'mastery': 'Deep Mastery',

  // Modules
  'modules': 'Modules',
  '01-foundation': 'Foundation',
  '02-architecture': 'Architecture',

  // Foundation modules
  '01-normalization': 'Normalization',
  '02-position-encoding': 'Position Encoding',
  '03-attention': 'Attention Mechanism',
  '04-feedforward': 'FeedForward Network',

  // Module sub-pages
  'teaching': 'Teaching Doc',
  'code_guide': 'Code Guide',
  'quiz': 'Quiz',
  'index': 'Overview',

  // Learning notes
  'learning_log': 'Learning Log',
  'knowledge_base': 'Knowledge Base',
  'notes': 'Notes Index',
  'learning_materials': 'Learning Materials',

  // Other pages
  'ROADMAP': 'Roadmap',
  'README': 'About',
}

/**
 * 获取指定语言的面包屑映射
 */
export function getBreadcrumbMappings(locale: string): BreadcrumbTranslations {
  // 标准化 locale
  const normalizedLocale = normalizeLocale(locale)

  switch (normalizedLocale) {
    case 'en-US':
    case 'en':
      return breadcrumbMappingsEn
    case 'zh-CN':
    default:
      return breadcrumbMappingsZh
  }
}

/**
 * 标准化 locale 字符串
 */
export function normalizeLocale(locale: string): Locale {
  if (locale.startsWith('en')) {
    return 'en'
  }
  return 'zh-CN'
}

/**
 * 获取"首页"的翻译
 */
export function getHomeLabel(locale: string): string {
  const normalizedLocale = normalizeLocale(locale)
  return normalizedLocale === 'en' ? 'Home' : '首页'
}
