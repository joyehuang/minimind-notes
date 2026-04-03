import { defineConfig } from 'vitepress'
import { generateBreadcrumbSchema } from './theme/utils/breadcrumbSchema'

const siteUrl = 'https://minimind.wiki'

const sharedHead = [
  // Resource hints for third-party domains (improve performance)
  ['link', { rel: 'preconnect', href: 'https://www.googletagmanager.com' }],
  ['link', { rel: 'dns-prefetch', href: 'https://vercel-analytics.com' }],
  ['link', { rel: 'dns-prefetch', href: 'https://vitals.vercel-insights.com' }],

  // Favicons
  ['link', { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }],
  ['link', { rel: 'icon', type: 'image/png', sizes: '16x16', href: '/favicon-16x16.png' }],
  ['link', { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/favicon-32x32.png' }],
  ['link', { rel: 'apple-touch-icon', sizes: '180x180', href: '/apple-touch-icon.png' }],
  ['link', { rel: 'manifest', href: '/site.webmanifest' }],

  // Meta tags
  ['meta', { name: 'theme-color', content: '#3b82f6' }],
  ['meta', { name: 'viewport', content: 'width=device-width, initial-scale=1.0, maximum-scale=5.0, minimum-scale=1.0' }],
  ['meta', { name: 'format-detection', content: 'telephone=no' }],

  // Analytics
  ['script', { async: true, src: 'https://www.googletagmanager.com/gtag/js?id=G-7B7HTLDJ65' }],
  ['script', {}, `
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-7B7HTLDJ65');
  `]
]

const zhHead = [
  ['meta', { name: 'keywords', content: 'MiniMind, LLM, Transformer, 训练原理, 归一化, 位置编码, 注意力机制, PyTorch, 教学, 实验' }],
  ['meta', { name: 'author', content: 'joyehuang' }],
  ['meta', { name: 'robots', content: 'index, follow' }],
  ['meta', { name: 'googlebot', content: 'index, follow' }],
  ['meta', { property: 'og:type', content: 'website' }],
  ['meta', { property: 'og:site_name', content: 'minimind从零理解llm训练' }],
  ['meta', { property: 'og:title', content: 'minimind从零理解llm训练 - 从原理出发的 LLM 实验课' }],
  ['meta', { property: 'og:description', content: '通过可执行实验深入理解 LLM 训练的关键设计选择，包含教学文档、实验代码与学习笔记。' }],
  ['meta', { property: 'og:image', content: `${siteUrl}/og-image.png` }],
  ['meta', { property: 'og:locale', content: 'zh_CN' }],
  ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
  ['meta', { name: 'twitter:title', content: 'minimind从零理解llm训练 - 从原理出发的 LLM 实验课' }],
  ['meta', { name: 'twitter:description', content: '通过可执行实验深入理解 LLM 训练的关键设计选择。' }],
  ['meta', { name: 'twitter:image', content: `${siteUrl}/og-image.png` }],
  ['script', { type: 'application/ld+json' }, JSON.stringify({
    '@context': 'https://schema.org',
    '@type': 'WebSite',
    name: 'minimind从零理解llm训练',
    description: '通过可执行实验深入理解 LLM 训练的关键设计选择。',
    url: siteUrl,
    author: {
      '@type': 'Person',
      name: 'joyehuang',
      url: 'https://github.com/joyehuang'
    },
    publisher: {
      '@type': 'Organization',
      name: 'MiniMind',
      logo: {
        '@type': 'ImageObject',
        url: `${siteUrl}/logo.svg`
      }
    },
    inLanguage: 'zh-CN',
    potentialAction: {
      '@type': 'SearchAction',
      target: `${siteUrl}/?q={search_term_string}`,
      'query-input': 'required name=search_term_string'
    }
  })],
  ['script', { type: 'application/ld+json' }, JSON.stringify({
    '@context': 'https://schema.org',
    '@type': 'Course',
    name: 'MiniMind LLM 训练入门与进阶',
    description: '从原理出发理解 Transformer、归一化、位置编码、注意力与完整训练流程。',
    provider: {
      '@type': 'Organization',
      name: 'MiniMind',
      sameAs: 'https://github.com/jingyaogong/minimind'
    },
    educationalLevel: 'Intermediate',
    inLanguage: 'zh-CN',
    isAccessibleForFree: true,
    url: siteUrl,
    hasCourseInstance: {
      '@type': 'CourseInstance',
      courseMode: 'online',
      courseWorkload: 'PT30H'
    }
  })]
]

const enHead = [
  ['meta', { name: 'keywords', content: 'MiniMind, LLM, Transformer, training principles, normalization, position encoding, attention, PyTorch, tutorial, experiments' }],
  ['meta', { name: 'author', content: 'joyehuang' }],
  ['meta', { name: 'robots', content: 'index, follow' }],
  ['meta', { name: 'googlebot', content: 'index, follow' }],
  ['meta', { property: 'og:type', content: 'website' }],
  ['meta', { property: 'og:site_name', content: 'MiniMind LLM Training' }],
  ['meta', { property: 'og:title', content: 'MiniMind LLM Training - Principle-first experimental lab' }],
  ['meta', { property: 'og:description', content: 'Understand LLM training with executable experiments, teaching docs, and practical notes.' }],
  ['meta', { property: 'og:image', content: `${siteUrl}/og-image.png` }],
  ['meta', { property: 'og:locale', content: 'en_US' }],
  ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
  ['meta', { name: 'twitter:title', content: 'MiniMind LLM Training - Principle-first experimental lab' }],
  ['meta', { name: 'twitter:description', content: 'Understand LLM training with executable experiments, teaching docs, and practical notes.' }],
  ['meta', { name: 'twitter:image', content: `${siteUrl}/og-image.png` }],
  ['script', { type: 'application/ld+json' }, JSON.stringify({
    '@context': 'https://schema.org',
    '@type': 'WebSite',
    name: 'MiniMind LLM Training',
    description: 'Understand LLM training with executable experiments and principle-first lessons.',
    url: `${siteUrl}/en/`,
    author: {
      '@type': 'Person',
      name: 'joyehuang',
      url: 'https://github.com/joyehuang'
    },
    publisher: {
      '@type': 'Organization',
      name: 'MiniMind',
      logo: {
        '@type': 'ImageObject',
        url: `${siteUrl}/logo.svg`
      }
    },
    inLanguage: 'en',
    potentialAction: {
      '@type': 'SearchAction',
      target: `${siteUrl}/en/?q={search_term_string}`,
      'query-input': 'required name=search_term_string'
    }
  })],
  ['script', { type: 'application/ld+json' }, JSON.stringify({
    '@context': 'https://schema.org',
    '@type': 'Course',
    name: 'MiniMind LLM Training Principles',
    description: 'Principle-first understanding of Transformer components and full training workflows.',
    provider: {
      '@type': 'Organization',
      name: 'MiniMind',
      sameAs: 'https://github.com/jingyaogong/minimind'
    },
    educationalLevel: 'Intermediate',
    inLanguage: 'en',
    isAccessibleForFree: true,
    url: `${siteUrl}/en/`,
    hasCourseInstance: {
      '@type': 'CourseInstance',
      courseMode: 'online',
      courseWorkload: 'PT30H'
    }
  })]
]

const toRoute = (relativePath: string) => {
  const normalized = `/${relativePath.replace(/\\/g, '/')}`
  if (normalized.endsWith('/index.md')) {
    return normalized.slice(0, -'index.md'.length)
  }
  return normalized.replace(/\.md$/, '')
}

export default defineConfig({
  lang: 'zh-CN',
  title: 'minimind从零理解llm训练',
  description: '深入理解 LLM 训练的每个设计选择',
  srcDir: '.',
  outDir: '.vitepress/dist',
  cleanUrls: true,
  ignoreDeadLinks: true,
  head: sharedHead,
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      title: 'minimind从零理解llm训练',
      description: '深入理解 LLM 训练的每个设计选择',
      head: zhHead,
      themeConfig: {
        nav: [
          { text: '文档', link: '/docs/' },
          {
            text: '学习路线',
            items: [
              { text: '快速体验 (30分钟)', link: '/docs/guide/quick-start' },
              { text: '系统学习 (6小时)', link: '/docs/guide/systematic' },
              { text: '深度掌握 (30+小时)', link: '/docs/guide/mastery' },
              { text: 'Roadmap', link: '/ROADMAP' }
            ]
          },
          {
            text: '模块教学',
            items: [
              { text: '模块总览', link: '/modules/' },
              { text: '基础组件', link: '/modules/01-foundation/' },
              { text: '架构组装', link: '/modules/02-architecture/' }
            ]
          },
          {
            text: '学习笔记',
            items: [
              { text: '学习日志', link: '/learning_log' },
              { text: '知识库', link: '/knowledge_base' },
              { text: '总索引', link: '/notes' }
            ]
          }
        ],
        sidebar: {
          '/docs/guide/': [
            {
              text: '学习路线',
              items: [
                { text: '快速体验', link: '/docs/guide/quick-start' },
                { text: '系统学习', link: '/docs/guide/systematic' },
                { text: '深度掌握', link: '/docs/guide/mastery' },
                { text: 'Roadmap', link: '/ROADMAP' }
              ]
            }
          ],
          '/modules/': [
            {
              text: '模块总览',
              items: [
                { text: '模块导航', link: '/modules/' }
              ]
            },
            {
              text: '基础组件 (Foundation)',
              collapsed: false,
              items: [
                { text: '01 归一化 (Normalization)', link: '/modules/01-foundation/01-normalization/' },
                { text: '02 位置编码 (Position Encoding)', link: '/modules/01-foundation/02-position-encoding/' },
                { text: '03 注意力机制 (Attention)', link: '/modules/01-foundation/03-attention/' },
                { text: '04 前馈网络 (FeedForward)', link: '/modules/01-foundation/04-feedforward/' }
              ]
            },
            {
              text: '架构组装 (Architecture)',
              items: [
                { text: '架构总览', link: '/modules/02-architecture/' }
              ]
            }
          ],
          '/': [
            {
              text: '开始',
              items: [
                { text: '文档', link: '/docs/' },
                { text: '快速体验', link: '/docs/guide/quick-start' }
              ]
            },
            {
              text: '学习笔记',
              items: [
                { text: '学习日志', link: '/learning_log' },
                { text: '知识库', link: '/knowledge_base' },
                { text: '总索引', link: '/notes' }
              ]
            },
            {
              text: '模块教学',
              items: [
                { text: '模块总览', link: '/modules/' },
                { text: '基础组件', link: '/modules/01-foundation/' },
                { text: '架构组装', link: '/modules/02-architecture/' }
              ]
            },
            {
              text: '资源',
              items: [
                { text: 'Roadmap', link: '/ROADMAP' },
                { text: '学习材料', link: '/learning_materials/README' }
              ]
            }
          ]
        },
        search: {
          provider: 'local',
          options: {
            locales: {
              root: {
                translations: {
                  button: {
                    buttonText: '搜索',
                    buttonAriaLabel: '搜索'
                  },
                  modal: {
                    noResultsText: '没有找到结果',
                    resetButtonTitle: '清除查询',
                    footer: {
                      selectText: '选择',
                      navigateText: '切换',
                      closeText: '关闭'
                    }
                  }
                }
              }
            }
          }
        },
        footer: {
          message: '基于 <a href="https://github.com/jingyaogong/minimind" target="_blank">MiniMind</a> 的学习笔记与实验',
          copyright: 'Copyright © 2025 joyehuang'
        },
        editLink: {
          pattern: 'https://github.com/joyehuang/minimind-notes/edit/main/:path',
          text: '在 GitHub 上编辑此页'
        },
        lastUpdated: {
          text: '更新时间',
          formatOptions: {
            dateStyle: 'short',
            timeStyle: 'short'
          }
        },
        docFooter: {
          prev: '上一页',
          next: '下一页'
        },
        outline: {
          level: [2, 3],
          label: '本页目录'
        },
        returnToTopLabel: '返回顶部',
        sidebarMenuLabel: '菜单',
        darkModeSwitchLabel: '主题',
        lightModeSwitchTitle: '切换到浅色模式',
        darkModeSwitchTitle: '切换到深色模式'
      }
    },
    en: {
      label: 'English',
      lang: 'en-US',
      title: 'MiniMind LLM Training',
      description: 'Principle-first experimental lab for understanding LLM training',
      head: enHead,
      themeConfig: {
        nav: [
          { text: 'Docs', link: '/en/docs/' },
          {
            text: 'Learning Paths',
            items: [
              { text: 'Quick Start (30 min)', link: '/en/docs/guide/quick-start' },
              { text: 'Systematic Study (6 hours)', link: '/en/docs/guide/systematic' },
              { text: 'Deep Mastery (30+ hours)', link: '/en/docs/guide/mastery' },
              { text: 'Roadmap', link: '/en/ROADMAP' }
            ]
          },
          {
            text: 'Modules',
            items: [
              { text: 'Modules Overview', link: '/en/modules/' },
              { text: 'Foundation', link: '/en/modules/01-foundation/' },
              { text: 'Architecture', link: '/en/modules/02-architecture/' }
            ]
          }
        ],
        sidebar: {
          '/en/docs/guide/': [
            {
              text: 'Learning Paths',
              items: [
                { text: 'Quick Start', link: '/en/docs/guide/quick-start' },
                { text: 'Systematic Study', link: '/en/docs/guide/systematic' },
                { text: 'Deep Mastery', link: '/en/docs/guide/mastery' },
                { text: 'Roadmap', link: '/en/ROADMAP' }
              ]
            }
          ],
          '/en/modules/': [
            {
              text: 'Modules Overview',
              items: [
                { text: 'Modules Navigator', link: '/en/modules/' }
              ]
            },
            {
              text: 'Foundation',
              collapsed: false,
              items: [
                { text: '01 Normalization', link: '/en/modules/01-foundation/01-normalization/' },
                { text: '02 Position Encoding', link: '/en/modules/01-foundation/02-position-encoding/' },
                { text: '03 Attention', link: '/en/modules/01-foundation/03-attention/' },
                { text: '04 FeedForward', link: '/en/modules/01-foundation/04-feedforward/' }
              ]
            },
            {
              text: 'Architecture',
              items: [
                { text: 'Architecture Overview', link: '/en/modules/02-architecture/' }
              ]
            }
          ],
          '/en/': [
            {
              text: 'Start Here',
              items: [
                { text: 'Docs', link: '/en/docs/' },
                { text: 'Quick Start', link: '/en/docs/guide/quick-start' }
              ]
            },
            {
              text: 'Learning Paths',
              items: [
                { text: 'Quick Start', link: '/en/docs/guide/quick-start' },
                { text: 'Systematic Study', link: '/en/docs/guide/systematic' },
                { text: 'Deep Mastery', link: '/en/docs/guide/mastery' }
              ]
            },
            {
              text: 'Modules',
              items: [
                { text: 'Modules Overview', link: '/en/modules/' },
                { text: 'Foundation', link: '/en/modules/01-foundation/' },
                { text: 'Architecture', link: '/en/modules/02-architecture/' }
              ]
            },
            {
              text: 'Resources',
              items: [
                { text: 'Roadmap', link: '/en/ROADMAP' }
              ]
            }
          ]
        },
        search: {
          provider: 'local'
        },
        footer: {
          message: 'Built on <a href="https://github.com/jingyaogong/minimind" target="_blank">MiniMind</a> for learning and experiments',
          copyright: 'Copyright © 2025 joyehuang'
        },
        editLink: {
          pattern: 'https://github.com/joyehuang/minimind-notes/edit/main/:path',
          text: 'Edit this page on GitHub'
        },
        lastUpdated: {
          text: 'Last updated',
          formatOptions: {
            dateStyle: 'short',
            timeStyle: 'short'
          }
        },
        docFooter: {
          prev: 'Previous',
          next: 'Next'
        },
        outline: {
          level: [2, 3],
          label: 'On this page'
        },
        returnToTopLabel: 'Back to top',
        sidebarMenuLabel: 'Menu',
        darkModeSwitchLabel: 'Theme',
        lightModeSwitchTitle: 'Switch to light theme',
        darkModeSwitchTitle: 'Switch to dark theme'
      }
    }
  },
  themeConfig: {
    logo: '/logo.svg',
    socialLinks: [
      { icon: 'github', link: 'https://github.com/joyehuang/minimind-notes' }
    ]
  },
  transformHead: ({ pageData }) => {
    const route = toRoute(pageData.relativePath)
    const isEn = route.startsWith('/en/')
    const zhRoute = isEn ? route.replace(/^\/en/, '') : route
    const enRoute = isEn ? route : `/en${route === '/' ? '/' : route}`

    const headTags: any[] = [
      ['link', { rel: 'canonical', href: `${siteUrl}${route}` }],
      ['link', { rel: 'alternate', hreflang: 'zh-CN', href: `${siteUrl}${zhRoute}` }],
      ['link', { rel: 'alternate', hreflang: 'en-US', href: `${siteUrl}${enRoute}` }],
      ['link', { rel: 'alternate', hreflang: 'x-default', href: `${siteUrl}${zhRoute}` }],
      ['meta', { property: 'og:url', content: `${siteUrl}${route}` }]
    ]

    // Add BreadcrumbList structured data (server-side, no DOM manipulation)
    const breadcrumbSchema = generateBreadcrumbSchema(pageData)
    if (breadcrumbSchema) {
      headTags.push([
        'script',
        { type: 'application/ld+json', 'data-schema': 'breadcrumb' },
        JSON.stringify(breadcrumbSchema)
      ])
    }

    return headTags
  },
  markdown: {
    math: true,
    lineNumbers: true,
    image: {
      lazyLoading: true
    },
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },
  sitemap: {
    hostname: siteUrl,
    transformItems: (items) => {
      const excludePatterns = [
        '/CLAUDE',
        '/CODE_OF_CONDUCT',
        '/NOTE_UPDATE_GUIDE',
        '/PR_DESCRIPTION',
        '/README',
        '/README_en',
        '/SEO_SETUP_SUMMARY',
        '/SEO_OPTIMIZATION_PLAN',
        '/VITEPRESS_DEV_PLAN',
        '/VITEPRESS_MIGRATION_PLAN',
        '/VITEPRESS_RECOMMENDED_STRUCTURE',
        '/VITEPRESS_SETUP_GUIDE',
        '/BREADCRUMB_I18N_GUIDE',
        '/docs-index-example',
        '/dataset/dataset'
      ]

      return items
        .filter((item) => {
          // 检查 URL 是否包含排除的路径
          return !excludePatterns.some(pattern => item.url.includes(pattern))
        })
        .map((item) => {
          // 根据 URL 设置优先级和更新频率
          let priority = 0.5
          let changefreq = 'monthly'

          // 首页 - 最高优先级
          if (item.url === 'https://minimind.wiki/') {
            priority = 1.0
            changefreq = 'weekly'
          }
          // 学习路线图 - 高优先级
          else if (item.url.includes('/ROADMAP')) {
            priority = 0.9
            changefreq = 'weekly'
          }
          // 学习指南页面 - 高优先级
          else if (
            item.url.includes('/docs/') ||
            item.url.includes('/docs/guide/')
          ) {
            priority = 0.8
            changefreq = 'weekly'
          }
          // 模块教学页面 - 高优先级
          else if (item.url.includes('/modules/')) {
            // 模块首页
            if (
              item.url.endsWith('/modules/') ||
              item.url.includes('/modules/index')
            ) {
              priority = 0.9
              changefreq = 'weekly'
            }
            // 模块分类页面
            else if (
              item.url.includes('/01-foundation/') ||
              item.url.includes('/02-architecture/')
            ) {
              // 分类首页
              if (
                item.url.match(/\/(01-foundation|02-architecture)\/?$/) ||
                item.url.match(/\/(01-foundation|02-architecture)\/index$/)
              ) {
                priority = 0.85
                changefreq = 'weekly'
              }
              // 具体模块内容（teaching, code_guide, quiz）
              else {
                priority = 0.8
                changefreq = 'weekly'
              }
            }
          }
          // 学习笔记页面 - 高优先级且频繁更新
          else if (
            item.url.includes('/learning_log') ||
            item.url.includes('/knowledge_base') ||
            item.url.includes('/notes')
          ) {
            priority = 0.7
            changefreq = 'daily'
          }
          // 学习材料
          else if (item.url.includes('/learning_materials')) {
            priority = 0.6
            changefreq = 'weekly'
          }

          return {
            ...item,
            priority,
            changefreq,
            // 添加最后修改时间（使用当前时间作为默认值，实际应该从 git 获取）
            lastmod: item.lastmod || new Date().toISOString(),
          }
        })
    }
  }
})
