import { defineConfig } from 'vitepress'

export default defineConfig({
  // 站点信息
  title: 'minimind从零理解llm训练',
  description: '深入理解 LLM 训练的每个设计选择',
  lang: 'zh-CN',

  // 源目录配置 - 关键: 指向根目录以读取现有文件
  srcDir: '.',
  outDir: '.vitepress/dist',

  // 清理 URL
  cleanUrls: true,

  // 忽略死链接检查 (很多链接指向原位置的文件)
  ignoreDeadLinks: true,

  // Head 配置
  head: [
    // Favicon 配置
    ['link', { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }],
    ['link', { rel: 'icon', type: 'image/png', sizes: '16x16', href: '/favicon-16x16.png' }],
    ['link', { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/favicon-32x32.png' }],
    ['link', { rel: 'apple-touch-icon', sizes: '180x180', href: '/apple-touch-icon.png' }],
    ['link', { rel: 'manifest', href: '/site.webmanifest' }],
    ['meta', { name: 'theme-color', content: '#3b82f6' }],

    // SEO Meta Tags
    ['meta', { name: 'keywords', content: 'MiniMind, LLM, Transformer, 大语言模型, 深度学习, 机器学习, 人工智能, PyTorch, 教程, 学习笔记' }],
    ['meta', { name: 'author', content: 'joyehuang' }],
    ['meta', { name: 'robots', content: 'index, follow' }],
    ['meta', { name: 'googlebot', content: 'index, follow' }],

    // Open Graph Meta Tags (for social media sharing)
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'minimind从零理解llm训练' }],
    ['meta', { property: 'og:title', content: 'minimind从零理解llm训练 - 深入理解 LLM 训练的每个设计选择' }],
    ['meta', { property: 'og:description', content: '通过对照实验彻底理解大语言模型训练的每个设计选择，包含模块化教学、代码示例和实践指南' }],
    ['meta', { property: 'og:url', content: 'https://minimind.wiki' }],
    ['meta', { property: 'og:image', content: 'https://minimind.wiki/og-image.png' }],
    ['meta', { property: 'og:locale', content: 'zh_CN' }],

    // Twitter Card Meta Tags
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:title', content: 'minimind从零理解llm训练 - 深入理解 LLM 训练' }],
    ['meta', { name: 'twitter:description', content: '通过对照实验彻底理解大语言模型训练的每个设计选择' }],
    ['meta', { name: 'twitter:image', content: 'https://minimind.wiki/og-image.png' }],

    // Mobile Meta Tags
    ['meta', { name: 'viewport', content: 'width=device-width, initial-scale=1.0, maximum-scale=5.0, minimum-scale=1.0' }],
    ['meta', { name: 'format-detection', content: 'telephone=no' }],

    // Canonical URL
    ['link', { rel: 'canonical', href: 'https://minimind.wiki' }],

    // Google Analytics 4
    ['script', { async: true, src: 'https://www.googletagmanager.com/gtag/js?id=G-7B7HTLDJ65' }],
    ['script', {}, `
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-7B7HTLDJ65');
    `],

    // Structured Data (JSON-LD) for better SEO
    ['script', { type: 'application/ld+json' }, JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'WebSite',
      name: 'minimind从零理解llm训练',
      description: '深入理解 LLM 训练的每个设计选择 - 通过对照实验彻底理解大语言模型训练',
      url: 'https://minimind.wiki',
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
          url: 'https://minimind.wiki/logo.svg'
        }
      },
      inLanguage: 'zh-CN',
      potentialAction: {
        '@type': 'SearchAction',
        target: 'https://minimind.wiki/?q={search_term_string}',
        'query-input': 'required name=search_term_string'
      }
    })],

    // Structured Data for Educational Content
    ['script', { type: 'application/ld+json' }, JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'Course',
      name: 'MiniMind LLM 训练教程',
      description: '从零开始学习大语言模型训练，包含 Transformer、注意力机制、位置编码等核心概念',
      provider: {
        '@type': 'Organization',
        name: 'MiniMind',
        sameAs: 'https://github.com/jingyaogong/minimind'
      },
      educationalLevel: 'Intermediate',
      inLanguage: 'zh-CN',
      isAccessibleForFree: true,
      url: 'https://minimind.wiki',
      hasCourseInstance: {
        '@type': 'CourseInstance',
        courseMode: 'online',
        courseWorkload: 'PT30H'
      }
    })],
  ],

  // 主题配置
  themeConfig: {
    // Logo
    logo: '/logo.svg',

    // 顶部导航
    nav: [
      { text: '首页', link: '/docs/' },
      {
        text: '📚 学习指南',
        items: [
          { text: '⚡ 快速体验 (30分钟)', link: '/docs/guide/quick-start' },
          { text: '📚 系统学习 (6小时)', link: '/docs/guide/systematic' },
          { text: '🎓 深度掌握 (30小时)', link: '/docs/guide/mastery' },
          { text: '🗺️ 完整路线图', link: '/ROADMAP' },
        ]
      },
      {
        text: '🧱 模块教学',
        items: [
          { text: '模块总览', link: '/modules/' },
          { text: '基础组件', link: '/modules/01-foundation/' },
          { text: '架构组装', link: '/modules/02-architecture/' },
        ]
      },
      {
        text: '📝 我的笔记',
        items: [
          { text: '📅 学习日志', link: '/learning_log' },
          { text: '📚 知识库', link: '/knowledge_base' },
          { text: '🗂️ 总索引', link: '/notes' },
        ]
      },
    ],

    // 侧边栏
    sidebar: {
      '/docs/guide/': [
        {
          text: '🚀 学习指南',
          items: [
            { text: '⚡ 快速体验', link: '/docs/guide/quick-start' },
            { text: '📚 系统学习', link: '/docs/guide/systematic' },
            { text: '🎓 深度掌握', link: '/docs/guide/mastery' },
            { text: '🗺️ 完整路线图', link: '/ROADMAP' },
          ]
        }
      ],

      '/modules/': [
        {
          text: '📖 模块总览',
          items: [
            { text: '模块导航', link: '/modules/' },
          ]
        },
        {
          text: '🧱 基础组件 (Foundation)',
          collapsed: false,
          items: [
            {
              text: '01 归一化 (Normalization)',
              link: '/modules/01-foundation/01-normalization/',
              items: [
                { text: '📖 教学文档', link: '/modules/01-foundation/01-normalization/teaching' },
                { text: '💻 代码导读', link: '/modules/01-foundation/01-normalization/code_guide' },
                { text: '❓ 自测题', link: '/modules/01-foundation/01-normalization/quiz' },
              ]
            },
            {
              text: '02 位置编码 (Position Encoding)',
              link: '/modules/01-foundation/02-position-encoding/',
              items: [
                { text: '📖 教学文档', link: '/modules/01-foundation/02-position-encoding/teaching' },
                { text: '💻 代码导读', link: '/modules/01-foundation/02-position-encoding/code_guide' },
                { text: '❓ 自测题', link: '/modules/01-foundation/02-position-encoding/quiz' },
              ]
            },
            {
              text: '03 注意力机制 (Attention)',
              link: '/modules/01-foundation/03-attention/',
              items: [
                { text: '📖 教学文档', link: '/modules/01-foundation/03-attention/teaching' },
                { text: '💻 代码导读', link: '/modules/01-foundation/03-attention/code_guide' },
                { text: '❓ 自测题', link: '/modules/01-foundation/03-attention/quiz' },
              ]
            },
            {
              text: '04 前馈网络 (FeedForward)',
              link: '/modules/01-foundation/04-feedforward/',
              items: [
                { text: '📖 教学文档', link: '/modules/01-foundation/04-feedforward/teaching' },
                { text: '💻 代码导读', link: '/modules/01-foundation/04-feedforward/code_guide' },
                { text: '❓ 自测题', link: '/modules/01-foundation/04-feedforward/quiz' },
              ]
            },
          ]
        },
        {
          text: '🏗️ 架构组装 (Architecture)',
          items: [
            { text: '架构总览', link: '/modules/02-architecture/' },
          ]
        }
      ],

      '/': [
        {
          text: '🚀 开始',
          items: [
            { text: '首页', link: '/docs/' },
            { text: '学习指南', link: '/docs/guide/quick-start' },
          ]
        },
        {
          text: '📝 我的学习笔记',
          items: [
            { text: '📅 学习日志', link: '/learning_log' },
            { text: '📚 知识库', link: '/knowledge_base' },
            { text: '🗂️ 总索引', link: '/notes' },
          ]
        },
        {
          text: '🧱 模块化教学',
          items: [
            { text: '模块总览', link: '/modules/' },
            { text: '基础组件', link: '/modules/01-foundation/' },
            { text: '架构组装', link: '/modules/02-architecture/' },
          ]
        },
        {
          text: '💻 代码示例',
          items: [
            { text: '示例总览', link: '/learning_materials/README' },
          ]
        },
        {
          text: '📖 参考文档',
          items: [
            { text: 'Claude 使用指南', link: '/CLAUDE' },
            { text: '笔记更新指南', link: '/NOTE_UPDATE_GUIDE' },
            { text: '学习路线图', link: '/ROADMAP' },
          ]
        }
      ]
    },

    // 社交链接
    socialLinks: [
      { icon: 'github', link: 'https://github.com/joyehuang/minimind-notes' }
    ],

    // 搜索
    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档'
              },
              modal: {
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
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

    // 页脚
    footer: {
      message: '基于 <a href="https://github.com/jingyaogong/minimind" target="_blank">MiniMind</a> 项目的学习笔记',
      copyright: 'Copyright © 2025 joyehuang'
    },

    // 编辑链接
    editLink: {
      pattern: 'https://github.com/joyehuang/minimind-notes/edit/main/:path',
      text: '在 GitHub 上编辑此页'
    },

    // 最后更新时间
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short'
      }
    },

    // 文档页脚导航
    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    // 大纲配置
    outline: {
      level: [2, 3],
      label: '目录'
    },

    // 返回顶部
    returnToTopLabel: '返回顶部',

    // 侧边栏菜单标签
    sidebarMenuLabel: '菜单',

    // 深色模式标签
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
  },

  // Markdown 配置
  markdown: {
    // 启用数学公式支持
    math: true,

    // 代码块行号
    lineNumbers: true,

    // 图片懒加载
    image: {
      lazyLoading: true
    },

    // 主题配置
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },

  // 站点地图
  sitemap: {
    hostname: 'https://minimind.wiki',
    transformItems: (items) => {
      // 排除不应该被索引的内部文档
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
