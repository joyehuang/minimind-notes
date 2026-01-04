import { defineConfig } from 'vitepress'

export default defineConfig({
  // 站点信息
  title: 'MiniMind 学习笔记',
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
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#3b82f6' }],
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
    hostname: 'https://minimind-notes.vercel.app'
  }
})
