// VitePress 配置示例
// 保存为 docs/.vitepress/config.ts

import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'MiniMind 学习笔记',
  description: '深入理解 LLM 训练的每个设计选择',
  lang: 'zh-CN',

  // 主题配置
  themeConfig: {
    logo: '/logo.svg',

    // 顶部导航
    nav: [
      { text: '首页', link: '/' },
      {
        text: '📚 学习指南',
        items: [
          { text: '⚡ 快速体验 (30分钟)', link: '/guide/quick-start' },
          { text: '📚 系统学习 (6小时)', link: '/guide/systematic' },
          { text: '🎓 深度掌握 (30小时)', link: '/guide/mastery' },
        ]
      },
      { text: '🧱 模块教学', link: '/modules/' },
      { text: '📝 我的笔记', link: '/notes/learning-log' },
    ],

    // 侧边栏
    sidebar: {
      '/guide/': [
        {
          text: '学习指南',
          items: [
            { text: '快速开始', link: '/guide/quick-start' },
            { text: '学习路线图', link: '/guide/roadmap' },
            { text: '学习方法', link: '/guide/learning-methods' },
          ]
        }
      ],

      '/modules/': [
        {
          text: '🧱 基础组件 (Foundation)',
          collapsed: false,
          items: [
            {
              text: '01 归一化 (Normalization)',
              link: '/modules/foundation/01-normalization/',
              items: [
                { text: '教学文档', link: '/modules/foundation/01-normalization/teaching' },
                { text: '代码导读', link: '/modules/foundation/01-normalization/code-guide' },
                { text: '自测题', link: '/modules/foundation/01-normalization/quiz' },
              ]
            },
            {
              text: '02 位置编码 (Position Encoding)',
              link: '/modules/foundation/02-position-encoding/',
              items: [
                { text: '教学文档', link: '/modules/foundation/02-position-encoding/teaching' },
                { text: '代码导读', link: '/modules/foundation/02-position-encoding/code-guide' },
                { text: '自测题', link: '/modules/foundation/02-position-encoding/quiz' },
              ]
            },
            {
              text: '03 注意力机制 (Attention)',
              link: '/modules/foundation/03-attention/',
              items: [
                { text: '教学文档', link: '/modules/foundation/03-attention/teaching' },
                { text: '代码导读', link: '/modules/foundation/03-attention/code-guide' },
                { text: '自测题', link: '/modules/foundation/03-attention/quiz' },
              ]
            },
            {
              text: '04 前馈网络 (FeedForward)',
              link: '/modules/foundation/04-feedforward/',
              items: [
                { text: '教学文档', link: '/modules/foundation/04-feedforward/teaching' },
                { text: '代码导读', link: '/modules/foundation/04-feedforward/code-guide' },
                { text: '自测题', link: '/modules/foundation/04-feedforward/quiz' },
              ]
            },
          ]
        },
        {
          text: '🏗️ 架构组装 (Architecture)',
          collapsed: false,
          items: [
            { text: '残差连接', link: '/modules/architecture/01-residual-connection/' },
            { text: 'Transformer Block', link: '/modules/architecture/02-transformer-block/' },
          ]
        }
      ],

      '/notes/': [
        {
          text: '我的学习笔记',
          items: [
            { text: '📅 学习日志', link: '/notes/learning-log' },
            { text: '📚 知识库', link: '/notes/knowledge-base' },
            { text: '💻 代码示例', link: '/notes/materials/' },
            { text: '❓ 问答集', link: '/notes/qa' },
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
                  navigateText: '切换'
                }
              }
            }
          }
        }
      }
    },

    // 页脚
    footer: {
      message: '基于 <a href="https://github.com/jingyaogong/minimind">MiniMind</a> 项目的学习笔记',
      copyright: 'Copyright © 2025'
    },

    // 编辑链接
    editLink: {
      pattern: 'https://github.com/joyehuang/minimind-notes/edit/master/:path',
      text: '在 GitHub 上编辑此页'
    },

    // 最后更新时间
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    }
  },

  // Markdown 配置
  markdown: {
    math: true,  // 启用数学公式 (KaTeX)
    lineNumbers: true,  // 代码块显示行号

    // 代码组
    codeTransformers: [
      // 可以添加代码高亮等
    ]
  },

  // 构建配置
  srcDir: '.',  // 源目录
  outDir: '.vitepress/dist',  // 输出目录

  // 路由配置
  cleanUrls: true,  // 清理 URL (去掉 .html)

  // 站点地图
  sitemap: {
    hostname: 'https://joyehuang.github.io/minimind-notes'
  }
})
