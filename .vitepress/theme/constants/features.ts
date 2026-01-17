/**
 * FeaturesCards component translations
 * Extracted to module-level constants to reduce memory overhead
 */

export interface FeatureCopy {
  badge: string
  title: string
  description: string
}

export interface FeatureData {
  icon: string
  title: string
  description: string
}

export const FEATURES_COPY = {
  en: {
    badge: 'Core Highlights',
    title: 'Truly understand how LLMs are trained',
    description: 'No more blind training — use controlled experiments to understand every design choice.'
  } as FeatureCopy,
  zh: {
    badge: '核心特点',
    title: '彻底理解 LLM 训练原理',
    description: '告别"跑通就行"的盲目训练 — 通过对照实验，深入每个设计选择背后的原理'
  } as FeatureCopy
} as const

export const FEATURES_DATA = {
  en: [
    {
      icon: 'lightbulb',
      title: 'Principles First',
      description: 'No black boxes — understand the tradeoffs behind every design decision.'
    },
    {
      icon: 'flask',
      title: 'Controlled Experiments',
      description: 'Show, don\'t tell — run experiments to see what breaks without a design.'
    },
    {
      icon: 'blocks',
      title: 'Modular Learning',
      description: 'From normalization to Transformer — 6 independent modules, progressive and clear.'
    },
    {
      icon: 'zap',
      title: 'Low Barrier',
      description: 'Tiny datasets run in minutes on CPU — verify ideas fast and cheaply.'
    }
  ] as FeatureData[],
  zh: [
    {
      icon: 'lightbulb',
      title: '原理优先',
      description: '不再黑盒训练 — 深入理解每个设计背后的原理和权衡，知其然更知其所以然'
    },
    {
      icon: 'flask',
      title: '对照实验',
      description: '不凭空说教 — 用可执行实验证明：不这样设计会发生什么？眼见为实'
    },
    {
      icon: 'blocks',
      title: '模块化学习',
      description: '从归一化到 Transformer — 6 个独立模块，渐进式掌握，由浅入深'
    },
    {
      icon: 'zap',
      title: '学习实验低门槛',
      description: '基于微型数据集，CPU 上几分钟即可运行，快速验证理论，降低学习成本'
    }
  ] as FeatureData[]
} as const
