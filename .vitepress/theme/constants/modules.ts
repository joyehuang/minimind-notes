/**
 * ModulesGrid component translations and module data
 * Extracted to module-level constants to reduce memory overhead
 */

export interface ModuleCopy {
  badge: string
  title: string
  description: string
  tier1Label: string
  tier1Description: string
  tier2Label: string
  tier2Description: string
  experimentLabel: string
  experimentsLabel: string
  legendComplete: string
  legendPartial: string
  legendPlanned: string
}

export interface ModuleData {
  id: string
  icon: string
  title: string
  question: string
  experiments?: number
  status: 'complete' | 'partial' | 'planned'
  link: string
}

export const MODULES_COPY = {
  en: {
    badge: 'Module Navigator',
    title: 'From core components to full architecture',
    description: 'Modular learning path — each module is self-contained and can be learned in any order.',
    tier1Label: 'Tier 1 · Foundation',
    tier1Description: 'Core components — master the building blocks of Transformer',
    tier2Label: 'Tier 2 · Architecture',
    tier2Description: 'Assembly — combine components into a full Transformer',
    experimentLabel: 'experiment',
    experimentsLabel: 'experiments',
    legendComplete: 'Complete: teaching + experiments + quiz',
    legendPartial: 'Experiments done: docs in progress',
    legendPlanned: 'Planned: structure only'
  } as ModuleCopy,
  zh: {
    badge: '模块导航',
    title: '从基础组件到完整架构',
    description: '模块化学习路径 — 每个模块独立完整，可按任意顺序学习',
    tier1Label: 'Tier 1 · 基础组件',
    tier1Description: '基础组件 — 掌握 Transformer 的核心模块',
    tier2Label: 'Tier 2 · 架构组装',
    tier2Description: '架构组装 — 将基础组件组合成完整 Transformer',
    experimentLabel: '个实验',
    experimentsLabel: '个实验',
    legendComplete: '完整：包含教学文档 + 实验代码 + 自测题',
    legendPartial: '实验完成：有实验代码，文档待补充',
    legendPlanned: '待开发：仅目录结构'
  } as ModuleCopy
} as const

// Module data factory functions (require localePath parameter)
export const createTier1Modules = (localePath: string, isEn: boolean): ModuleData[] => {
  const withLocale = (path: string) => `${localePath}${path}`

  return isEn
    ? [
        {
          id: '01-normalization',
          icon: 'waves',
          title: 'Normalization',
          question: 'Why is normalization necessary? How different are Pre-LN and Post-LN?',
          experiments: 2,
          status: 'complete',
          link: withLocale('/modules/01-foundation/01-normalization/')
        },
        {
          id: '02-position-encoding',
          icon: 'compass',
          title: 'Position Encoding',
          question: 'Why did RoPE become the default? Is it really better?',
          experiments: 4,
          status: 'complete',
          link: withLocale('/modules/01-foundation/02-position-encoding/')
        },
        {
          id: '03-attention',
          icon: 'eye',
          title: 'Attention',
          question: 'What do Q/K/V really do? Are multi-heads necessary or overkill?',
          experiments: 3,
          status: 'complete',
          link: withLocale('/modules/01-foundation/03-attention/')
        },
        {
          id: '04-feedforward',
          icon: 'layers',
          title: 'FeedForward',
          question: 'Why can FFN store knowledge? Is 4x expansion optimal?',
          experiments: 1,
          status: 'complete',
          link: withLocale('/modules/01-foundation/04-feedforward/')
        }
      ]
    : [
        {
          id: '01-normalization',
          icon: 'waves',
          title: '归一化',
          question: '为什么必须归一化？Pre-LN 和 Post-LN 差异有多大？',
          experiments: 2,
          status: 'complete',
          link: withLocale('/modules/01-foundation/01-normalization/')
        },
        {
          id: '02-position-encoding',
          icon: 'compass',
          title: '位置编码',
          question: 'RoPE 为什么成为标配？它真的比其他方案好吗？',
          experiments: 4,
          status: 'complete',
          link: withLocale('/modules/01-foundation/02-position-encoding/')
        },
        {
          id: '03-attention',
          icon: 'eye',
          title: '注意力机制',
          question: 'QKV 到底在做什么？多头是必需还是过度设计？',
          experiments: 3,
          status: 'complete',
          link: withLocale('/modules/01-foundation/03-attention/')
        },
        {
          id: '04-feedforward',
          icon: 'layers',
          title: '前馈网络',
          question: 'FFN 为什么能存储知识？扩张比 4x 是最佳选择吗？',
          experiments: 1,
          status: 'complete',
          link: withLocale('/modules/01-foundation/04-feedforward/')
        }
      ]
}

export const createTier2Modules = (localePath: string, isEn: boolean): ModuleData[] => {
  const withLocale = (path: string) => `${localePath}${path}`

  return isEn
    ? [
        {
          id: '01-residual-connection',
          icon: 'link',
          title: 'Residual Connection',
          question: 'The savior for deep nets — or something else?',
          status: 'planned',
          link: withLocale('/modules/02-architecture/01-residual-connection/')
        },
        {
          id: '02-transformer-block',
          icon: 'box',
          title: 'Transformer Block',
          question: 'The golden assembly order — why this one?',
          status: 'planned',
          link: withLocale('/modules/02-architecture/02-transformer-block/')
        }
      ]
    : [
        {
          id: '01-residual-connection',
          icon: 'link',
          title: '残差连接',
          question: '深层网络训练的救星？还是另有玄机？',
          status: 'planned',
          link: withLocale('/modules/02-architecture/01-residual-connection/')
        },
        {
          id: '02-transformer-block',
          icon: 'box',
          title: 'Transformer Block',
          question: '组件组装的黄金顺序 — 为什么是这一个？',
          status: 'planned',
          link: withLocale('/modules/02-architecture/02-transformer-block/')
        }
      ]
}

export const STATUS_TEXT = {
  en: {
    complete: 'Complete',
    partial: 'Experiments done',
    planned: 'Planned'
  },
  zh: {
    complete: '完整',
    partial: '实验完成',
    planned: '待开发'
  }
} as const
