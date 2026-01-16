/**
 * LearningPathCards component translations
 * Extracted to module-level constants to reduce memory overhead
 */

export interface LearningPathCopy {
  badge: string
  title: string
  description: string
  cta: string
}

export interface PathData {
  icon: string
  title: string
  duration: string
  description: string
  badge: string
  variant: 'primary' | 'secondary' | 'tertiary'
  link: string
}

export const LEARNING_PATH_COPY = {
  en: {
    badge: 'Pick Your Learning Path',
    title: 'Choose the best path for your time and goals',
    description: 'Different paths for different needs — from quick taste to deep mastery.',
    cta: 'Start Learning'
  } as LearningPathCopy,
  zh: {
    badge: '选择你的学习路径',
    title: '根据时间和目标选择合适的学习路线',
    description: '不同路径适合不同需求 — 从快速体验到深度掌握，循序渐进',
    cta: '开始学习'
  } as LearningPathCopy
} as const

// Path data factory functions (require localePath parameter)
export const createLearningPaths = (localePath: string, isEn: boolean): PathData[] => {
  const withLocale = (path: string) => `${localePath}${path}`

  return isEn
    ? [
        {
          icon: 'zap',
          title: 'Quick Start',
          duration: '30 min',
          description: 'Use 3 experiments to grasp key LLM design choices. Great for first timers.',
          badge: 'Most Popular',
          variant: 'primary',
          link: withLocale('/ROADMAP#quick-start-30-min')
        },
        {
          icon: 'book',
          title: 'Systematic Study',
          duration: '6 hours',
          description: 'Master all Transformer fundamentals with a complete, structured path.',
          badge: 'Comprehensive',
          variant: 'secondary',
          link: withLocale('/ROADMAP#systematic-study-6-hours')
        },
        {
          icon: 'graduation',
          title: 'Deep Mastery',
          duration: '30+ hours',
          description: 'Train a full LLM from scratch and go deep into architecture and training.',
          badge: 'Ultimate Challenge',
          variant: 'tertiary',
          link: withLocale('/ROADMAP#deep-mastery-30-hours')
        }
      ]
    : [
        {
          icon: 'zap',
          title: '快速体验',
          duration: '30 分钟',
          description: '用 3 个实验快速理解 LLM 训练的核心设计选择，适合初次接触',
          badge: '最受欢迎',
          variant: 'primary',
          link: withLocale('/ROADMAP#-快速体验-30-分钟')
        },
        {
          icon: 'book',
          title: '系统学习',
          duration: '6 小时',
          description: '完整掌握 Transformer 的所有基础组件，适合系统学习',
          badge: '系统全面',
          variant: 'secondary',
          link: withLocale('/ROADMAP#-系统学习-6-小时')
        },
        {
          icon: 'graduation',
          title: '深度掌握',
          duration: '30+ 小时',
          description: '从零开始完整训练你的第一个 LLM，适合深入研究',
          badge: '终极挑战',
          variant: 'tertiary',
          link: withLocale('/ROADMAP#-深度掌握-30-小时')
        }
      ]
}
