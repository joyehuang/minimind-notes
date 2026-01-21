// .vitepress/theme/index.ts
import DefaultTheme from 'vitepress/theme'
import { injectSpeedInsights } from '@vercel/speed-insights'
import { inject } from '@vercel/analytics'
import { defineAsyncComponent } from 'vue'
import Layout from './Layout.vue'
import './style.css'

// Use dynamic imports for code splitting
// Components are loaded only when actually used on a page
const QuickStartTimeline = defineAsyncComponent(() => import('./components/QuickStartTimeline.vue'))
const InteractiveQuiz = defineAsyncComponent(() => import('./components/InteractiveQuiz.vue'))
const FeaturesCards = defineAsyncComponent(() => import('./components/FeaturesCards.vue'))
const LearningPathCards = defineAsyncComponent(() => import('./components/LearningPathCards.vue'))
const ModulesGrid = defineAsyncComponent(() => import('./components/ModulesGrid.vue'))
const TerminalCode = defineAsyncComponent(() => import('./components/TerminalCode.vue'))
const HomeHeroVideo = defineAsyncComponent(() => import('./components/HomeHeroVideo.vue'))

// 注入 Vercel Analytics 和 Speed Insights
// 使用 requestIdleCallback 延迟加载，避免阻塞主线程
if (typeof window !== 'undefined') {
  const loadAnalytics = () => {
    inject()
    injectSpeedInsights()
  }

  // 使用 requestIdleCallback 在浏览器空闲时加载
  if ('requestIdleCallback' in window) {
    requestIdleCallback(loadAnalytics)
  } else {
    // 降级方案：使用 setTimeout
    setTimeout(loadAnalytics, 1)
  }
}

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    // 注册全局组件
    app.component('QuickStartTimeline', QuickStartTimeline)
    app.component('InteractiveQuiz', InteractiveQuiz)
    app.component('FeaturesCards', FeaturesCards)
    app.component('LearningPathCards', LearningPathCards)
    app.component('ModulesGrid', ModulesGrid)
    app.component('TerminalCode', TerminalCode)
    app.component('HomeHeroVideo', HomeHeroVideo)
  }
}
