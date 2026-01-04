// .vitepress/theme/index.ts
import DefaultTheme from 'vitepress/theme'
import { injectSpeedInsights } from '@vercel/speed-insights'
import { inject } from '@vercel/analytics'
import QuickStartTimeline from './components/QuickStartTimeline.vue'
import InteractiveQuiz from './components/InteractiveQuiz.vue'
import './style.css'

// 注入 Vercel Analytics 和 Speed Insights
if (typeof window !== 'undefined') {
  inject()
  injectSpeedInsights()
}

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // 注册全局组件
    app.component('QuickStartTimeline', QuickStartTimeline)
    app.component('InteractiveQuiz', InteractiveQuiz)
  }
}
