// .vitepress/theme/index.ts
import DefaultTheme from 'vitepress/theme'
import { injectSpeedInsights } from '@vercel/speed-insights'
import { inject } from '@vercel/analytics'
import './style.css'

// 注入 Vercel Analytics 和 Speed Insights
if (typeof window !== 'undefined') {
  inject()
  injectSpeedInsights()
}

export default DefaultTheme
