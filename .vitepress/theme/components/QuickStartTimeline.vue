<template>
  <section class="quickstart-section">
    <div class="quickstart-container">
      <div class="section-header">
        <div class="header-badge">
          <svg class="badge-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
          </svg>
          30分钟快速体验
        </div>
        <h2>通过 3 个关键实验理解核心设计</h2>
        <p>每个实验只需 5-10 分钟，在 CPU 上即可运行，快速掌握 LLM 训练的核心秘密</p>
      </div>

      <div class="timeline-wrapper">
        <div class="timeline-line"></div>

        <div
          v-for="(step, index) in steps"
          :key="index"
          class="timeline-step"
          :class="{ 'step-active': activeStep === index }"
          @mouseenter="activeStep = index"
        >
          <div class="step-marker">
            <div class="marker-inner">
              <svg v-if="step.completed" class="check-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <svg v-else-if="step.icon === 'chart'" class="step-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M3 3v18h18" />
                <path d="M18 17V9" />
                <path d="M13 17V5" />
                <path d="M8 17v-3" />
              </svg>
              <svg v-else-if="step.icon === 'compass'" class="step-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <circle cx="12" cy="12" r="10" />
                <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
              </svg>
              <svg v-else-if="step.icon === 'link'" class="step-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
              </svg>
            </div>
            <div class="marker-glow"></div>
          </div>

          <div class="step-card-wrapper">
            <div class="step-card">
              <div class="step-header">
                <div class="step-number">Step 0{{ index + 1 }}</div>
                <div class="step-duration">
                  <svg class="clock-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                  {{ step.duration }}
                </div>
              </div>

              <h3>{{ step.title }}</h3>
              <p class="step-description">{{ step.description }}</p>

              <div class="step-meta">
                <span class="module-tag">
                  <svg class="tag-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z" />
                    <line x1="7" y1="7" x2="7.01" y2="7" />
                  </svg>
                  {{ step.module }}
                </span>
              </div>

              <a :href="step.link" class="step-button">
                开始实验
                <svg class="button-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <line x1="5" y1="12" x2="19" y2="12" />
                  <polyline points="12 5 19 12 12 19" />
                </svg>
              </a>
            </div>
          </div>
        </div>
      </div>

      <div class="section-footer">
        <a href="/ROADMAP" class="footer-link">
          查看完整学习路线图
          <svg class="link-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="5" y1="12" x2="19" y2="12" />
            <polyline points="12 5 19 12 12 19" />
          </svg>
        </a>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const activeStep = ref(0)

const steps = [
  {
    icon: 'chart',
    title: '为什么需要归一化？',
    description: '观察梯度消失现象，理解 RMSNorm 如何稳定训练过程',
    duration: '5 分钟',
    module: '归一化模块',
    link: '/modules/01-foundation/01-normalization/teaching',
    completed: false
  },
  {
    icon: 'compass',
    title: '为什么用 RoPE？',
    description: '对比绝对位置编码，理解旋转位置编码在外推性上的优势',
    duration: '10 分钟',
    module: '位置编码模块',
    link: '/modules/01-foundation/02-position-encoding/teaching',
    completed: false
  },
  {
    icon: 'link',
    title: '为什么需要残差连接？',
    description: '实验验证深层网络训练的梯度流问题，见证残差连接的威力',
    duration: '5 分钟',
    module: '残差连接模块',
    link: '/modules/02-architecture/01-residual-connection/teaching',
    completed: false
  }
]
</script>

<style scoped>
.quickstart-section {
  padding: 5rem 0;
  background: linear-gradient(180deg,
    transparent 0%,
    var(--vp-c-bg-soft) 50%,
    transparent 100%);
}

:global(.dark) .quickstart-section {
  background: linear-gradient(180deg,
    transparent 0%,
    rgba(var(--vp-c-brand-rgb), 0.03) 50%,
    transparent 100%);
}

.quickstart-container {
  max-width: 900px;
  margin: 0 auto;
  padding: 0 1.5rem;
}

.section-header {
  text-align: center;
  margin-bottom: 4rem;
}

.header-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 1.2rem;
  background: linear-gradient(135deg,
    var(--vp-c-brand-1) 0%,
    var(--vp-c-brand-2) 100%);
  color: white;
  border-radius: 25px;
  font-size: 0.9em;
  font-weight: 600;
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 15px rgba(var(--vp-c-brand-rgb), 0.3);
}

.badge-icon {
  width: 18px;
  height: 18px;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.1);
  }
}

.section-header h2 {
  font-size: 2.3em;
  font-weight: 700;
  margin: 0 0 1rem 0;
  color: var(--vp-c-text-1);
  letter-spacing: -0.02em;
}

.section-header p {
  font-size: 1.1em;
  color: var(--vp-c-text-2);
  line-height: 1.7;
  margin: 0;
}

.timeline-wrapper {
  position: relative;
  margin-bottom: 3rem;
}

.timeline-line {
  position: absolute;
  left: 31px;
  top: 0;
  bottom: 0;
  width: 2px;
  background: linear-gradient(180deg,
    var(--vp-c-brand-1) 0%,
    var(--vp-c-brand-2) 100%);
  opacity: 0.2;
}

.timeline-step {
  position: relative;
  margin-bottom: 3rem;
  padding-left: 80px;
  opacity: 0;
  animation: fadeInUp 0.6s ease-out forwards;
}

.timeline-step:nth-child(1) { animation-delay: 0.1s; }
.timeline-step:nth-child(2) { animation-delay: 0.2s; }
.timeline-step:nth-child(3) { animation-delay: 0.3s; }

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.timeline-step:last-child {
  margin-bottom: 0;
}

.step-marker {
  position: absolute;
  left: 0;
  top: 0;
  width: 62px;
  height: 62px;
  z-index: 2;
}

.marker-inner {
  width: 100%;
  height: 100%;
  background: var(--vp-c-bg);
  border: 2px solid var(--vp-c-divider);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  z-index: 2;
}

:global(.dark) .marker-inner {
  background: #09090b;
  border-color: var(--vp-c-divider);
}

.timeline-step:hover .marker-inner {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-1);
  transform: scale(1.1);
  box-shadow: 0 8px 25px rgba(var(--vp-c-brand-rgb), 0.3);
}

.timeline-step:hover .step-icon,
.timeline-step:hover .check-icon {
  color: white;
}

.step-icon,
.check-icon {
  width: 28px;
  height: 28px;
  color: var(--vp-c-brand-1);
  transition: color 0.3s ease;
}

.marker-glow {
  position: absolute;
  inset: -4px;
  background: radial-gradient(circle, var(--vp-c-brand-1) 0%, transparent 70%);
  border-radius: 20px;
  opacity: 0;
  transition: opacity 0.3s ease;
  z-index: 1;
}

.timeline-step:hover .marker-glow {
  opacity: 0.2;
}

.step-card-wrapper {
  position: relative;
}

.step-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  padding: 2rem;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}

.step-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg,
    rgba(var(--vp-c-brand-rgb), 0.03) 0%,
    transparent 100%);
  opacity: 0;
  transition: opacity 0.3s ease;
}

:global(.dark) .step-card {
  background: rgba(255, 255, 255, 0.02);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.timeline-step:hover .step-card {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08);
  transform: translateX(8px);
}

:global(.dark) .timeline-step:hover .step-card {
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
}

.timeline-step:hover .step-card::before {
  opacity: 1;
}

.step-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.step-number {
  font-size: 0.85em;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.step-duration {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.9em;
  color: var(--vp-c-text-2);
  padding: 0.4rem 0.8rem;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
}

:global(.dark) .step-duration {
  background: rgba(255, 255, 255, 0.05);
}

.clock-icon {
  width: 14px;
  height: 14px;
}

.step-card h3 {
  font-size: 1.4em;
  font-weight: 600;
  margin: 0 0 0.75rem 0;
  color: var(--vp-c-text-1);
  letter-spacing: -0.01em;
}

.step-description {
  font-size: 1em;
  color: var(--vp-c-text-2);
  line-height: 1.7;
  margin: 0 0 1.5rem 0;
}

.step-meta {
  display: flex;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}

.module-tag {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.5rem 0.85rem;
  background: rgba(var(--vp-c-brand-rgb), 0.08);
  color: var(--vp-c-brand-1);
  border-radius: 8px;
  font-size: 0.85em;
  font-weight: 500;
}

:global(.dark) .module-tag {
  background: rgba(var(--vp-c-brand-rgb), 0.15);
}

.tag-icon {
  width: 14px;
  height: 14px;
}

.step-button {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.85rem 1.5rem;
  background: var(--vp-c-brand-1);
  color: white;
  text-decoration: none;
  border-radius: 10px;
  font-weight: 600;
  font-size: 0.95em;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.step-button:hover {
  background: var(--vp-c-brand-2);
  transform: translateX(4px);
  box-shadow: 0 4px 15px rgba(var(--vp-c-brand-rgb), 0.3);
}

.button-arrow {
  width: 18px;
  height: 18px;
  transition: transform 0.3s ease;
}

.step-button:hover .button-arrow {
  transform: translateX(4px);
}

.section-footer {
  text-align: center;
  padding-top: 2rem;
}

.footer-link {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--vp-c-text-1);
  text-decoration: none;
  font-weight: 600;
  font-size: 1.05em;
  transition: all 0.3s ease;
  padding: 0.75rem 1.5rem;
  border-radius: 10px;
}

.footer-link:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-brand-1);
}

:global(.dark) .footer-link:hover {
  background: rgba(255, 255, 255, 0.05);
}

.link-arrow {
  width: 18px;
  height: 18px;
  transition: transform 0.3s ease;
}

.footer-link:hover .link-arrow {
  transform: translateX(4px);
}

@media (max-width: 768px) {
  .quickstart-section {
    padding: 3rem 0;
  }

  .section-header h2 {
    font-size: 1.8em;
  }

  .section-header p {
    font-size: 1em;
  }

  .timeline-step {
    padding-left: 0;
    margin-bottom: 2rem;
  }

  .timeline-line {
    display: none;
  }

  .step-marker {
    position: relative;
    margin-bottom: 1rem;
  }

  .step-card {
    padding: 1.5rem;
  }

  .step-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }

  .step-button {
    width: 100%;
    justify-content: center;
  }
}
</style>
