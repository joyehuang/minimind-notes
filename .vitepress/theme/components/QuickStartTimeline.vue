<template>
  <section class="quickstart-section">
    <div class="quickstart-container">
      <div class="section-header">
        <div class="header-badge">
          <Icon name="zap" class-name="badge-icon" />
          {{ copy.badge }}
        </div>
        <h2>{{ copy.title }}</h2>
        <p>{{ copy.description }}</p>
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
              <Icon v-if="step.completed" name="check" class-name="check-icon" />
              <Icon v-else :name="step.icon" class-name="step-icon" />
            </div>
            <div class="marker-glow"></div>
          </div>

          <div class="step-card-wrapper">
            <div class="step-card">
              <div class="step-header">
                <div class="step-number">{{ copy.stepLabel }} 0{{ index + 1 }}</div>
                <div class="step-duration">
                  <Icon name="clock" class-name="clock-icon" />
                  {{ step.duration }}
                </div>
              </div>

              <h3>{{ step.title }}</h3>
              <p class="step-description">{{ step.description }}</p>

              <div class="step-meta">
                <span class="module-tag">
                  <Icon name="tag" class-name="tag-icon" />
                  {{ step.module }}
                </span>
              </div>

              <a :href="step.link" class="step-button">
                {{ copy.cta }}
                <Icon name="arrow-right" class-name="button-arrow" />
              </a>
            </div>
          </div>
        </div>
      </div>

      <div class="section-footer">
        <a :href="withLocale('/ROADMAP')" class="footer-link">
          {{ copy.footer }}
          <Icon name="arrow-right" class-name="link-arrow" />
        </a>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useLocale } from '../i18n'
import Icon from './Icon.vue'

const activeStep = ref(0)
const { isEn, localePath } = useLocale()
const withLocale = (path: string) => `${localePath.value}${path}`

const copy = computed(() =>
  isEn.value
    ? {
        badge: 'Quick Start in 30 Minutes',
        title: 'Understand core design choices with 3 experiments',
        description: 'Each experiment takes 5–10 minutes on CPU. Quickly grasp the essentials behind LLM training.',
        stepLabel: 'Step',
        cta: 'Start Experiments',
        footer: 'View the full learning roadmap'
      }
    : {
        badge: '30分钟快速体验',
        title: '通过 3 个关键实验理解核心设计',
        description: '每个实验只需 5-10 分钟，在 CPU 上即可运行，快速掌握 LLM 训练的核心秘密',
        stepLabel: '步骤',
        cta: '开始实验',
        footer: '查看完整学习路线图'
      }
)

const steps = computed(() =>
  isEn.value
    ? [
        {
          icon: 'chart',
          title: 'Why normalization?',
          description: 'Observe gradient vanishing and see how RMSNorm stabilizes training.',
          duration: '5 min',
          module: 'Normalization',
          link: withLocale('/modules/01-foundation/01-normalization/'),
          completed: false
        },
        {
          icon: 'compass',
          title: 'Why RoPE?',
          description: 'Compare absolute position encoding and learn why RoPE extrapolates better.',
          duration: '10 min',
          module: 'Position Encoding',
          link: withLocale('/modules/01-foundation/02-position-encoding/'),
          completed: false
        },
        {
          icon: 'link',
          title: 'Why residual connections?',
          description: 'Validate gradient flow issues in deep nets and see the power of residuals.',
          duration: '5 min',
          module: 'Residual Connection',
          link: withLocale('/modules/02-architecture/01-residual-connection/'),
          completed: false
        }
      ]
    : [
        {
          icon: 'chart',
          title: '为什么需要归一化？',
          description: '观察梯度消失现象，理解 RMSNorm 如何稳定训练过程',
          duration: '5 分钟',
          module: '归一化模块',
          link: withLocale('/modules/01-foundation/01-normalization/'),
          completed: false
        },
        {
          icon: 'compass',
          title: '为什么用 RoPE？',
          description: '对比绝对位置编码，理解旋转位置编码在外推性上的优势',
          duration: '10 分钟',
          module: '位置编码模块',
          link: withLocale('/modules/01-foundation/02-position-encoding/'),
          completed: false
        },
        {
          icon: 'link',
          title: '为什么需要残差连接？',
          description: '实验验证深层网络训练的梯度流问题，见证残差连接的威力',
          duration: '5 分钟',
          module: '残差连接模块',
          link: withLocale('/modules/02-architecture/01-residual-connection/'),
          completed: false
        }
      ]
)
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
