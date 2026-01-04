<template>
  <div class="quick-start-timeline">
    <div class="timeline-header">
      <h3>⚡ 30分钟快速体验路线</h3>
      <p>通过 3 个关键实验理解核心设计选择</p>
    </div>

    <div class="timeline-container">
      <div
        v-for="(step, index) in steps"
        :key="index"
        class="timeline-step"
        :class="{ 'step-active': activeStep >= index }"
        @mouseenter="activeStep = index"
      >
        <div class="step-connector" v-if="index < steps.length - 1">
          <div class="connector-line"></div>
          <div class="connector-arrow">→</div>
        </div>

        <div class="step-content">
          <div class="step-number">{{ index + 1 }}</div>
          <div class="step-icon">{{ step.icon }}</div>
          <h4 class="step-title">{{ step.title }}</h4>
          <p class="step-description">{{ step.description }}</p>
          <div class="step-meta">
            <span class="step-duration">⏱️ {{ step.duration }}</span>
            <span class="step-module">{{ step.module }}</span>
          </div>
          <a :href="step.link" class="step-link">
            开始实验 →
          </a>
        </div>
      </div>
    </div>

    <div class="timeline-footer">
      <a href="/docs/guide/quick-start" class="view-full-guide">
        查看完整快速入门指南 →
      </a>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const activeStep = ref(0)

const steps = [
  {
    icon: '📊',
    title: '为什么需要归一化？',
    description: '观察梯度消失现象，理解 RMSNorm 的作用',
    duration: '5 分钟',
    module: '归一化',
    link: '/modules/01-foundation/01-normalization/teaching'
  },
  {
    icon: '📐',
    title: '为什么用 RoPE？',
    description: '对比绝对位置编码，理解旋转位置编码的优势',
    duration: '10 分钟',
    module: '位置编码',
    link: '/modules/01-foundation/02-position-encoding/teaching'
  },
  {
    icon: '🔗',
    title: '为什么需要残差连接？',
    description: '实验验证深层网络训练的梯度流问题',
    duration: '5 分钟',
    module: '残差连接',
    link: '/modules/02-architecture/01-residual-connection/teaching'
  }
]
</script>

<style scoped>
.quick-start-timeline {
  margin: 3rem 0;
  padding: 2rem;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  border: 1px solid var(--vp-c-divider);
}

.timeline-header {
  text-align: center;
  margin-bottom: 3rem;
}

.timeline-header h3 {
  margin: 0 0 0.5rem 0;
  font-size: 1.8em;
  color: var(--vp-c-brand-1);
}

.timeline-header p {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 1.1em;
}

.timeline-container {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  position: relative;
  margin-bottom: 2rem;
}

.timeline-step {
  flex: 1;
  position: relative;
  opacity: 0;
  transform: translateX(-20px);
  animation: slideIn 0.6s ease-out forwards;
}

.timeline-step:nth-child(1) {
  animation-delay: 0.1s;
}

.timeline-step:nth-child(2) {
  animation-delay: 0.3s;
}

.timeline-step:nth-child(3) {
  animation-delay: 0.5s;
}

@keyframes slideIn {
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.step-connector {
  position: absolute;
  top: 50%;
  right: -1rem;
  transform: translateY(-50%);
  z-index: 1;
  display: flex;
  align-items: center;
  width: 2rem;
}

.connector-line {
  flex: 1;
  height: 2px;
  background: linear-gradient(to right, var(--vp-c-brand-1), var(--vp-c-brand-2));
  opacity: 0.5;
}

.connector-arrow {
  font-size: 1.5em;
  color: var(--vp-c-brand-1);
  margin-left: -0.3rem;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 0.5;
    transform: translateX(0);
  }
  50% {
    opacity: 1;
    transform: translateX(3px);
  }
}

.step-content {
  background: var(--vp-c-bg);
  border: 2px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 1.5rem;
  transition: all 0.3s ease;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.timeline-step:hover .step-content,
.timeline-step.step-active .step-content {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 4px 20px rgba(var(--vp-c-brand-rgb), 0.2);
  transform: translateY(-5px);
}

.step-number {
  position: absolute;
  top: -12px;
  left: 50%;
  transform: translateX(-50%);
  width: 32px;
  height: 32px;
  background: var(--vp-c-brand-1);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 0.9em;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.step-icon {
  font-size: 3em;
  text-align: center;
  margin: 1rem 0;
}

.step-title {
  margin: 0.5rem 0;
  font-size: 1.1em;
  color: var(--vp-c-text-1);
  text-align: center;
  line-height: 1.4;
}

.step-description {
  margin: 0.5rem 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
  text-align: center;
  flex-grow: 1;
  line-height: 1.6;
}

.step-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 1rem 0 0.5rem 0;
  padding-top: 1rem;
  border-top: 1px solid var(--vp-c-divider);
  font-size: 0.85em;
}

.step-duration {
  color: var(--vp-c-text-2);
}

.step-module {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  padding: 0.2rem 0.6rem;
  border-radius: 4px;
  font-weight: 500;
}

.step-link {
  display: inline-block;
  width: 100%;
  text-align: center;
  padding: 0.6rem 1rem;
  background: var(--vp-c-brand-1);
  color: white;
  text-decoration: none;
  border-radius: 6px;
  font-weight: 500;
  transition: all 0.2s ease;
  margin-top: 0.5rem;
}

.step-link:hover {
  background: var(--vp-c-brand-2);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(var(--vp-c-brand-rgb), 0.3);
}

.timeline-footer {
  text-align: center;
  padding-top: 1.5rem;
  border-top: 1px solid var(--vp-c-divider);
}

.view-full-guide {
  display: inline-block;
  color: var(--vp-c-brand-1);
  text-decoration: none;
  font-weight: 500;
  font-size: 1.05em;
  transition: all 0.2s ease;
}

.view-full-guide:hover {
  color: var(--vp-c-brand-2);
  transform: translateX(5px);
}

/* Mobile responsive */
@media (max-width: 768px) {
  .quick-start-timeline {
    padding: 1.5rem 1rem;
  }

  .timeline-container {
    flex-direction: column;
    gap: 2rem;
  }

  .step-connector {
    top: auto;
    bottom: -2rem;
    left: 50%;
    right: auto;
    transform: translateX(-50%) rotate(90deg);
    width: 2rem;
  }

  .connector-arrow {
    animation: pulseVertical 2s ease-in-out infinite;
  }

  @keyframes pulseVertical {
    0%, 100% {
      opacity: 0.5;
      transform: translateX(0);
    }
    50% {
      opacity: 1;
      transform: translateX(3px);
    }
  }

  .timeline-step {
    width: 100%;
  }

  .step-icon {
    font-size: 2.5em;
  }

  .step-title {
    font-size: 1em;
  }
}
</style>
