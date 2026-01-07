<template>
  <section class="features-section">
    <div class="features-container">
      <div class="features-header">
        <div class="header-badge">核心特点</div>
        <h2>彻底理解 LLM 训练原理</h2>
        <p>告别"跑通就行"的盲目训练 — 通过对照实验，深入每个设计选择背后的原理</p>
      </div>

      <div class="features-grid">
        <div
          v-for="(feature, index) in features"
          :key="feature.title"
          class="feature-card"
          :style="{ animationDelay: `${index * 0.1}s` }"
        >
          <div class="feature-icon-wrapper">
            <svg v-if="feature.icon === 'lightbulb'" class="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M9 18h6" />
              <path d="M10 22h4" />
              <path d="M12 2v1" />
              <path d="M12 18v-2.5" />
              <path d="M12 15.5a5.5 5.5 0 1 0-4-8.5" />
            </svg>
            <svg v-else-if="feature.icon === 'flask'" class="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M10 2v7.31" />
              <path d="M14 2v7.31" />
              <path d="M8.5 2h7" />
              <path d="M14 9.3a6.5 6.5 0 1 1-4 0" />
              <path d="M5.3 21h13.4" />
            </svg>
            <svg v-else-if="feature.icon === 'blocks'" class="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
              <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
              <line x1="12" y1="22.08" x2="12" y2="12" />
            </svg>
            <svg v-else-if="feature.icon === 'zap'" class="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
            </svg>
          </div>

          <div class="feature-content">
            <h3>{{ feature.title }}</h3>
            <p>{{ feature.description }}</p>
          </div>

          <div class="feature-glow"></div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
const features = [
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
]
</script>

<style scoped>
.features-section {
  padding: 5rem 0;
  position: relative;
  overflow: hidden;
}

/* 背景渐变装饰 */
.features-section::before {
  content: '';
  position: absolute;
  top: -50%;
  right: -10%;
  width: 600px;
  height: 600px;
  background: radial-gradient(circle, var(--vp-c-brand-1) 0%, transparent 70%);
  opacity: 0.03;
  border-radius: 50%;
  pointer-events: none;
}

.features-section::after {
  content: '';
  position: absolute;
  bottom: -30%;
  left: -5%;
  width: 500px;
  height: 500px;
  background: radial-gradient(circle, var(--vp-c-brand-2) 0%, transparent 70%);
  opacity: 0.03;
  border-radius: 50%;
  pointer-events: none;
}

.features-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1.5rem;
  position: relative;
  z-index: 1;
}

.features-header {
  text-align: center;
  margin-bottom: 4rem;
  max-width: 700px;
  margin-left: auto;
  margin-right: auto;
}

.header-badge {
  display: inline-block;
  padding: 0.5rem 1rem;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  border-radius: 20px;
  font-size: 0.85em;
  font-weight: 600;
  margin-bottom: 1.5rem;
  letter-spacing: 0.5px;
}

:global(.dark) .header-badge {
  background: rgba(var(--vp-c-brand-rgb), 0.15);
  border: 1px solid rgba(var(--vp-c-brand-rgb), 0.2);
}

.features-header h2 {
  font-size: 2.5em;
  font-weight: 700;
  margin: 0 0 1rem 0;
  color: var(--vp-c-text-1);
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, var(--vp-c-text-1) 0%, var(--vp-c-text-2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.features-header p {
  font-size: 1.15em;
  color: var(--vp-c-text-2);
  margin: 0;
  line-height: 1.7;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1.5rem;
}

.feature-card {
  position: relative;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  padding: 2rem;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  overflow: hidden;
  opacity: 0;
  animation: fadeSlideUp 0.6s ease-out forwards;
}

@keyframes fadeSlideUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.feature-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg,
    rgba(var(--vp-c-brand-rgb), 0.05) 0%,
    rgba(var(--vp-c-brand-rgb), 0) 100%);
  opacity: 0;
  transition: opacity 0.4s ease;
  border-radius: 16px;
}

.feature-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-8px);
  box-shadow:
    0 20px 40px rgba(0, 0, 0, 0.1),
    0 0 0 1px rgba(var(--vp-c-brand-rgb), 0.1);
}

:global(.dark) .feature-card:hover {
  box-shadow:
    0 20px 40px rgba(0, 0, 0, 0.3),
    0 0 0 1px rgba(var(--vp-c-brand-rgb), 0.15);
}

.feature-card:hover::before {
  opacity: 1;
}

.feature-icon-wrapper {
  width: 56px;
  height: 56px;
  background: linear-gradient(135deg,
    var(--vp-c-brand-soft) 0%,
    rgba(var(--vp-c-brand-rgb), 0.1) 100%);
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 1.5rem;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

:global(.dark) .feature-icon-wrapper {
  background: rgba(var(--vp-c-brand-rgb), 0.1);
  border: 1px solid rgba(var(--vp-c-brand-rgb), 0.2);
}

.feature-card:hover .feature-icon-wrapper {
  transform: scale(1.1) rotate(-5deg);
  background: var(--vp-c-brand-1);
}

.feature-card:hover .feature-icon {
  color: white;
}

.feature-icon {
  width: 28px;
  height: 28px;
  color: var(--vp-c-brand-1);
  transition: color 0.3s ease;
}

.feature-content h3 {
  font-size: 1.25em;
  font-weight: 600;
  margin: 0 0 0.75rem 0;
  color: var(--vp-c-text-1);
  letter-spacing: -0.01em;
}

.feature-content p {
  font-size: 0.95em;
  color: var(--vp-c-text-2);
  line-height: 1.7;
  margin: 0;
}

.feature-glow {
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle, var(--vp-c-brand-1) 0%, transparent 70%);
  opacity: 0;
  transition: opacity 0.4s ease;
  pointer-events: none;
}

.feature-card:hover .feature-glow {
  opacity: 0.03;
}

@media (max-width: 768px) {
  .features-section {
    padding: 3rem 0;
  }

  .features-header h2 {
    font-size: 2em;
  }

  .features-header p {
    font-size: 1em;
  }

  .features-grid {
    grid-template-columns: 1fr;
  }

  .feature-card {
    padding: 1.5rem;
  }
}
</style>
