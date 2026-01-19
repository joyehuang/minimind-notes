<template>
  <section class="features-section">
    <div class="features-container">
      <div class="features-header">
        <div class="header-badge">{{ copy.badge }}</div>
        <h2>{{ copy.title }}</h2>
        <p>{{ copy.description }}</p>
      </div>

      <div class="features-grid">
        <div
          v-for="(feature, index) in features"
          :key="feature.title"
          class="feature-card"
          :style="{ animationDelay: `${index * 0.1}s` }"
        >
          <div class="feature-icon-wrapper">
            <Icon :name="feature.icon" class-name="feature-icon" />
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
import { computed } from 'vue'
import { useLocale } from '../i18n'
import { FEATURES_COPY, FEATURES_DATA } from '../constants/features'
import Icon from './Icon.vue'

const { isEn } = useLocale()

// Use pre-defined translation constants
const copy = computed(() => isEn.value ? FEATURES_COPY.en : FEATURES_COPY.zh)
const features = computed(() => isEn.value ? FEATURES_DATA.en : FEATURES_DATA.zh)
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
  line-height: 1.25;
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
