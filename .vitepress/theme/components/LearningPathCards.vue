<template>
  <section class="learning-path-section">
    <div class="path-container">
      <div class="path-header">
        <div class="header-badge">{{ copy.badge }}</div>
        <h2>{{ copy.title }}</h2>
        <p>{{ copy.description }}</p>
      </div>

      <div class="paths-grid">
        <div
          v-for="(path, index) in paths"
          :key="path.title"
          class="path-card"
          :class="`path-${path.variant}`"
          :style="{ animationDelay: `${index * 0.1}s` }"
        >
          <div class="card-background"></div>
          <div class="card-content">
            <div class="path-icon-wrapper">
              <Icon :name="path.icon" class-name="path-icon" />
            </div>

            <div class="path-badge" :class="`badge-${path.variant}`">
              {{ path.badge }}
            </div>

            <h3>{{ path.title }}</h3>
            <p class="path-description">{{ path.description }}</p>

            <div class="path-duration">
              <Icon name="clock" class-name="clock-icon" />
              <span>{{ path.duration }}</span>
            </div>

            <a :href="path.link" class="path-button">
              {{ copy.cta }}
              <Icon name="arrow-right" class-name="button-arrow" />
            </a>
          </div>

          <div class="card-glow"></div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useLocale } from '../i18n'
import { LEARNING_PATH_COPY, createLearningPaths } from '../constants/learning-paths'
import Icon from './Icon.vue'

const { isEn, localePath } = useLocale()

// Use pre-defined translation constants
const copy = computed(() => isEn.value ? LEARNING_PATH_COPY.en : LEARNING_PATH_COPY.zh)
const paths = computed(() => createLearningPaths(localePath.value, isEn.value))
</script>

<style scoped>
.learning-path-section {
  padding: 5rem 0;
  position: relative;
  overflow: hidden;
}

/* 背景装饰 */
.learning-path-section::before {
  content: '';
  position: absolute;
  top: 20%;
  left: -10%;
  width: 500px;
  height: 500px;
  background: radial-gradient(circle, var(--vp-c-brand-2) 0%, transparent 70%);
  opacity: 0.03;
  border-radius: 50%;
  pointer-events: none;
}

.path-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1.5rem;
  position: relative;
  z-index: 1;
}

.path-header {
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

.path-header h2 {
  font-size: 2.3em;
  font-weight: 700;
  margin: 0 0 1rem 0;
  color: var(--vp-c-text-1);
  letter-spacing: -0.02em;
}

.path-header p {
  font-size: 1.1em;
  color: var(--vp-c-text-2);
  margin: 0;
  line-height: 1.7;
}

.paths-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
}

.path-card {
  position: relative;
  border-radius: 20px;
  padding: 2.5rem 2rem;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  opacity: 0;
  animation: fadeSlideUp 0.6s ease-out forwards;
  cursor: pointer;
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

.card-background {
  position: absolute;
  inset: 0;
  border-radius: 20px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  transition: all 0.4s ease;
}

:global(.dark) .path-card .card-background {
  background: rgba(255, 255, 255, 0.02);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.path-card:hover .card-background {
  transform: translateY(-4px);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

:global(.dark) .path-card:hover .card-background {
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}

.path-primary:hover .card-background {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 20px 40px rgba(var(--vp-c-brand-rgb), 0.2);
}

.path-secondary:hover .card-background {
  border-color: var(--vp-c-brand-2);
}

.path-tertiary:hover .card-background {
  border-color: var(--vp-c-text-2);
}

.card-content {
  position: relative;
  z-index: 2;
  display: flex;
  flex-direction: column;
  height: 100%;
}

.path-icon-wrapper {
  width: 64px;
  height: 64px;
  background: linear-gradient(135deg,
    var(--vp-c-brand-soft) 0%,
    rgba(var(--vp-c-brand-rgb), 0.1) 100%);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 1.5rem;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

:global(.dark) .path-icon-wrapper {
  background: rgba(var(--vp-c-brand-rgb), 0.1);
  border: 1px solid rgba(var(--vp-c-brand-rgb), 0.2);
}

.path-card:hover .path-icon-wrapper {
  transform: scale(1.1) rotate(-5deg);
}

.path-primary:hover .path-icon-wrapper {
  background: var(--vp-c-brand-1);
}

.path-primary:hover .path-icon {
  color: white;
}

.path-secondary:hover .path-icon-wrapper {
  background: var(--vp-c-brand-2);
}

.path-secondary:hover .path-icon {
  color: white;
}

.path-icon {
  width: 32px;
  height: 32px;
  color: var(--vp-c-brand-1);
  transition: color 0.3s ease;
}

.path-badge {
  position: absolute;
  top: 1.5rem;
  right: 1.5rem;
  padding: 0.4rem 0.9rem;
  border-radius: 20px;
  font-size: 0.75em;
  font-weight: 600;
  letter-spacing: 0.3px;
}

.badge-primary {
  background: linear-gradient(135deg, var(--vp-c-brand-1) 0%, var(--vp-c-brand-2) 100%);
  color: white;
  box-shadow: 0 2px 8px rgba(var(--vp-c-brand-rgb), 0.3);
}

.badge-secondary {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
}

:global(.dark) .badge-secondary {
  background: rgba(var(--vp-c-brand-rgb), 0.15);
  border: 1px solid rgba(var(--vp-c-brand-rgb), 0.3);
}

.badge-tertiary {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  border: 1px solid var(--vp-c-divider);
}

:global(.dark) .badge-tertiary {
  background: rgba(255, 255, 255, 0.05);
}

.path-card h3 {
  font-size: 1.5em;
  font-weight: 600;
  margin: 0 0 0.75rem 0;
  color: var(--vp-c-text-1);
  letter-spacing: -0.01em;
}

.path-description {
  font-size: 0.95em;
  color: var(--vp-c-text-2);
  line-height: 1.7;
  margin: 0 0 1.5rem 0;
  flex-grow: 1;
}

.path-duration {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
  font-size: 0.95em;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.clock-icon {
  width: 18px;
  height: 18px;
  color: var(--vp-c-brand-1);
}

.path-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  width: 100%;
  padding: 0.85rem 1.5rem;
  border-radius: 10px;
  font-weight: 600;
  font-size: 0.95em;
  text-decoration: none;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.path-primary .path-button {
  background: var(--vp-c-brand-1);
  color: white;
}

.path-primary .path-button:hover {
  background: var(--vp-c-brand-2);
  transform: translateX(4px);
  box-shadow: 0 4px 15px rgba(var(--vp-c-brand-rgb), 0.3);
}

.path-secondary .path-button {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  border: 1px solid transparent;
}

:global(.dark) .path-secondary .path-button {
  background: rgba(var(--vp-c-brand-rgb), 0.15);
}

.path-secondary .path-button:hover {
  background: var(--vp-c-brand-1);
  color: white;
  border-color: var(--vp-c-brand-1);
  transform: translateX(4px);
}

.path-tertiary .path-button {
  background: transparent;
  color: var(--vp-c-text-1);
  border: 1px solid var(--vp-c-divider);
}

.path-tertiary .path-button:hover {
  background: var(--vp-c-bg-soft);
  border-color: var(--vp-c-text-2);
  transform: translateX(4px);
}

:global(.dark) .path-tertiary .path-button:hover {
  background: rgba(255, 255, 255, 0.05);
}

.button-arrow {
  width: 18px;
  height: 18px;
  transition: transform 0.3s ease;
}

.path-button:hover .button-arrow {
  transform: translateX(4px);
}

.card-glow {
  position: absolute;
  inset: -50%;
  background: radial-gradient(circle, var(--vp-c-brand-1) 0%, transparent 70%);
  opacity: 0;
  transition: opacity 0.4s ease;
  pointer-events: none;
}

.path-card:hover .card-glow {
  opacity: 0.03;
}

@media (max-width: 768px) {
  .learning-path-section {
    padding: 3rem 0;
  }

  .path-header h2 {
    font-size: 1.8em;
  }

  .paths-grid {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }

  .path-card {
    padding: 2rem 1.5rem;
  }
}
</style>
