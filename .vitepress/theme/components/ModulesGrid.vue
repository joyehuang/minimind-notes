<template>
  <section class="modules-section">
    <div class="modules-container">
      <div class="modules-header">
        <div class="header-badge">{{ copy.badge }}</div>
        <h2>{{ copy.title }}</h2>
        <p>{{ copy.description }}</p>
      </div>

      <!-- Tier 1 -->
      <div class="tier-section">
        <div class="tier-header">
          <div class="tier-badge tier-1">
            <Icon name="grid" class-name="tier-icon" />
            {{ copy.tier1Label }}
          </div>
          <p class="tier-description">{{ copy.tier1Description }}</p>
        </div>

        <div class="module-cards">
          <a
            v-for="module in tier1Modules"
            :key="module.id"
            :href="module.link"
            class="module-card"
          >
            <div class="module-icon-wrapper">
              <Icon :name="module.icon" class-name="module-icon" />
            </div>

            <div class="module-content">
              <div class="module-header">
                <span class="module-id">{{ module.id }}</span>
                <span class="module-status" :class="`status-${module.status}`">
                  <Icon :name="getStatusIcon(module.status)" class-name="status-icon" />
                  {{ getStatusText(module.status) }}
                </span>
              </div>

              <h4>{{ module.title }}</h4>
              <p class="module-question">{{ module.question }}</p>

              <div v-if="module.experiments" class="module-meta">
                <div class="meta-item">
                  <Icon name="tool" class-name="meta-icon" />
                  <span>{{ module.experiments }} {{ module.experiments === 1 ? copy.experimentLabel : copy.experimentsLabel }}</span>
                </div>
              </div>
            </div>

            <div class="module-arrow">
              <Icon name="chevron-right" class-name="arrow-icon" />
            </div>
          </a>
        </div>
      </div>

      <!-- Tier 2 -->
      <div class="tier-section">
        <div class="tier-header">
          <div class="tier-badge tier-2">
            <Icon name="layers-alt" class-name="tier-icon" />
            {{ copy.tier2Label }}
          </div>
          <p class="tier-description">{{ copy.tier2Description }}</p>
        </div>

        <div class="module-cards">
          <a
            v-for="module in tier2Modules"
            :key="module.id"
            :href="module.link"
            class="module-card"
            :class="{ 'module-locked': module.status === 'planned' }"
          >
            <div class="module-icon-wrapper">
              <Icon :name="module.icon" class-name="module-icon" />
            </div>

            <div class="module-content">
              <div class="module-header">
                <span class="module-id">{{ module.id }}</span>
                <span class="module-status" :class="`status-${module.status}`">
                  <Icon :name="getStatusIcon(module.status)" class-name="status-icon" />
                  {{ getStatusText(module.status) }}
                </span>
              </div>

              <h4>{{ module.title }}</h4>
              <p class="module-question">{{ module.question }}</p>
            </div>

            <div class="module-arrow">
              <Icon name="chevron-right" class-name="arrow-icon" />
            </div>
          </a>
        </div>
      </div>

      <!-- Legend -->
      <div class="status-legend">
        <div class="legend-item">
          <span class="legend-dot status-complete"></span>
          <span>{{ copy.legendComplete }}</span>
        </div>
        <div class="legend-item">
          <span class="legend-dot status-partial"></span>
          <span>{{ copy.legendPartial }}</span>
        </div>
        <div class="legend-item">
          <span class="legend-dot status-planned"></span>
          <span>{{ copy.legendPlanned }}</span>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useLocale } from '../i18n'
import { MODULES_COPY, createTier1Modules, createTier2Modules, STATUS_TEXT } from '../constants/modules'
import Icon from './Icon.vue'

const { isEn, localePath } = useLocale()

// Use pre-defined translation constants
const copy = computed(() => isEn.value ? MODULES_COPY.en : MODULES_COPY.zh)

// Module data with locale-aware links
const tier1Modules = computed(() => createTier1Modules(localePath.value, isEn.value))
const tier2Modules = computed(() => createTier2Modules(localePath.value, isEn.value))

function getStatusText(status: string): string {
  const statusMap = isEn.value ? STATUS_TEXT.en : STATUS_TEXT.zh
  return statusMap[status] || status
}

function getStatusIcon(status: string): string {
  const iconMap: Record<string, string> = {
    complete: 'check',
    partial: 'clock',
    planned: 'lock'
  }
  return iconMap[status] || 'clock'
}
</script>

<style scoped>
.modules-section {
  padding: 5rem 0;
  background: var(--vp-c-bg-soft);
  border-top: 1px solid var(--vp-c-divider);
  border-bottom: 1px solid var(--vp-c-divider);
}

:global(.dark) .modules-section {
  background: rgba(var(--vp-c-brand-rgb), 0.02);
  border-top: 1px solid var(--vp-c-divider);
  border-bottom: 1px solid var(--vp-c-divider);
}

.modules-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1.5rem;
}

.modules-header {
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

.modules-header h2 {
  font-size: 2.3em;
  font-weight: 700;
  margin: 0 0 1rem 0;
  color: var(--vp-c-text-1);
  letter-spacing: -0.02em;
}

.modules-header p {
  font-size: 1.1em;
  color: var(--vp-c-text-2);
  margin: 0;
  line-height: 1.7;
}

.tier-section {
  margin-bottom: 4rem;
}

.tier-section:last-of-type {
  margin-bottom: 2rem;
}

.tier-header {
  margin-bottom: 2rem;
}

.tier-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 1.2rem;
  border-radius: 12px;
  font-size: 0.9em;
  font-weight: 600;
  margin-bottom: 0.75rem;
}

.tier-icon {
  width: 18px;
  height: 18px;
}

.tier-1 {
  background: linear-gradient(135deg, var(--vp-c-brand-1) 0%, var(--vp-c-brand-2) 100%);
  color: white;
  box-shadow: 0 2px 8px rgba(var(--vp-c-brand-rgb), 0.3);
}

.tier-2 {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
}

:global(.dark) .tier-2 {
  background: rgba(var(--vp-c-brand-rgb), 0.15);
  border: 1px solid rgba(var(--vp-c-brand-rgb), 0.2);
}

.tier-description {
  color: var(--vp-c-text-2);
  font-size: 1em;
  margin: 0;
}

.module-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
}

.module-card {
  position: relative;
  display: flex;
  align-items: center;
  gap: 1rem;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  padding: 1.25rem;
  text-decoration: none;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

:global(.dark) .module-card {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.module-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
}

:global(.dark) .module-card:hover {
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
}

.module-card:hover .module-icon-wrapper {
  background: var(--vp-c-brand-1);
  transform: scale(1.05);
}

.module-card:hover .module-icon {
  color: white;
}

.module-card:hover .module-arrow {
  transform: translateX(4px);
  color: var(--vp-c-brand-1);
}

.module-icon-wrapper {
  flex-shrink: 0;
  width: 48px;
  height: 48px;
  background: var(--vp-c-brand-soft);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

:global(.dark) .module-icon-wrapper {
  background: rgba(var(--vp-c-brand-rgb), 0.1);
}

.module-icon {
  width: 24px;
  height: 24px;
  color: var(--vp-c-brand-1);
  transition: color 0.3s ease;
}

.module-content {
  flex: 1;
  min-width: 0;
}

.module-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  gap: 0.5rem;
}

.module-id {
  font-family: monospace;
  font-size: 0.75em;
  color: var(--vp-c-text-3);
  font-weight: 500;
}

.module-status {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.75em;
  font-weight: 600;
  padding: 0.25rem 0.6rem;
  border-radius: 6px;
  white-space: nowrap;
}

.status-icon {
  width: 12px;
  height: 12px;
}

.status-complete {
  background: rgba(16, 185, 129, 0.1);
  color: #10b981;
}

:global(.dark) .status-complete {
  background: rgba(16, 185, 129, 0.15);
}

.status-partial {
  background: rgba(245, 158, 11, 0.1);
  color: #f59e0b;
}

:global(.dark) .status-partial {
  background: rgba(245, 158, 11, 0.15);
}

.status-planned {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
}

:global(.dark) .status-planned {
  background: rgba(255, 255, 255, 0.05);
}

.module-card h4 {
  font-size: 1em;
  font-weight: 600;
  margin: 0 0 0.35rem 0;
  color: var(--vp-c-text-1);
}

.module-card:hover h4 {
  color: var(--vp-c-brand-1);
}

.module-question {
  font-size: 0.875em;
  color: var(--vp-c-text-2);
  line-height: 1.5;
  margin: 0;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.module-meta {
  display: flex;
  gap: 0.75rem;
  margin-top: 0.75rem;
}

.meta-item {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.8em;
  color: var(--vp-c-text-2);
  font-weight: 500;
}

.meta-icon {
  width: 14px;
  height: 14px;
  color: var(--vp-c-brand-1);
}

.module-arrow {
  flex-shrink: 0;
  color: var(--vp-c-text-3);
  transition: all 0.3s ease;
}

.module-arrow svg {
  width: 20px;
  height: 20px;
}

.status-legend {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 2rem;
  padding-top: 2rem;
  border-top: 1px solid var(--vp-c-divider);
  font-size: 0.9em;
  color: var(--vp-c-text-2);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}

.legend-dot.status-complete {
  background: #10b981;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
}

.legend-dot.status-partial {
  background: #f59e0b;
  box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.2);
}

.legend-dot.status-planned {
  background: var(--vp-c-text-3);
}

@media (max-width: 768px) {
  .modules-section {
    padding: 3rem 0;
  }

  .modules-header h2 {
    font-size: 1.8em;
  }

  .tier-badge {
    font-size: 0.85em;
    padding: 0.5rem 1rem;
  }

  .module-cards {
    grid-template-columns: 1fr;
  }

  .status-legend {
    flex-direction: column;
    gap: 0.75rem;
    align-items: flex-start;
  }
}
</style>
