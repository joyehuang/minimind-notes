<template>
  <Layout>
    <template #doc-before>
      <Breadcrumbs />
    </template>
    <template #doc-after>
      <GitHubFooter />
    </template>
  </Layout>
</template>

<script setup lang="ts">
import { onMounted, watch } from 'vue'
import { useData } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import GitHubFooter from './components/GitHubFooter.vue'
import Breadcrumbs from './components/Breadcrumbs.vue'
import { useBreadcrumbSchema } from './composables/useBreadcrumbSchema'

const { Layout } = DefaultTheme
const { page } = useData()
const { breadcrumbSchema } = useBreadcrumbSchema()

// 动态注入 BreadcrumbList 结构化数据到 head
const injectBreadcrumbSchema = () => {
  // 移除旧的 breadcrumb schema
  const existingSchema = document.querySelector('script[data-schema="breadcrumb"]')
  if (existingSchema) {
    existingSchema.remove()
  }

  // 如果有有效的 schema，注入新的
  if (breadcrumbSchema.value) {
    const script = document.createElement('script')
    script.type = 'application/ld+json'
    script.setAttribute('data-schema', 'breadcrumb')
    script.textContent = JSON.stringify(breadcrumbSchema.value)
    document.head.appendChild(script)
  }
}

// 初始挂载时注入
onMounted(() => {
  injectBreadcrumbSchema()
})

// 监听路由变化，更新 schema
watch(() => page.value.relativePath, () => {
  injectBreadcrumbSchema()
})
</script>