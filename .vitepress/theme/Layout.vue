<template>
  <Layout>
    <template #layout-top>
      <SvgSprite />
    </template>
    <template #doc-before>
      <Breadcrumbs />
    </template>
    <template #doc-after>
      <GitHubFooter />
    </template>
  </Layout>
</template>

<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import GitHubFooter from './components/GitHubFooter.vue'
import Breadcrumbs from './components/Breadcrumbs.vue'
import SvgSprite from './components/SvgSprite.vue'

const { Layout } = DefaultTheme

// Note: BreadcrumbList structured data is now server-side rendered
// via VitePress transformHead hook (no DOM manipulation needed)
</script>
