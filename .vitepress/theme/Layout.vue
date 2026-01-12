<template>
  <Layout>
    <template #doc-after>
      <GitHubFooter />
    </template>
  </Layout>
</template>

<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import GitHubFooter from './components/GitHubFooter.vue'

const { Layout } = DefaultTheme
</script>