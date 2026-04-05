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
import { onMounted, onUnmounted } from 'vue'

const { Layout } = DefaultTheme

/**
 * Fix language switcher order: always 简体中文 first, English second.
 * VitePress puts the current locale first by default; we reorder the
 * menu items so 简体中文 (linking to "/") is always on top.
 */
function fixLangMenuOrder() {
  const groups = document.querySelectorAll(
    '.VPNavBarTranslations .VPMenuGroup, .VPNavScreenTranslations .VPMenuGroup'
  )
  groups.forEach((group) => {
    const items = Array.from(group.querySelectorAll(':scope > .VPMenuLink'))
    if (items.length < 2) return
    const zhItem = items.find((el) => {
      const a = el.querySelector('a')
      return a && !a.getAttribute('href')?.startsWith('/en')
    })
    if (zhItem && zhItem !== items[0]) {
      group.prepend(zhItem)
    }
  })
}

let observer: MutationObserver | null = null

onMounted(() => {
  // The flyout menu is lazily rendered on hover/click, so we watch only
  // the nav bar containers for new children (lightweight).
  const navBar = document.querySelector('.VPNavBar')
  if (navBar) {
    observer = new MutationObserver(fixLangMenuOrder)
    observer.observe(navBar, { childList: true, subtree: true })
  }
  fixLangMenuOrder()
})

onUnmounted(() => {
  observer?.disconnect()
})
</script>
