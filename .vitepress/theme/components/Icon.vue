<template>
  <svg :class="className" aria-hidden="true">
    <use :href="`${withBase('/icons-sprite.svg')}#icon-${name}`" />
  </svg>
</template>

<script setup lang="ts">
import { withBase } from 'vitepress'

interface Props {
  name: string
  className?: string
}

withDefaults(defineProps<Props>(), {
  className: ''
})
</script>

<style scoped>
svg {
  display: inline-block;
  fill: currentColor;
  stroke: currentColor;
}
</style>
