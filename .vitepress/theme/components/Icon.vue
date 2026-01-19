<template>
  <svg :class="className" aria-hidden="true">
    <use :href="spriteHref" :xlink:href="spriteHref" />
  </svg>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'

interface Props {
  name: string
  className?: string
}

const props = withDefaults(defineProps<Props>(), {
  className: ''
})

const spriteHref = computed(() => `${withBase('/icons-sprite.svg')}#icon-${props.name}`)
</script>

<style scoped>
svg {
  display: inline-block;
  fill: currentColor;
  stroke: currentColor;
}
</style>
