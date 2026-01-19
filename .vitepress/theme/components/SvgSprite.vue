<template>
  <div class="svg-sprite" aria-hidden="true" v-html="sprite"></div>
</template>

<script setup lang="ts">
import sprite from '../assets/icons-sprite.svg?raw'
</script>

<style scoped>
.svg-sprite {
  position: absolute;
  width: 0;
  height: 0;
  overflow: hidden;
}
</style>
