<script setup lang="ts">
import { defineAsyncComponent } from 'vue'

const SparkGaussianViewer = defineAsyncComponent(
  () => import('./components/SparkGaussianViewer.vue'),
)
const MarkerARViewer = defineAsyncComponent(
  () => import('./components/MarkerARViewer.vue'),
)

const params = new URLSearchParams(window.location.search)
const mode = params.get('mode') || 'viewer'

window.BrainDanceChannel?.postMessage?.(JSON.stringify({
  status: 'info',
  msg: 'Spark App mode resolved',
  mode,
  href: window.location.href,
}))
</script>

<template>
  <main>
    <MarkerARViewer v-if="mode === 'marker-ar'" />
    <SparkGaussianViewer v-else />
  </main>
</template>

<style>
body, html, #app {
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
}
</style>
