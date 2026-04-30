<script setup lang="ts">
import SparkGaussianViewer from './components/SparkGaussianViewer.vue'
import MarkerARViewer from './components/MarkerARViewer.vue'

const params = new URLSearchParams(window.location.search)
const mode = params.get('mode') || 'viewer'
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
