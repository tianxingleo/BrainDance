<script setup>
// 1. 引入刚才写的查看器组件
import GaussianViewer from './components/GaussianViewer.vue'
</script>

<template>
  <main>
    <GaussianViewer />
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
