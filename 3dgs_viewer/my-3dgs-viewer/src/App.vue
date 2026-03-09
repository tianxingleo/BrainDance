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
/* 3. 这里写全局样式 */
/* 这一步很重要！一定要把浏览器默认的边距去掉，否则屏幕四周会有白边 */
body, html, #app {
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  overflow: hidden; /* 防止出现滚动条 */
}
</style>