<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue';
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

// 1. 定义容器 Ref
const containerRef = ref(null);
let viewer = null;

onMounted(() => {
  if (!containerRef.value) return;

  // 2. 初始化 Viewer (相当于启动 Three.js 的引擎)
  viewer = new GaussianSplats3D.Viewer({
    'rootElement': containerRef.value,
    'rotation': [1, 0, 0, 0],
    'cameraUp': [0, 1, 0],         // Y轴朝上 (标准3D习惯)
    'initialCameraPosition': [0, 0, 5], // 放在物体正前方5米处
    'initialCameraLookAt': [0, 0, 0],   // 盯着中心看

    'optimizeSplatData': false,


    'sharedMemoryForWorkers': false,
    'enableSIMDInSort': false,
    'logLevel': 1,
    'webGLRendererParameters': {
        'antialias': false
    }
  });

  window.THREE = THREE;
  window.viewer = viewer;

  // 3. 加载模型
  // 注意：模型文件要放在 public 文件夹下，或者填写远程 URL
  viewer.addSplatScene('./models/scene.ply', {
    'showLoadingUI': true,
    'splatAlphaRemovalThreshold': 0,
    'progressiveLoad': false
  })
  .then(() => {
    viewer.start();
    console.log("✅ 模型加载成功！");
  })
  .catch(err => {
    console.error("❌ 加载报错:", err);
  });
  
});

// 4. 销毁防止内存泄漏 (非常重要！手机显存有限)
onBeforeUnmount(() => {
  if (viewer) {
    viewer.dispose();
    viewer = null;
  }
});
</script>

<template>
  <div ref="containerRef" class="viewer-container"></div>
</template>

<style scoped>
/* 👇 必须给容器设置宽高，否则 Canvas 也是 0x0 */
.viewer-container {
  width: 100vw;
  height: 100vh;
  background-color: #333; /* 先改成灰色背景，别用黑色 */
  position: absolute; /* 建议加上绝对定位 */
  top: 0;
  left: 0;
}
</style>