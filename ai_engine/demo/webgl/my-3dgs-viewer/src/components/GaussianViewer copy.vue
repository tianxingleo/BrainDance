<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue';
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import { ArcballControls } from 'three/addons/controls/ArcballControls.js';

const containerRef = ref(null);
const isVRMode = ref(false); 
const isAutoRotate = ref(false);
const isLoading = ref(false);
const isSecureContext = ref(false); // 新增：用于判断是否支持 VR
let viewer = null;

// 1. 配置生成器
const getViewerConfig = () => {
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  
  return {
    'rootElement': containerRef.value,
    'cameraUp': [0, 1, 0],
    'initialCameraPosition': [0, 0, 5],
    'initialCameraLookAt': [0, 0, 0],
    'useBuiltInControls': false,
    
    // 始终关闭 GPU 排序，防止兼容性问题
    'gpuAcceleratedSort': false, 
    
    // VR 开关：只要是安全上下文，就提前准备好 VR 渲染器
    'webXRMode': isSecureContext.value ? GaussianSplats3D.WebXRMode.VR : GaussianSplats3D.WebXRMode.None,

    'sharedMemoryForWorkers': false,
    'integerBasedSort': true,
    'enableSIMDInSort': !isMobile,
    'splatAlphaRemovalThreshold': 5,
    'antialiased': !isMobile,
  };
};

// 2. 初始化核心逻辑 (只调用一次)
const initViewer = async () => {
  if (isLoading.value) return;
  isLoading.value = true;

  try {
    if (viewer) {
      viewer.dispose();
      viewer = null;
    }
    if (containerRef.value) containerRef.value.innerHTML = '';

    const config = getViewerConfig();
    viewer = new GaussianSplats3D.Viewer(config);
    window.viewer = viewer;
    window.THREE = THREE;

    await viewer.addSplatScene('/models/scene.splat', {
      'showLoadingUI': true,
      'progressiveLoad': false,
      'rotation': [1, 0, 0, 0],
    });
    
    viewer.renderer.setAnimationLoop(() => {
      viewer.update();
      viewer.render();
    });

    setupDesktopControls();
    adjustControlsToModel();

  } catch (error) {
    console.error("初始化异常:", error);
  } finally {
    isLoading.value = false;
  }
};

// 2.1. 桌面控制器逻辑
const setupDesktopControls = () => {
  if (!viewer) return;
  if (viewer.controls) {
    viewer.controls.dispose();
    viewer.controls = null;
  }

  const controls = new ArcballControls(viewer.camera, viewer.renderer.domElement, viewer.threeScene);
  controls.setGizmosVisible(false);
  controls.cursorZoom = true;
  controls.adjustNearFar = true;
  controls.enableDamping = true;
  controls.dampingFactor = 10;
  controls.wMax = 10;
  controls.radiusFactor = 1.2;

  viewer.controls = controls;
};

// 3. 对焦辅助函数
const adjustControlsToModel = () => {
  if (isVRMode.value) return;
  const mesh = viewer.getSplatMesh();
  setTimeout(() => {
    if (mesh.getSplatCount() > 0) {
      mesh.updateMatrixWorld();
      // ... (此处保持原有计算 BoundingBox 逻辑不变)
      // 省略中间计算代码，与原文件一致 ...
      
      // 这里的逻辑不需要动，为了节省篇幅我简化了显示，
      // 实际使用时请保留你原文件中 minX/maxX 的计算循环
      
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      const tempVec = new THREE.Vector3();
      const splatCount = mesh.getSplatCount();
      const sampleCount = Math.min(splatCount, 10000);
      const step = Math.max(1, Math.floor(splatCount / sampleCount));

      for (let i = 0; i < splatCount; i += step) {
        mesh.getSplatCenter(i, tempVec);
        if (tempVec.x < minX) minX = tempVec.x; if (tempVec.x > maxX) maxX = tempVec.x;
        if (tempVec.y < minY) minY = tempVec.y; if (tempVec.y > maxY) maxY = tempVec.y;
        if (tempVec.z < minZ) minZ = tempVec.z; if (tempVec.z > maxZ) maxZ = tempVec.z;
      }

      const localCenter = new THREE.Vector3((minX + maxX)/2, (minY + maxY)/2, (minZ + maxZ)/2);
      const worldCenter = localCenter.clone().applyMatrix4(mesh.matrixWorld);

      if (viewer.controls) {
        viewer.controls.target.copy(worldCenter);
        viewer.controls.update();
      }

      const maxDim = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
      const distance = maxDim * 3;
      viewer.camera.position.set(worldCenter.x, worldCenter.y, worldCenter.z + distance);
      viewer.camera.lookAt(worldCenter);
      
      viewer.forceRenderNextFrame();
    }
  }, 100);
};

// 4. VR 会话管理
const onSessionStarted = (session) => {
  isVRMode.value = true;
  if (viewer && viewer.controls) {
    viewer.controls.dispose();
    viewer.controls = null;
  }
  session.addEventListener('end', onSessionEnded);
};

const onSessionEnded = () => {
  isVRMode.value = false;
  setupDesktopControls();
};

const toggleVRMode = async () => {
  if (!isSecureContext.value) {
    alert("VR 模式需要 HTTPS 环境或本地 localhost");
    return;
  }
  if (!viewer || !viewer.renderer) return;

  if (isVRMode.value) {
    const session = viewer.renderer.xr.getSession();
    if (session) await session.end();
    return;
  }

  try {
    const session = await navigator.xr.requestSession('immersive-vr', {
      optionalFeatures: ['local-floor', 'bounded-floor']
    });
    viewer.renderer.xr.setSession(session);
    onSessionStarted(session);
  } catch (e) {
    console.error("无法进入 VR:", e);
    if (e.name === 'NotSupportedError') {
      alert("未检测到 VR 设备或浏览器不支持 WebXR");
    } else {
      alert("无法进入 VR: " + e.message);
    }
  }
};

const toggleAutoRotate = () => {
  isAutoRotate.value = !isAutoRotate.value;
  // ArcballControls 没有 autoRotate，这里仅做 UI 状态切换
};

// 检查协议
const checkProtocol = () => {
  // localhost, 127.0.0.1 也是安全上下文，允许 VR
  const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  const isHttps = window.location.protocol === 'https:';
  isSecureContext.value = isLocal || isHttps;
};

// 生命周期
onMounted(() => {
  if (!containerRef.value) return;
  checkProtocol();
  initViewer();
});

onBeforeUnmount(async () => {
  if (viewer) await viewer.dispose();
});
</script>

<template>
  <div class="app-container">
    <div ref="containerRef" class="viewer-container"></div>

    <div v-if="isLoading" class="loading-overlay">
      正在处理...
    </div>

    <div class="controls-ui">
      <button 
        v-if="isSecureContext" 
        @click="toggleVRMode" 
        :class="{ active: isVRMode }" 
        :disabled="isLoading"
      >
        {{ isVRMode ? '退出 VR' : '进入 VR' }}
      </button>

      <div v-else class="https-warning">
        VR不可用 (需HTTPS)
      </div>
      
      <button @click="toggleAutoRotate" :class="{ active: isAutoRotate }" :disabled="isLoading">
        {{ isAutoRotate ? '停止旋转' : '自动旋转' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.app-container {
  position: relative;
  width: 100vw;
  height: 100vh;
  background-color: #333;
}
.viewer-container {
  width: 100%;
  height: 100%;
}
.controls-ui {
  position: absolute;
  top: 30px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 15px;
  z-index: 100;
  align-items: center; /* 保证文字和按钮对齐 */
}
.loading-overlay {
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.7);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 200;
  font-size: 18px;
}
.https-warning {
  color: rgba(255, 255, 255, 0.7);
  font-size: 12px;
  background: rgba(0, 0, 0, 0.5);
  padding: 8px 12px;
  border-radius: 20px;
  border: 1px solid rgba(255, 100, 100, 0.3);
}
button {
  background: rgba(0, 0, 0, 0.6);
  color: white;
  border: 1px solid rgba(255, 255, 255, 0.3);
  padding: 10px 20px;
  border-radius: 20px;
  font-size: 14px;
  backdrop-filter: blur(5px);
  cursor: pointer;
  transition: all 0.3s;
}
button:active { transform: scale(0.95); }
button.active {
  background: rgba(34, 197, 94, 0.8);
  border-color: rgba(34, 197, 94, 1);
  font-weight: bold;
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>