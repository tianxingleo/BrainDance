<script setup>
import { onMounted, onBeforeUnmount, ref, shallowRef } from 'vue';
import * as THREE from 'three';
import { SplatMesh } from '@sparkjsdev/spark';
import { ArcballControls } from 'three/addons/controls/ArcballControls.js';

const containerRef = ref(null);
const isVRMode = ref(false);
const isAutoRotate = ref(false);
const isLoading = ref(false);
const isSecureContext = ref(false);

const renderer = shallowRef(null);
const scene = shallowRef(null);
const camera = shallowRef(null);
const controls = shallowRef(null);
let splatMesh = null;

// 2. 初始化核心逻辑 (只调用一次)
// 2. 初始化核心逻辑
const initViewer = async () => {
  if (isLoading.value) return;
  isLoading.value = true;

  try {
    // --- 清理 ---
    if (renderer.value) {
      renderer.value.dispose();
      renderer.value.forceContextLoss();
      renderer.value.domElement.remove();
      renderer.value = null;
    }
    if (containerRef.value) containerRef.value.innerHTML = '';

    // --- Three.js 基础 ---
    scene.value = new THREE.Scene();
    scene.value.background = new THREE.Color(0x202020); // 改为深灰色背景，防止黑色模型看不见

    const { clientWidth, clientHeight } = containerRef.value;
    camera.value = new THREE.PerspectiveCamera(70, clientWidth / clientHeight, 0.01, 1000);
    camera.value.position.set(0, 0, 5);

    renderer.value = new THREE.WebGLRenderer({ antialias: false });
    renderer.value.setSize(clientWidth, clientHeight);
    renderer.value.xr.enabled = true;
    containerRef.value.appendChild(renderer.value.domElement);

    // --- 🔴 关键修改：直接加载，开启调试模式 ---
    console.log('🚀 开始标准加载...');
    
    // 尝试 1：不指定 format，让 Spark 根据后缀自己猜
    // 如果失败，我们稍后修改这里强制指定 'ksplat' 或 'ply'
    splatMesh = new SplatMesh('/models/scene.splat', {
        alphaTest: 0.1,
        logLevel: 'debug' // 👈 开启 Spark 内部详细日志
    });

    scene.value.add(splatMesh);

    // 等待加载
    await splatMesh.ready;
    console.log('✅ 加载过程结束');

    // 🔴 诊断：打印整个对象，看看数据到底在哪
    console.log('📦 SplatMesh 对象详情:', splatMesh);

    // 尝试获取粒子数 (不同版本属性名可能不同)
    const count = splatMesh.splatCount || splatMesh.count || (splatMesh.geometry ? splatMesh.geometry.getAttribute('position').count : 0);
    console.log(`📊 粒子数检测: ${count}`);

    if (count > 0) {
        // 强制修正位置和缩放
        splatMesh.position.set(0, 0, 0);
        splatMesh.rotation.set(0, 0, 0);
        splatMesh.scale.set(1, 1, 1);
        splatMesh.frustumCulled = false;
        
        // 自动对焦
        const box = new THREE.Box3().setFromObject(splatMesh);
        const center = box.getCenter(new THREE.Vector3());
        console.log('📏 模型中心:', center);
        controls.value.target.copy(center);
        camera.value.lookAt(center);
    } else {
        console.warn('⚠️ 粒子数为 0，尝试缩放或检查格式...');
    }

    // --- 启动循环 ---
    renderer.value.setAnimationLoop(() => {
      if (controls.value) controls.value.update();
      if (renderer.value && scene.value && camera.value) {
        renderer.value.render(scene.value, camera.value);
      }
    });

    setupDesktopControls();
    adjustControlsToModel();
    window.addEventListener('resize', onWindowResize);

  } catch (error) {
    console.error("❌ 错误:", error);
  } finally {
    isLoading.value = false;
  }
};

// 2.1. 桌面控制器逻辑
const setupDesktopControls = () => {
  if (!renderer.value || !camera.value) return;
  if (controls.value) {
    controls.value.dispose();
    controls.value = null;
  }

  const _controls = new ArcballControls(camera.value, renderer.value.domElement, scene.value);
  _controls.setGizmosVisible(false);
  _controls.cursorZoom = true;
  _controls.adjustNearFar = true;
  _controls.enableDamping = true;
  _controls.dampingFactor = 10;
  _controls.wMax = 10;
  _controls.radiusFactor = 1.2;

  controls.value = _controls;
};

// 3. 对焦辅助函数
const adjustControlsToModel = () => {
  if (isVRMode.value || !splatMesh) return;
  setTimeout(() => {
    const box = new THREE.Box3().setFromObject(splatMesh);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    if (size.length() > 0 && size.length() < Infinity) {
      if (controls.value) {
        controls.value.target.copy(center);
        controls.value.update();
      }

      const maxDim = Math.max(size.x, size.y, size.z);
      const distance = maxDim * 2.0;
      if (camera.value) {
        camera.value.position.set(center.x, center.y, center.z + distance);
        camera.value.lookAt(center);
      }
    }
  }, 100);
};

// 4. VR 会话管理
const onSessionStarted = (session) => {
  isVRMode.value = true;
  if (controls.value) {
    controls.value.dispose();
    controls.value = null;
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
  if (!renderer.value) return;

  if (isVRMode.value) {
    const session = renderer.value.xr.getSession();
    if (session) await session.end();
    return;
  }

  try {
    const session = await navigator.xr.requestSession('immersive-vr', {
      optionalFeatures: ['local-floor', 'bounded-floor']
    });
    renderer.value.xr.setSession(session);
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

const onWindowResize = () => {
  if (camera.value && renderer.value && containerRef.value) {
    const { clientWidth, clientHeight } = containerRef.value;
    camera.value.aspect = clientWidth / clientHeight;
    camera.value.updateProjectionMatrix();
    renderer.value.setSize(clientWidth, clientHeight);
  }
};

onBeforeUnmount(() => {
  window.removeEventListener('resize', onWindowResize);
  if (renderer.value) {
    renderer.value.dispose();
    renderer.value.forceContextLoss();
  }
  if (splatMesh) {
    splatMesh.dispose();
    splatMesh = null;
  }
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