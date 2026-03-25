<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue';
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import { ArcballControls } from 'three/addons/controls/ArcballControls.js';

const containerRef = ref(null);
const isVRMode = ref(false); 
const isAutoRotate = ref(false);
const isLoading = ref(false);
const isSecureContext = ref(false);
let viewer = null;

// --- 1. 状态管理 ---
const globalUniforms = {
  uTime: { value: 0 },
  uCenter: { value: new THREE.Vector3(0, 0, 0) },
  uGeoRadius: { value: 0 },   // 控制"灰色小点"扩散
  uColorRadius: { value: 0 }, // 控制"彩色大球"恢复
  uMaxRadius: { value: 50 },
};

const animationState = {
  isLoaded: false,
  startTime: 0,
  isFinished: false 
};

// --- 2. Shader 注入逻辑 ---
const applyAdvancedShader = (mesh) => {
  if (!mesh || !mesh.material) return;
  const material = mesh.material;
  
  material.uniforms = material.uniforms || {};
  material.uniforms.uGeoRadius = globalUniforms.uGeoRadius;
  material.uniforms.uColorRadius = globalUniforms.uColorRadius;
  material.uniforms.uMaxRadius = globalUniforms.uMaxRadius;
  material.uniforms.uCenter = globalUniforms.uCenter;

  // A. Vertex Shader (仅计算世界坐标)
  material.vertexShader = `
    varying vec3 vWorldPosition;
  ` + material.vertexShader;

  const vsEndIndex = material.vertexShader.lastIndexOf('}');
  if (vsEndIndex !== -1) {
    const vsLogic = `
      vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz; 
    `;
    material.vertexShader = material.vertexShader.substring(0, vsEndIndex) + vsLogic + '}';
  }

  // B. Fragment Shader (核心修改：小点 -> 大球)
  const commonFragment = `
    uniform float uGeoRadius;
    uniform float uColorRadius;
    uniform float uMaxRadius;
    uniform vec3 uCenter;
    varying vec3 vWorldPosition;
  `;

  material.fragmentShader = commonFragment + material.fragmentShader;

  const fsEndIndex = material.fragmentShader.lastIndexOf('}');
  if (fsEndIndex !== -1) {
    const originalContent = material.fragmentShader.substring(0, fsEndIndex);

    const visualLogic = `
      // --- 视觉逻辑 ---
      float distFromCenter = distance(vWorldPosition, uCenter);
      
      // 1. 未扩散区域：完全隐藏
      if (distFromCenter > uGeoRadius) {
          discard;
      }

      // 2. 灰色小点阶段 (处于 几何边界 和 上色边界 之间)
      if (distFromCenter > uColorRadius) {
          
          // --- 关键修改：制造"小点" ---
          // Gaussian Splat 本质是中心不透明、边缘透明的椭球。
          // 这里我们把 alpha < 0.8 的部分全部切掉，只剩中心一个极小的硬核。
          if (gl_FragColor.a < 0.8) discard; 
          
          // 强制不透明，看起来像实心粒子
          gl_FragColor.a = 1.0; 
          
          // 强制纯灰色
          gl_FragColor.rgb = vec3(0.6, 0.6, 0.6);
      } 
      else {
          // 3. 彩色大球阶段 (上色波浪经过后)
          // 恢复原始逻辑：不 discard，显示原本的大椭球和半透明边缘
          // 这里的颜色就是原本的模型颜色
      }
    `;

    material.fragmentShader = originalContent + visualLogic + '}';
  }
  material.needsUpdate = true;
};

// --- 3. 配置与初始化 ---
const getViewerConfig = () => {
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  return {
    'rootElement': containerRef.value,
    'cameraUp': [0, 1, 0],
    'initialCameraPosition': [0, 0, 5],
    'initialCameraLookAt': [0, 0, 0],
    'useBuiltInControls': false,
    'gpuAcceleratedSort': false, 
    'webXRMode': isSecureContext.value ? GaussianSplats3D.WebXRMode.VR : GaussianSplats3D.WebXRMode.None,
    'sharedMemoryForWorkers': false,
    'integerBasedSort': true,
    'enableSIMDInSort': !isMobile,
    'splatAlphaRemovalThreshold': 5,
    'antialiased': !isMobile,
  };
};

const initViewer = async () => {
  if (isLoading.value) return;
  isLoading.value = true;

  try {
    if (viewer) {
      viewer.renderer.setAnimationLoop(null);
      if(viewer.dispose) await viewer.dispose();
      viewer = null;
    }
    if (containerRef.value) containerRef.value.innerHTML = '';
    
    // 重置状态
    animationState.isLoaded = false;
    animationState.isFinished = false;
    globalUniforms.uGeoRadius.value = 0;
    globalUniforms.uColorRadius.value = 0;

    const config = getViewerConfig();
    viewer = new GaussianSplats3D.Viewer(config);
    window.viewer = viewer;

    await viewer.addSplatScene('/models/point_cloud_cleaned.ply', {
      'showLoadingUI': true,
      'progressiveLoad': false,
      'rotation': [1, 0, 0, 0],
    });
    
    const splatMesh = viewer.getSplatMesh();
    setTimeout(() => { if (splatMesh) applyAdvancedShader(splatMesh); }, 100);

    animationState.startTime = Date.now();
    animationState.isLoaded = true;
    
    // --- 4. 动画循环 ---
    viewer.renderer.setAnimationLoop(() => {
      if (animationState.isLoaded && !animationState.isFinished) {
          const now = Date.now();
          const time = (now - animationState.startTime) / 1000;
          globalUniforms.uTime.value = time;
          
          const maxR = globalUniforms.uMaxRadius.value;
          
          // --- 速度设置 (秒) ---
          const geoDuration = 15.0;   // 灰色小点扩散时长
          const colorDuration = 10.0; // 彩色大球恢复时长
          const colorDelay = 1.0;     // 颜色延迟启动
          
          const geoStep = maxR / (geoDuration * 60.0);
          const colorStep = maxR / (colorDuration * 60.0);
          
          // 更新几何半径
          if (globalUniforms.uGeoRadius.value < maxR + 5.0) {
              globalUniforms.uGeoRadius.value += geoStep;
          }
          
          // 更新颜色半径
          if (time > colorDelay && globalUniforms.uColorRadius.value < maxR + 5.0) {
              globalUniforms.uColorRadius.value += colorStep;
          }
          
          // 结束检测
          if (globalUniforms.uGeoRadius.value >= maxR + 5.0 && 
              globalUniforms.uColorRadius.value >= maxR + 5.0) {
              
              animationState.isFinished = true;
              globalUniforms.uGeoRadius.value = 99999.0;
              globalUniforms.uColorRadius.value = 99999.0;
          }
      }
      
      viewer.update();
      viewer.render();
    });

    setupDesktopControls();
    adjustControlsToModel();

  } catch (error) {
    console.error("error:", error);
  } finally {
    isLoading.value = false;
  }
};

const setupDesktopControls = () => {
  if (!viewer) return;
  if (viewer.controls) { viewer.controls.dispose(); viewer.controls = null; }
  const controls = new ArcballControls(viewer.camera, viewer.renderer.domElement, viewer.threeScene);
  controls.setGizmosVisible(false);
  controls.enableDamping = true;
  viewer.controls = controls;
};

const adjustControlsToModel = () => {
  if (isVRMode.value) return;
  const mesh = viewer.getSplatMesh();
  setTimeout(() => {
    if (mesh.getSplatCount() > 0) {
      mesh.updateMatrixWorld();
      
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      const tempVec = new THREE.Vector3();
      const splatCount = mesh.getSplatCount();
      const sampleCount = Math.min(splatCount, 1000);
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
      const distance = maxDim * 2;
      
      globalUniforms.uCenter.value.copy(worldCenter);
      globalUniforms.uMaxRadius.value = maxDim * 0.7; 
      
      // 重置动画
      globalUniforms.uGeoRadius.value = 0.0;
      globalUniforms.uColorRadius.value = 0.0;
      animationState.startTime = Date.now();
      animationState.isFinished = false;

      viewer.camera.position.set(worldCenter.x, worldCenter.y, worldCenter.z + distance);
      viewer.camera.lookAt(worldCenter);
    }
  }, 100);
};

// ... VR 和其他代码保持不变 ...
const onSessionStarted = (session) => {
  isVRMode.value = true;
  if (viewer && viewer.controls) { viewer.controls.dispose(); viewer.controls = null; }
  session.addEventListener('end', onSessionEnded);
};
const onSessionEnded = () => { isVRMode.value = false; setupDesktopControls(); };
const toggleVRMode = async () => { 
  if (!isSecureContext.value) { alert("需HTTPS"); return; }
  if (isVRMode.value) { if(viewer.xr) viewer.xr.exitVR(); isVRMode.value = false; }
  else { if(viewer.xr) viewer.xr.enterVR(); isVRMode.value = true; }
};
const toggleAutoRotate = () => { isAutoRotate.value = !isAutoRotate.value; };
const checkProtocol = () => { 
  const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  const isHttps = window.location.protocol === 'https:';
  isSecureContext.value = isLocal || isHttps;
};

onMounted(() => { if (containerRef.value) { checkProtocol(); initViewer(); } });
onBeforeUnmount(async () => {
  if (viewer) {
      viewer.renderer.setAnimationLoop(null);
      await viewer.dispose();
  }
});
</script>

<template>
  <div class="app-container">
    <div ref="containerRef" class="viewer-container"></div>
    <div v-if="isLoading" class="loading-overlay">正在处理...</div>
    <div class="controls-ui">
      <button v-if="isSecureContext" @click="toggleVRMode" :class="{ active: isVRMode }">
        {{ isVRMode ? '退出 VR' : '进入 VR' }}
      </button>
      <button @click="toggleAutoRotate" :class="{ active: isAutoRotate }">
        {{ isAutoRotate ? '停止旋转' : '自动旋转' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.app-container { position: relative; width: 100vw; height: 100vh; background-color: #000000; }
.viewer-container { width: 100%; height: 100%; }
.controls-ui { position: absolute; top: 30px; left: 50%; transform: translateX(-50%); display: flex; gap: 15px; z-index: 100; }
.loading-overlay { position: absolute; inset: 0; background: rgba(0,0,0,0.8); color: white; display: flex; justify-content: center; align-items: center; z-index: 200; font-size: 20px; }
button { background: rgba(0,0,0,0.6); color: white; border: 1px solid rgba(255,255,255,0.3); padding: 10px 20px; border-radius: 20px; cursor: pointer; transition: 0.3s; }
button.active { background: #22c55e; border-color: #22c55e; }
</style>