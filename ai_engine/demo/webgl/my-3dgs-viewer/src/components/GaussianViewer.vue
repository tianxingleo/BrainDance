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
let particleSystem = null;

// --- 1. 状态管理 ---
const PHASE = {
  FLY_IN: 0,    // 粒子从远处飞来
  DIFFUSION: 1, // 关键阶段：粒子渐隐 + 3DGS从中心扩散出现
  COLORING: 2,  // 3DGS 上色
  FINISHED: 3   
};

const animationState = {
  isLoaded: false,
  lastFrameTime: 0,
  phase: PHASE.FLY_IN, 
  
  // 时间配置
  flyDuration: 2,      // 粒子飞行时间
  diffusionDuration: 1.5, // 扩散切换时间 (决定了模型出现的快慢)
  colorDuration: 4.0,    // 上色时间
};

const globalUniforms = {
  uTime: { value: 0 },
  uCenter: { value: new THREE.Vector3(0, 0, 0) },
  uGeoRadius: { value: 0 },   
  uColorRadius: { value: 0 }, 
  uMaxRadius: { value: 50 },
  uParticleProgress: { value: 0 }, 
};

// --- 2. 粒子系统 (生成稀疏的小圆点) ---
const createParticleSystem = (splatMesh) => {
  if (!viewer) return;

  const splatCount = splatMesh.getSplatCount();
  const maxParticles = 100000; // 4万个点足够勾勒轮廓，太多会卡且不好看
  const step = Math.ceil(splatCount / maxParticles);
  
  const geometry = new THREE.BufferGeometry();
  const startPositions = [];
  const targetPositions = [];
  const randoms = [];

  const tempVec = new THREE.Vector3();
  const box = new THREE.Box3();
  
  splatMesh.updateMatrixWorld();

  for (let i = 0; i < splatCount; i += step) {
    splatMesh.getSplatCenter(i, tempVec);
    tempVec.applyMatrix4(splatMesh.matrixWorld);
    
    targetPositions.push(tempVec.x, tempVec.y, tempVec.z);
    box.expandByPoint(tempVec);

    // 随机分布在远处 (球形分布)
    const r = 30 + Math.random() * 30; 
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    
    const startX = tempVec.x + r * Math.sin(phi) * Math.cos(theta);
    const startY = tempVec.y + r * Math.sin(phi) * Math.sin(theta);
    const startZ = tempVec.z + r * Math.cos(phi);

    startPositions.push(startX, startY, startZ);
    randoms.push(Math.random());
  }

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(startPositions, 3));
  geometry.setAttribute('aTarget', new THREE.Float32BufferAttribute(targetPositions, 3));
  geometry.setAttribute('aRandom', new THREE.Float32BufferAttribute(randoms, 1));

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uProgress: globalUniforms.uParticleProgress,
      uSize: { value: window.devicePixelRatio * 0.7 }, // 小点尺寸
      uColor: { value: new THREE.Color(0.6, 0.6, 0.6) } // 灰色
    },
    vertexShader: `
      uniform float uProgress;
      uniform float uSize;
      attribute vec3 aTarget;
      attribute float aRandom;
      
      // 缓动函数
      float easeOutCubic(float x) {
          return 1.0 - pow(1.0 - x, 3.0);
      }
      
      void main() {
        // 计算每个粒子的进度，稍微错开一点
        float t = (uProgress - aRandom * 0.1) / 0.9;
        t = clamp(t, 0.0, 1.0);
        
        // 插值位置
        vec3 pos = mix(position, aTarget, easeOutCubic(t));
        
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        
        // 距离衰减
        gl_PointSize = uSize * (20.0 / -mvPosition.z);
        if(gl_PointSize < 1.0) gl_PointSize = 1.0;
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      void main() {
        // 圆形裁剪
        vec2 coord = gl_PointCoord - vec2(0.5);
        if(length(coord) > 0.5) discard;
        
        // 颜色输出 (透明度由 JS 控制 material.opacity)
        gl_FragColor = vec4(uColor, 1.0);
      }
    `,
    transparent: true,
    opacity: 1.0,
    depthTest: true,
    depthWrite: false, 
  });

  particleSystem = new THREE.Points(geometry, material);
  particleSystem.frustumCulled = false;
  viewer.threeScene.add(particleSystem);
  
  // 计算中心和包围盒，用于后续扩散动画
  box.getCenter(globalUniforms.uCenter.value);
  const size = new THREE.Vector3();
  box.getSize(size);
  globalUniforms.uMaxRadius.value = Math.max(size.x, size.y, size.z) * 0.7;
};

// --- 3. Shader 注入 (控制 3DGS 的扩散和变色) ---
const applyAdvancedShader = (mesh) => {
  if (!mesh || !mesh.material) return;
  const material = mesh.material;
  
  material.uniforms = material.uniforms || {};
  material.uniforms.uGeoRadius = globalUniforms.uGeoRadius;
  material.uniforms.uColorRadius = globalUniforms.uColorRadius;
  material.uniforms.uMaxRadius = globalUniforms.uMaxRadius;
  material.uniforms.uCenter = globalUniforms.uCenter;

  material.vertexShader = `varying vec3 vWorldPosition;\n` + material.vertexShader;
  const vsEndIndex = material.vertexShader.lastIndexOf('}');
  if (vsEndIndex !== -1) {
    const vsLogic = `vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;\n`;
    material.vertexShader = material.vertexShader.substring(0, vsEndIndex) + vsLogic + '}';
  }

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
      float distFromCenter = distance(vWorldPosition, uCenter);
      
      // 1. 几何扩散逻辑：小于半径才显示
      if (distFromCenter > uGeoRadius) {
          discard;
      }

      // 2. 颜色逻辑：处于 GeoRadius 和 ColorRadius 之间显示为灰色小点
      if (distFromCenter > uColorRadius) {
          if (gl_FragColor.a < 0.8) discard; 
          gl_FragColor.a = 1.0; 
          gl_FragColor.rgb = vec3(0.6, 0.6, 0.6);
      } 
    `;
    material.fragmentShader = originalContent + visualLogic + '}';
  }
  material.needsUpdate = true;
};

// --- 4. 初始化 ---
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
    
    // 初始化状态
    animationState.isLoaded = false;
    animationState.phase = PHASE.FLY_IN;
    globalUniforms.uParticleProgress.value = 0;
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
    splatMesh.visible = false; // 初始隐藏 3DGS，只看粒子

    setTimeout(() => { 
      if (splatMesh) {
        applyAdvancedShader(splatMesh);
        createParticleSystem(splatMesh);
        adjustControlsToModel();
        
        animationState.lastFrameTime = Date.now();
        animationState.startTime = Date.now(); // 记录起始时间
        animationState.isLoaded = true;
      }
    }, 200);
    
    // --- 5. 动画循环 (核心逻辑) ---
    viewer.renderer.setAnimationLoop(() => {
      viewer.update();
      viewer.render();

      if (!animationState.isLoaded || animationState.phase === PHASE.FINISHED) return;

      const now = Date.now();
      const dt = (now - animationState.lastFrameTime) / 1000 || 0.016;
      animationState.lastFrameTime = now;

      // === 阶段 1: 粒子飞入 ===
      if (animationState.phase === PHASE.FLY_IN) {
        const speed = 1.0 / animationState.flyDuration;
        let p = globalUniforms.uParticleProgress.value + (dt * speed);
        
        if (p >= 0.9) {
          p = 0.9; 
          
          // 飞入结束，立即让 3DGS 可见
          const splatMesh = viewer.getSplatMesh();
          if (splatMesh) splatMesh.visible = true;
          
          // 进入扩散阶段，重置计时器
          animationState.phase = PHASE.DIFFUSION;
          animationState.diffuseTime = 0; 
        }
        globalUniforms.uParticleProgress.value = p;
      } 
      
      // === 阶段 2: 扩散切换 (Diffusion) ===
      // 这里实现：粒子渐隐 + 真实模型(灰色点状态)从中心向外渐显
      else if (animationState.phase === PHASE.DIFFUSION) {
        animationState.diffuseTime += dt;
        const progress = Math.min(animationState.diffuseTime / animationState.diffusionDuration, 1.0);
        
        const maxR = globalUniforms.uMaxRadius.value;
        
        // A. 真实模型扩散: 半径 0 -> Max + Buffer
        // 这样真实的点云看起来是从中心长出来的
        globalUniforms.uGeoRadius.value = progress * (maxR + 20.0);
        
        // B. 粒子渐隐: Opacity 1.0 -> 0.0
        // 在模型长出来的同时，虚假粒子慢慢消失
        if (particleSystem && particleSystem.material) {
           particleSystem.material.opacity = 1.0 - progress;
        }

        if (progress >= 1.0) {
          // 确保粒子完全隐藏
          if(particleSystem) particleSystem.visible = false;
          // 确保模型完全显示
          globalUniforms.uGeoRadius.value = 99999.0;
          
          // 进入上色阶段
          animationState.phase = PHASE.COLORING;
          animationState.colorStartTime = now;
        }
      }
      
      // === 阶段 3: 上色扩散 ===
      else if (animationState.phase === PHASE.COLORING) {
        const colorTime = (now - animationState.colorStartTime) / 1000;
        const maxR = globalUniforms.uMaxRadius.value;
        
        // 上色半径扩大
        const progress = colorTime / animationState.colorDuration;
        globalUniforms.uColorRadius.value = progress * (maxR + 10.0);

        if (progress >= 1.0) {
          animationState.phase = PHASE.FINISHED;
          globalUniforms.uColorRadius.value = 99999.0;
        }
      }
    });

    setupDesktopControls();

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
  const worldCenter = globalUniforms.uCenter.value;
  const maxDim = globalUniforms.uMaxRadius.value / 0.7; 
  const distance = maxDim * 2;

  if (viewer.controls) {
    viewer.controls.target.copy(worldCenter);
    viewer.controls.update();
  }
  
  viewer.camera.position.set(worldCenter.x, worldCenter.y, worldCenter.z + distance);
  viewer.camera.lookAt(worldCenter);
};

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