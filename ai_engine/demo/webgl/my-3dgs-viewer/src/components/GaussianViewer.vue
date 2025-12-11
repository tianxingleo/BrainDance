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
  FLY_IN: 0,    
  DIFFUSION: 1, 
  COLORING: 2,  
  FINISHED: 3   
};

const animationState = {
  isLoaded: false,
  lastFrameTime: 0,
  phase: PHASE.FLY_IN, 
  
  flyDuration: 1.5,      
  diffusionDuration: 1.0, 
  colorDuration: 4.0,    
};

const globalUniforms = {
  uTime: { value: 0 },
  uCenter: { value: new THREE.Vector3(0, 0, 0) },
  uGeoRadius: { value: 0 },   
  uColorRadius: { value: 0 }, 
  uMaxRadius: { value: 50 }, // 将由自适应逻辑动态更新
  uParticleProgress: { value: 0 }, 
};

// --- 2. 自适应粒子系统 (核心修改) ---
const createParticleSystem = (splatMesh) => {
  if (!viewer) return;

  const splatCount = splatMesh.getSplatCount();
  splatMesh.updateMatrixWorld();

  // === A. 预计算：计算包围盒与尺寸 ===
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  const tempVec = new THREE.Vector3();
  
  // 为了性能，不需要遍历所有点，每隔 100 个点采样一次即可估算包围盒
  const boundSampleStep = Math.max(1, Math.floor(splatCount / 1000));
  
  for (let i = 0; i < splatCount; i += boundSampleStep) {
    splatMesh.getSplatCenter(i, tempVec);
    tempVec.applyMatrix4(splatMesh.matrixWorld); // 转为世界坐标
    if (tempVec.x < minX) minX = tempVec.x; if (tempVec.x > maxX) maxX = tempVec.x;
    if (tempVec.y < minY) minY = tempVec.y; if (tempVec.y > maxY) maxY = tempVec.y;
    if (tempVec.z < minZ) minZ = tempVec.z; if (tempVec.z > maxZ) maxZ = tempVec.z;
  }

  // 计算中心点和最大边长
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const centerZ = (minZ + maxZ) / 2;
  const maxDim = Math.max(maxX - minX, maxY - minY, maxZ - minZ);

  // 更新全局 Uniforms (供 Shader 和 相机使用)
  globalUniforms.uCenter.value.set(centerX, centerY, centerZ);
  globalUniforms.uMaxRadius.value = maxDim * 0.7; // 扩散半径覆盖大部分模型

  // === B. 自适应参数计算 ===
  
  // 1. 自适应粒子数量
  // 逻辑：至少显示 1万个点，最多显示 40万个点。
  // 如果模型本身小于 4万点，则全部显示。
  let targetParticleCount = 60000; 
  if (splatCount < 40000) targetParticleCount = splatCount; // 小模型全显
  else if (splatCount > 1000000) targetParticleCount = 400000; // 大模型上限
  
  const step = Math.ceil(splatCount / targetParticleCount);

  // 2. 自适应粒子大小
  // 逻辑：模型越大，单个粒子在世界空间中应该越大才能被看见。
  // 系数 150.0 是经验值，表示将最大边长切分多少份。
  let adaptiveSize = (maxDim / 200.0) * window.devicePixelRatio;
  // 限制最小值，防止极小模型看不见
  if (adaptiveSize < 0.5) adaptiveSize = 0.5;

  // 3. 自适应飞行距离
  // 粒子应该从包围盒外面飞进来
  const flyRadiusBase = maxDim * 1.0; 

  console.log(`[Adaptive] MaxDim: ${maxDim.toFixed(2)}, Particles: ~${Math.floor(splatCount/step)}, Size: ${adaptiveSize.toFixed(2)}`);

  // === C. 生成几何体 ===
  const geometry = new THREE.BufferGeometry();
  const startPositions = [];
  const targetPositions = [];
  const randoms = [];

  for (let i = 0; i < splatCount; i += step) {
    splatMesh.getSplatCenter(i, tempVec);
    tempVec.applyMatrix4(splatMesh.matrixWorld);
    
    targetPositions.push(tempVec.x, tempVec.y, tempVec.z);

    // 随机分布在远处 (基于自适应的 maxDim)
    const r = flyRadiusBase + Math.random() * (maxDim * 0.5); 
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    
    // 从中心点向外偏移
    const startX = centerX + r * Math.sin(phi) * Math.cos(theta);
    const startY = centerY + r * Math.sin(phi) * Math.sin(theta);
    const startZ = centerZ + r * Math.cos(phi);

    startPositions.push(startX, startY, startZ);
    randoms.push(Math.random());
  }

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(startPositions, 3));
  geometry.setAttribute('aTarget', new THREE.Float32BufferAttribute(targetPositions, 3));
  geometry.setAttribute('aRandom', new THREE.Float32BufferAttribute(randoms, 1));

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uProgress: globalUniforms.uParticleProgress,
      uSize: { value: adaptiveSize }, // 使用计算出的大小
      uColor: { value: new THREE.Color(0.6, 0.6, 0.6) }
    },
    vertexShader: `
      uniform float uProgress;
      uniform float uSize;
      attribute vec3 aTarget;
      attribute float aRandom;
      
      float easeOutCubic(float x) { return 1.0 - pow(1.0 - x, 3.0); }
      
      void main() {
        float t = (uProgress - aRandom * 0.1) / 0.9;
        t = clamp(t, 0.0, 1.0);
        vec3 pos = mix(position, aTarget, easeOutCubic(t));
        
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        
        // 距离衰减 (20.0 是透视缩放因子，配合世界单位的 uSize 使用)
        gl_PointSize = uSize * (20.0 / -mvPosition.z);
        if(gl_PointSize < 1.0) gl_PointSize = 1.0;
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        if(length(coord) > 0.5) discard;
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
};

// --- 3. Shader 注入 ---
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
      
      if (distFromCenter > uGeoRadius) {
          discard;
      }
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
    
    animationState.isLoaded = false;
    animationState.phase = PHASE.FLY_IN;
    globalUniforms.uParticleProgress.value = 0;
    globalUniforms.uGeoRadius.value = 0; 
    globalUniforms.uColorRadius.value = 0;

    const config = getViewerConfig();
    viewer = new GaussianSplats3D.Viewer(config);
    window.viewer = viewer;

    // 加载你的模型
    await viewer.addSplatScene('/models/scene.splat', {
      'showLoadingUI': true,
      'progressiveLoad': false,
      'rotation': [1, 0, 0, 0],
    });
    
    const splatMesh = viewer.getSplatMesh();
    splatMesh.visible = false; 

    setTimeout(() => { 
      if (splatMesh) {
        // 先生成粒子系统，这会计算出 uCenter 和 uMaxRadius
        createParticleSystem(splatMesh);
        // 然后应用 Shader
        applyAdvancedShader(splatMesh);
        // 最后调整相机，因为现在我们已经有了准确的 Center 和 Radius
        adjustControlsToModel();
        
        animationState.lastFrameTime = Date.now();
        animationState.startTime = Date.now(); 
        animationState.isLoaded = true;
      }
    }, 200);
    
    // --- 5. 动画循环 ---
    viewer.renderer.setAnimationLoop(() => {
      viewer.update();
      viewer.render();

      if (!animationState.isLoaded || animationState.phase === PHASE.FINISHED) return;

      const now = Date.now();
      const dt = (now - animationState.lastFrameTime) / 1000 || 0.016;
      animationState.lastFrameTime = now;

      // 1. 飞入
      if (animationState.phase === PHASE.FLY_IN) {
        const speed = 1.0 / animationState.flyDuration;
        let p = globalUniforms.uParticleProgress.value + (dt * speed);
        
        if (p >= 1.2) { // 稍微给点余量保证完全到达
          p = 1.2; 
          const splatMesh = viewer.getSplatMesh();
          if (splatMesh) splatMesh.visible = true;
          
          animationState.phase = PHASE.DIFFUSION;
          animationState.diffuseTime = 0; 
        }
        globalUniforms.uParticleProgress.value = p;
      } 
      
      // 2. 扩散切换
      else if (animationState.phase === PHASE.DIFFUSION) {
        animationState.diffuseTime += dt;
        const progress = Math.min(animationState.diffuseTime / animationState.diffusionDuration, 1.0);
        
        const maxR = globalUniforms.uMaxRadius.value;
        globalUniforms.uGeoRadius.value = progress * (maxR * 1.5); // 确保覆盖角落
        
        if (particleSystem && particleSystem.material) {
           particleSystem.material.opacity = 1.0 - progress;
        }

        if (progress >= 1.0) {
          if(particleSystem) particleSystem.visible = false;
          globalUniforms.uGeoRadius.value = 99999.0;
          
          animationState.phase = PHASE.COLORING;
          animationState.colorStartTime = now;
        }
      }
      
      // 3. 上色
      else if (animationState.phase === PHASE.COLORING) {
        const colorTime = (now - animationState.colorStartTime) / 1000;
        const maxR = globalUniforms.uMaxRadius.value;
        const progress = colorTime / animationState.colorDuration;
        
        globalUniforms.uColorRadius.value = progress * (maxR * 1.5);

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

// 修改后的 adjustControlsToModel，直接使用预计算好的值
const adjustControlsToModel = () => {
  if (isVRMode.value) return;
  
  // createParticleSystem 已经计算了最准确的 uCenter 和 uMaxRadius，直接用
  const worldCenter = globalUniforms.uCenter.value;
  const maxDim = globalUniforms.uMaxRadius.value / 0.7; // 还原回实际尺寸估计
  const distance = maxDim * 2.0;

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