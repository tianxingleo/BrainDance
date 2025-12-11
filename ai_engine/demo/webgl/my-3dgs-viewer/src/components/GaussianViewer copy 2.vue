<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue';
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import { ArcballControls } from 'three/addons/controls/ArcballControls.js';

const containerRef = ref(null);
const isLoading = ref(true);
const loadingText = ref("正在准备场景...");
const isSecureContext = ref(false);

// --- 核心变量 ---
let viewer = null;          
let particleSystem = null;  
let controls = null;        
let animationId = null;
let clock = new THREE.Clock();

const CONFIG = {
  filePath: '/models/point_cloud_cleaned.ply',
  gatherDuration: 3.5,      
  fadeDuration: 1.5,        
  colorDuration: 4.0,       
  particleCount: 50000,     
};

const state = {
  phase: 0, 
  startTime: 0,
};

const modelUniforms = {
  uColorProgress: { value: 0.0 }, 
  uOpacity: { value: 0.0 },       
  uCenter: { value: new THREE.Vector3(0,0,0) }
};

// ==========================================
// 1. 创建替身粒子 (修复 Shader 报错)
// ==========================================
const createProxyParticles = (targetCenter, targetRadius) => {
  const count = CONFIG.particleCount;
  const geometry = new THREE.BufferGeometry();
  const positions = [];     
  const startPositions = [];
  const endPositions = [];  
  const colors = [];

  for (let i = 0; i < count; i++) {
    const r = targetRadius * Math.cbrt(Math.random()); 
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    
    const x = targetCenter.x + r * Math.sin(phi) * Math.cos(theta);
    const y = targetCenter.y + r * Math.sin(phi) * Math.sin(theta);
    const z = targetCenter.z + r * Math.cos(phi);
    
    endPositions.push(x, y, z);
    positions.push(x, y, z); 

    const flyDist = 30 + Math.random() * 30;
    const dir = new THREE.Vector3(x - targetCenter.x, y - targetCenter.y, z - targetCenter.z).normalize();
    if (dir.length() === 0) dir.set(0,1,0);
    
    startPositions.push(
      targetCenter.x + dir.x * flyDist, 
      targetCenter.y + dir.y * flyDist, 
      targetCenter.z + dir.z * flyDist
    );

    colors.push(0.6, 0.6, 0.6); 
  }

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('aStart', new THREE.Float32BufferAttribute(startPositions, 3));
  geometry.setAttribute('aEnd', new THREE.Float32BufferAttribute(endPositions, 3));
  // 🔴 修复 1: 改名为 aColor，避免与内置 color 冲突
  geometry.setAttribute('aColor', new THREE.Float32BufferAttribute(colors, 3));

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uProgress: { value: 0.0 }, 
      uAlpha: { value: 1.0 },    
      uSize: { value: 3.0 * window.devicePixelRatio }
    },
    // 🔴 修复 2: 显式声明所有 attribute
    vertexShader: `
      uniform float uProgress;
      uniform float uSize;
      
      attribute vec3 aStart;
      attribute vec3 aEnd;
      attribute vec3 aColor; // 显式声明颜色属性
      
      varying vec3 vColor;
      
      float easeOutCubic(float x) { return 1.0 - pow(1.0 - x, 3.0); }

      void main() {
        vColor = aColor; // 使用自定义的 aColor
        
        float t = easeOutCubic(uProgress);
        vec3 pos = mix(aStart, aEnd, t);
        
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        
        gl_PointSize = uSize * (8.0 / -mvPosition.z);
      }
    `,
    fragmentShader: `
      uniform float uAlpha;
      varying vec3 vColor;
      void main() {
        // 圆形裁切
        vec2 coord = gl_PointCoord - vec2(0.5);
        if(length(coord) > 0.5) discard;
        
        gl_FragColor = vec4(vColor, uAlpha);
      }
    `,
    transparent: true,
    depthWrite: false, 
    blending: THREE.AdditiveBlending,
    vertexColors: false // 关闭自动颜色处理，完全手动接管
  });

  return new THREE.Points(geometry, material);
};

// ==========================================
// 2. 注入模型 Shader (稳健版)
// ==========================================
const injectModelShader = (mesh) => {
  const material = mesh.material;
  material.uniforms = material.uniforms || {};
  material.uniforms.uColorProgress = modelUniforms.uColorProgress;
  material.uniforms.uOpacity = modelUniforms.uOpacity;
  material.uniforms.uCenter = modelUniforms.uCenter;

  const vsHead = `varying vec3 vPos;`;
  if (!material.vertexShader.includes(vsHead)) {
    material.vertexShader = vsHead + material.vertexShader;
    const end = material.vertexShader.lastIndexOf('}');
    material.vertexShader = material.vertexShader.substring(0, end) + 
      `vPos = (modelMatrix * vec4(position, 1.0)).xyz;\n}` ;
  }

  const fsHead = `
    uniform float uOpacity;
    uniform float uColorProgress;
    uniform vec3 uCenter;
    varying vec3 vPos;
  `;
  if (!material.fragmentShader.includes('uniform float uOpacity;')) {
    material.fragmentShader = fsHead + material.fragmentShader;
    
    const end = material.fragmentShader.lastIndexOf('}');
    const logic = `
      // 1. 透明度淡入
      gl_FragColor.a *= uOpacity;
      
      // 2. 变色逻辑 (从中心向外变彩)
      float dist = distance(vPos, uCenter);
      float colorRadius = uColorProgress * 100.0; 
      
      if (dist > colorRadius) {
         // 变灰
         float gray = dot(gl_FragColor.rgb, vec3(0.299, 0.587, 0.114));
         gl_FragColor.rgb = vec3(gray);
      }
    `;
    material.fragmentShader = material.fragmentShader.substring(0, end) + logic + '}';
  }
  
  material.needsUpdate = true;
};

// ==========================================
// 3. 初始化全流程
// ==========================================
const initViewer = async () => {
  if (containerRef.value) containerRef.value.innerHTML = '';
  
  viewer = new GaussianSplats3D.Viewer({
    'rootElement': containerRef.value,
    'cameraUp': [0, 1, 0],
    'initialCameraPosition': [0, 0, 10], 
    'initialCameraLookAt': [0, 0, 0],
    'useBuiltInControls': false,         
    'gpuAcceleratedSort': true,
    'splatAlphaRemovalThreshold': 5      
  });
  
  try {
    loadingText.value = "加载模型...";
    await viewer.addSplatScene(CONFIG.filePath, {
      'showLoadingUI': false,
      'progressiveLoad': false,
      'rotation': [1, 0, 0, 0]
    });
    
    console.log("✅ 模型加载完成");
    loadingText.value = "";
    isLoading.value = false;

    const splatMesh = viewer.getSplatMesh();
    splatMesh.visible = true; 
    splatMesh.frustumCulled = false;
    
    // 计算中心
    splatMesh.updateMatrixWorld();
    const center = new THREE.Vector3(0, 0, 0);
    const radius = 10.0;
    modelUniforms.uCenter.value.copy(center);
    
    // 初始状态：模型透明
    modelUniforms.uOpacity.value = 0.0; 
    injectModelShader(splatMesh);

    // 添加粒子
    particleSystem = createProxyParticles(center, radius);
    viewer.threeScene.add(particleSystem);

    // 控制器
    if (controls) controls.dispose();
    controls = new ArcballControls(viewer.camera, viewer.renderer.domElement, viewer.threeScene);
    controls.setGizmosVisible(false); // 🔴 关闭红绿蓝球
    controls.enableDamping = true;
    
    viewer.start();
    state.startTime = clock.getElapsedTime();
    animate();

  } catch (e) {
    console.error("初始化失败", e);
    loadingText.value = "加载失败: " + e.message;
  }
};

// ==========================================
// 4. 动画循环
// ==========================================
const animate = () => {
  animationId = requestAnimationFrame(animate);
  
  const now = clock.getElapsedTime();
  const time = now - state.startTime;
  
  if (controls) controls.update();

  // 1. 聚拢
  if (time <= CONFIG.gatherDuration) {
    const p = time / CONFIG.gatherDuration; 
    if (particleSystem) {
      particleSystem.material.uniforms.uProgress.value = p;
      particleSystem.material.uniforms.uAlpha.value = 1.0;
    }
    modelUniforms.uOpacity.value = 0.0;
  }
  
  // 2. 融合 (粒子淡出，模型淡入)
  else if (time <= CONFIG.gatherDuration + CONFIG.fadeDuration) {
    const fadeP = (time - CONFIG.gatherDuration) / CONFIG.fadeDuration;
    
    if (particleSystem) {
      particleSystem.material.uniforms.uProgress.value = 1.0;
      particleSystem.material.uniforms.uAlpha.value = 1.0 - fadeP;
    }
    modelUniforms.uOpacity.value = fadeP;
  }
  
  // 3. 上色
  else {
    if (particleSystem && particleSystem.parent) {
      particleSystem.parent.remove(particleSystem);
      particleSystem.geometry.dispose();
      particleSystem = null; 
    }
    
    modelUniforms.uOpacity.value = 1.0;
    
    const colorStartTime = CONFIG.gatherDuration + CONFIG.fadeDuration;
    const colorP = (time - colorStartTime) / CONFIG.colorDuration;
    
    modelUniforms.uColorProgress.value = Math.min(colorP, 1.0);
  }
};

const checkProtocol = () => { 
  const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  isSecureContext.value = isLocal || window.location.protocol === 'https:';
};

onMounted(() => { 
  if (containerRef.value) { 
    checkProtocol(); 
    initViewer(); 
  } 
});

onBeforeUnmount(() => {
  if (animationId) cancelAnimationFrame(animationId);
  if (viewer) viewer.dispose();
});
</script>

<template>
  <div class="app-container">
    <div ref="containerRef" class="viewer-container"></div>
    <div v-if="isLoading" class="loading-overlay">
      <div class="loader-text">{{ loadingText }}</div>
    </div>
    <div class="controls-ui">
      <button v-if="isSecureContext" class="btn">VR 模式</button>
    </div>
  </div>
</template>

<style scoped>
.app-container { position: relative; width: 100vw; height: 100vh; background-color: #000000; }
.viewer-container { width: 100%; height: 100%; }
.loading-overlay { 
  position: absolute; inset: 0; background: black; 
  display: flex; justify-content: center; align-items: center; z-index: 200; 
}
.loader-text { color: #22c55e; font-family: monospace; font-size: 18px; }
.controls-ui { position: absolute; top: 30px; left: 50%; transform: translateX(-50%); z-index: 100; }
.btn { background: rgba(0,0,0,0.5); border: 1px solid #444; color: white; padding: 8px 16px; border-radius: 20px; }
</style>