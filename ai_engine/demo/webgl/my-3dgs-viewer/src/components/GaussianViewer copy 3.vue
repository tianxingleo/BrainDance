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

const globalUniforms = {
  uTime: { value: 0 },
  uRevealRadius: { value: 0 },
  uMaxRadius: { value: 50 },
  uCenter: { value: new THREE.Vector3(0, 0, 0) },
  uCenterDistance: { value: 0 },
  // 新增：画布分辨率，用于修正球体扩散形状
  uResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) }
};

const animationState = {
  isLoaded: false,
  startTime: 0
};

// --- 修复并优化后的 Shader 注入函数 ---
const applyAdvancedShader = (mesh) => {
  if (!mesh || !mesh.material) return;
  const material = mesh.material;
  material.uniforms = material.uniforms || {};

  material.uniforms.uTime = globalUniforms.uTime;
  material.uniforms.uRevealRadius = globalUniforms.uRevealRadius;
  material.uniforms.uMaxRadius = globalUniforms.uMaxRadius;
  material.uniforms.uCenterDistance = globalUniforms.uCenterDistance;
  material.uniforms.uResolution = globalUniforms.uResolution;

  const commonFragment = `
    uniform float uTime;
    uniform float uRevealRadius;
    uniform float uMaxRadius;
    uniform float uCenterDistance;
    uniform vec2 uResolution;
  `;

  material.fragmentShader = commonFragment + material.fragmentShader;

  const lastBracketIndex = material.fragmentShader.lastIndexOf('}');
  if (lastBracketIndex !== -1) {
    const originalContent = material.fragmentShader.substring(0, lastBracketIndex);

    const visualLogic = `
      // --- 视觉特效注入 ---
      
      float viewZ = 1.0 / gl_FragCoord.w;
      vec2 ndc = (gl_FragCoord.xy / uResolution.xy) * 2.0 - 1.0;
      ndc.x *= uResolution.x / uResolution.y;
      float offsetX = ndc.x * viewZ * 0.5;
      float offsetY = ndc.y * viewZ * 0.5;
      float offsetZ = viewZ - uCenterDistance;
      float distFromCenter = sqrt(offsetX * offsetX + offsetY * offsetY + offsetZ * offsetZ);
      
      if (uRevealRadius < uMaxRadius) {
          if (distFromCenter > uRevealRadius) {
              if (mod(gl_FragCoord.x + gl_FragCoord.y, 2.0) == 0.0) discard;
              if (gl_FragColor.a < 0.83) discard;
              gl_FragColor.a = 1.0;
              float gray = dot(gl_FragColor.rgb, vec3(0.299, 0.587, 0.114));
              gl_FragColor.rgb = vec3(gray * 0.5 + 0.1);
          }
          float distDiff = abs(distFromCenter - uRevealRadius);
          if (distDiff < 1.0) {
             float intensity = 1.0 - (distDiff / 1.0);
             intensity = pow(intensity, 2.0);
             gl_FragColor.rgb += vec3(0.6, 0.6, 0.6) * intensity;
          }
      }
    `;

    material.fragmentShader = originalContent + visualLogic + '}';
  }

  material.needsUpdate = true;
};

// 1. 配置生成器
const getViewerConfig = () => {
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  return {
    'rootElement': containerRef.value,
    'cameraUp': [0, 1, 0],
    'initialCameraPosition': [0, 0, 5],
    'initialCameraLookAt': [0, 0, 0],
    'useBuiltInControls': false,
    'gpuAcceleratedSort': false, // 保持关闭以避免兼容性问题
    'webXRMode': isSecureContext.value ? GaussianSplats3D.WebXRMode.VR : GaussianSplats3D.WebXRMode.None,
    'sharedMemoryForWorkers': false,
    'integerBasedSort': true,
    'enableSIMDInSort': !isMobile,
    'splatAlphaRemovalThreshold': 1, // 设低一点，由我们的 shader 接管剔除
    'antialiased': !isMobile,
  };
};

// 2. 初始化核心逻辑
const initViewer = async () => {
  if (isLoading.value) return;
  isLoading.value = true;

  try {
    // 安全清理旧实例
    if (viewer) {
      // 停止循环
      viewer.renderer.setAnimationLoop(null);
      // 尝试清理
      if(viewer.dispose) await viewer.dispose();
      viewer = null;
    }
    
    // 清理 DOM (解决 removeChild 报错)
    if (containerRef.value) {
      while (containerRef.value.firstChild) {
        containerRef.value.removeChild(containerRef.value.firstChild);
      }
    }
    
    // 重置状态
    animationState.isLoaded = false;
    globalUniforms.uRevealRadius.value = 0;

    const config = getViewerConfig();
    viewer = new GaussianSplats3D.Viewer(config);
    window.viewer = viewer;
    window.THREE = THREE;

    // 加载模型
    await viewer.addSplatScene('/models/scene.splat', {
      'showLoadingUI': true,
      'progressiveLoad': false, // 必须关闭，否则 Material 会被重置
      'rotation': [1, 0, 0, 0],
    });
    
    // 应用特效
    const splatMesh = viewer.getSplatMesh();
    // 稍微延迟确保 mesh 构建完成
    setTimeout(() => {
       if (splatMesh) {
           applyAdvancedShader(splatMesh);
       }
    }, 100);

    // 启动动画循环
    animationState.startTime = Date.now();
    animationState.isLoaded = true;
    
// ... inside initViewer ...

    viewer.renderer.setAnimationLoop(() => {
      if (animationState.isLoaded) {
          const now = Date.now();
          const time = (now - animationState.startTime) / 1000;
          globalUniforms.uTime.value = time;
          
          // 实时计算相机到模型中心的距离
          if (viewer && viewer.camera) {
              const dist = viewer.camera.position.distanceTo(globalUniforms.uCenter.value);
              globalUniforms.uCenterDistance.value = dist;
          }
          
          const maxR = globalUniforms.uMaxRadius.value;
          
          // --- 核心修改：极慢速度控制 ---
          
          // 目标：我们希望动画持续约 8 秒
          // 假设 60 FPS，总帧数大约 480 帧
          // 每一帧增加的距离 = 总距离 / 480
          // 增加一个 Math.max 确保最小速度不为 0 (防止模型极小时卡住)
          const step = Math.max(0.005, maxR / 480.0); 

          if (globalUniforms.uRevealRadius.value < maxR) {
              globalUniforms.uRevealRadius.value += step; 
          } else {
              // 动画结束
              globalUniforms.uRevealRadius.value = 99999.0;
          }
      }
      
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
  controls.enableDamping = true;
  viewer.controls = controls;
};

// 3. 对焦辅助函数
const adjustControlsToModel = () => {
  if (isVRMode.value) return;
  const mesh = viewer.getSplatMesh();
  setTimeout(() => {
    if (mesh.getSplatCount() > 0) {
      mesh.updateMatrixWorld();
      
      // 简单的包围盒计算
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
      
      // --- 更新动画参数 ---
      globalUniforms.uCenter.value.copy(worldCenter);
      
      // MaxRadius 设为模型最大尺寸的一半左右即可覆盖全身
      // 为了保险，设大一点
      globalUniforms.uMaxRadius.value = maxDim * 0.8; 
      
      globalUniforms.uRevealRadius.value = 0.0; 
      animationState.startTime = Date.now();
      // ----------------

      viewer.camera.position.set(worldCenter.x, worldCenter.y, worldCenter.z + distance);
      viewer.camera.lookAt(worldCenter);
    }
  }, 100);
};

// VR 和辅助函数保持不变
const onSessionStarted = (session) => {
  isVRMode.value = true;
  if (viewer && viewer.controls) { viewer.controls.dispose(); viewer.controls = null; }
  session.addEventListener('end', onSessionEnded);
};
const onSessionEnded = () => { isVRMode.value = false; setupDesktopControls(); };
const toggleVRMode = async () => { /* ...保持原样... */ };
const toggleAutoRotate = () => { isAutoRotate.value = !isAutoRotate.value; };
const checkProtocol = () => { /* ...保持原样... */ };

// 生命周期
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
      <!-- 按钮代码保持原样 -->
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
/* 样式保持原样 */
.app-container { position: relative; width: 100vw; height: 100vh; background-color: #000000; }
.viewer-container { width: 100%; height: 100%; }
.controls-ui { position: absolute; top: 30px; left: 50%; transform: translateX(-50%); display: flex; gap: 15px; z-index: 100; }
.loading-overlay { position: absolute; inset: 0; background: rgba(0,0,0,0.8); color: white; display: flex; justify-content: center; align-items: center; z-index: 200; font-size: 20px; }
button { background: rgba(0,0,0,0.6); color: white; border: 1px solid rgba(255,255,255,0.3); padding: 10px 20px; border-radius: 20px; cursor: pointer; transition: 0.3s; }
button.active { background: #22c55e; border-color: #22c55e; }
</style>