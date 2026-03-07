<script setup>
import { onMounted, onBeforeUnmount, ref, computed } from 'vue';
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import { ArcballControls } from 'three/addons/controls/ArcballControls.js';
import gsap from 'gsap';

const containerRef = ref(null);
const isVRMode = ref(false);
const isAutoRotate = ref(false);
const isLoading = ref(false);
const isSecureContext = ref(false);
const cameraPoses = ref([]);
const searchQuery = ref(''); // 绑定搜索框的数据
const activeImage = ref(''); // 当前激活的参考图
const activeTag = ref(''); // 当前激活的标签
const sceneMetadata = ref({}); // 存储 FOV 等元数据
const debugInfo = ref({ x: 0, y: 0, z: 0 }); // 调试用的旋转信息
const arrivalEuler = ref({ x: 0, y: 0, z: 0 }); // 刚飞到时的欧拉角
const loadError = ref(''); // 添加错误状态
const currentFps = ref(0); // 实时帧数
const showFocalSettings = ref(false); // 焦距设置面板
const currentViewFov = ref(0); // 当前相机FOV
const currentViewFocalPx = ref(0); // 当前相机等效焦距（像素）
const manualFocalPx = ref(null); // 手动焦距输入

const filteredPoses = computed(() => {
  if (!searchQuery.value.trim()) {
    // 没搜索时，优先展示有打标的镜头。如果全都没打标，则展示全部
    const withTags = cameraPoses.value.filter(pose => pose.tag);
    if (withTags.length > 0) return withTags;
    // 如果都无标签，最多展示 60 个，避免缩略图过多卡顿
    return cameraPoses.value.slice(0, 60);
  }
  // 如果输入了搜索词，执行基于标签的文本包含匹配
  const query = searchQuery.value.trim().toLowerCase();
  return cameraPoses.value.filter(pose =>
    pose.tag && pose.tag.toLowerCase().includes(query)
  );
});

const searchAndFly = () => {
  if (filteredPoses.value.length > 0) {
    // 直接飞向过滤后的第一个镜头
    flyToImage(filteredPoses.value[0]);
  } else {
    alert("场景中没有找到符合该描述的视角哦~");
  }
};

let viewer;
let particleSystem;

const rotationDelta = ref({ x: 0, y: 0 }); // 记录用户微调了多少度

const calcFovFromFocal = (focalPx, imageHeightPx) => {
  if (!focalPx || !imageHeightPx) return null;
  return 2 * Math.atan((imageHeightPx / 2) / focalPx) * (180 / Math.PI);
};

const calcFocalFromFov = (fovDeg, imageHeightPx) => {
  if (!fovDeg || !imageHeightPx) return null;
  const halfFovRad = (fovDeg * Math.PI / 180) / 2;
  if (halfFovRad <= 0) return null;
  return (imageHeightPx / 2) / Math.tan(halfFovRad);
};

const refreshCurrentFocalInfo = () => {
  if (!viewer || !viewer.camera) return;
  const h = sceneMetadata.value.h || containerRef.value?.clientHeight || window.innerHeight;
  currentViewFov.value = Number(viewer.camera.fov || 0);
  if (h && currentViewFov.value > 0 && currentViewFov.value < 179) {
    const focal = calcFocalFromFov(currentViewFov.value, h);
    currentViewFocalPx.value = focal ? Number(focal.toFixed(1)) : 0;
  }
};

const applyFocalLengthPx = (focalPx, options = {}) => {
  if (!viewer || !viewer.camera) return;
  // 优先使用位姿 JSON 中记录的真实图像高度，否则回退到视口高度
  const h = sceneMetadata.value.h || containerRef.value?.clientHeight || window.innerHeight;
  if (!h || !focalPx) return;

  const targetFov = calcFovFromFocal(focalPx, h);
  if (!targetFov || !Number.isFinite(targetFov)) return;

  const cam = viewer.camera;
  const duration = options.duration ?? 0;
  if (duration > 0) {
    gsap.to(cam, {
      fov: targetFov,
      duration,
      ease: options.ease || 'power2.out',
      onUpdate: () => {
        cam.updateProjectionMatrix();
        // 更新 splat shader uniforms 并立即重绘
        try { viewer.update(); viewer.render(); } catch (_) {}
        refreshCurrentFocalInfo();
      }
    });
  } else {
    cam.fov = targetFov;
    cam.updateProjectionMatrix();
    // 更新 splat shader uniforms 并立即重绘，无需等到下一帧
    try { viewer.update(); viewer.render(); } catch (_) {}
    refreshCurrentFocalInfo();
  }
};

const focalMin = computed(() => {
  const base = Number(sceneMetadata.value.fl_y || 0);
  if (base > 0) return Math.max(50, Math.floor(base * 0.4));
  return 50;
});

const focalMax = computed(() => {
  const base = Number(sceneMetadata.value.fl_y || 0);
  if (base > 0) return Math.max(500, Math.ceil(base * 2.5));
  return 3000;
});

const toggleFocalSettings = () => {
  showFocalSettings.value = !showFocalSettings.value;
  if (showFocalSettings.value && !manualFocalPx.value) {
    manualFocalPx.value = Number(
      (currentViewFocalPx.value || sceneMetadata.value.fl_y || 500).toFixed(1)
    );
  }
};

const onManualFocalChange = () => {
  const value = Number(manualFocalPx.value);
  if (!Number.isFinite(value) || value <= 0) return;
  applyFocalLengthPx(value);
};

const resetFocalToCapture = () => {
  const captureFocal = Number(sceneMetadata.value.fl_y || 0);
  if (!captureFocal) return;
  manualFocalPx.value = Number(captureFocal.toFixed(1));
  applyFocalLengthPx(captureFocal, { duration: 0.5, ease: 'power2.inOut' });
};

const updateDebugInfo = () => {
  if (!viewer || !viewer.camera) return;
  const euler = new THREE.Euler().setFromQuaternion(viewer.camera.quaternion, 'YXZ');
  debugInfo.value = {
    x: (euler.x * 180 / Math.PI).toFixed(1),
    y: (euler.y * 180 / Math.PI).toFixed(1),
    z: (euler.z * 180 / Math.PI).toFixed(1)
  };
  refreshCurrentFocalInfo();
};

const copyMatrixToClipboard = () => {
  if (!viewer || !viewer.camera) return;
  const matrix = viewer.camera.matrixWorld.elements;
  const data = {
    image: activeImage.value,
    matrix: Array.from(matrix),
    debug: debugInfo.value,
    delta: rotationDelta.value
  };
  const str = JSON.stringify(data);
  navigator.clipboard.writeText(str);
  alert("相机数据已复制到剪贴板！请发送给我。");
  console.log("Current Debug Camera Data:", data);
};

const manualMove = (axis, dist) => {
  if (!viewer || !viewer.camera) return;
  if (viewer.controls) viewer.controls.enabled = false;

  if (axis === 'x') viewer.camera.translateX(dist);
  if (axis === 'y') viewer.camera.translateY(dist);
  if (axis === 'z') viewer.camera.translateZ(dist);

  viewer.camera.updateProjectionMatrix();
};

const manualRotate = (axis, angleDeg) => {
  if (!viewer || !viewer.camera) return;

  if (viewer.controls) viewer.controls.enabled = false;

  const angle = angleDeg * Math.PI / 180;

  if (axis === 'x') {
    viewer.camera.rotateX(angle);
    rotationDelta.value.x += angleDeg;
  }
  if (axis === 'y') {
    viewer.camera.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), angle);
    rotationDelta.value.y += angleDeg;
  }
  if (axis === 'z') {
    viewer.camera.rotateZ(angle);
  }

  viewer.camera.updateProjectionMatrix();
  updateDebugInfo();
};

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

  console.log(`[Adaptive] MaxDim: ${maxDim.toFixed(2)}, Particles: ~${Math.floor(splatCount / step)}, Size: ${adaptiveSize.toFixed(2)}`);

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

  material.vertexShader = `varying vec3 vWorldPosition;
` + material.vertexShader;
  const vsEndIndex = material.vertexShader.lastIndexOf('}');
  if (vsEndIndex !== -1) {
    const vsLogic = `vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
`;
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

// --- 5. 初始化 ---
const flyToImage = (poseData) => {
  if (!viewer || !viewer.camera) return;

  const cam = viewer.camera;
  const splatMesh = viewer.getSplatMesh(); // 获取当前加载的高斯模型

  // 更新参考图
  activeImage.value = poseData.image_url;
  activeTag.value = poseData.tag || '';

  // 1. 读取原始矩阵 (假设后端传来的是按列优先的 16 位数组)
  const rawMatrix = new THREE.Matrix4().fromArray(poseData.matrix);

  // === 修正：移除多余的坐标系转换 ===
  // 用户反馈：当前状态下再旋转X轴180度才正确。
  // 原有的 makeScale(1, -1, -1) 本质就是X轴转180度。
  // 如果还需要再转180度，说明不需要转，或者需要抵消。
  // 我们先尝试直接移除这个转换，保持原始矩阵方向。
  // const cvToGl = new THREE.Matrix4().makeScale(1, -1, -1);
  // rawMatrix.multiply(cvToGl); 

  // 如果移除后反了，说明 export_poses.py 也没转，那就需要取消注释下面这行来手动修正：
  // const manualFix = new THREE.Matrix4().makeRotationX(Math.PI);
  // rawMatrix.multiply(manualFix);

  // === 核心修正 2：跟随模型的世界矩阵同步旋转/缩放 ===
  // 将相机的原始矩阵，乘以高斯模型目前在 Three.js 世界中的矩阵
  const finalMatrix = new THREE.Matrix4();
  if (splatMesh) {
    // 这样无论模型怎么被 `rotation: [1,0,0,0]` 旋转，相机都会跟过去
    splatMesh.updateMatrixWorld();
    finalMatrix.copy(splatMesh.matrixWorld).multiply(rawMatrix);
  } else {
    finalMatrix.copy(rawMatrix);
  }

  // 提取最终对齐后的 位置 和 旋转
  const targetPosition = new THREE.Vector3();
  const targetQuaternion = new THREE.Quaternion();
  const targetScale = new THREE.Vector3();
  finalMatrix.decompose(targetPosition, targetQuaternion, targetScale);

  // === 核心修正 3：同步真实相机的视场角 (FOV) ===
  const fl_y = poseData.fl_y || sceneMetadata.value.fl_y;
  const h = poseData.h || sceneMetadata.value.h;
  if (fl_y && h) {
    sceneMetadata.value.h = h;
    manualFocalPx.value = Number(fl_y.toFixed(1));
    applyFocalLengthPx(fl_y, { duration: 1.5, ease: 'power3.inOut' });
  }

  // 强行减小近剪裁面，防止“穿模”或由于贴太近导致不显示
  if (cam.near > 0.001) {
    cam.near = 0.001;
    cam.updateProjectionMatrix();
  }

  // 计算控制器焦点：看向相机正前方 5 个单位处
  const forwardVector = new THREE.Vector3(0, 0, -1).applyQuaternion(targetQuaternion);
  const targetLookAt = targetPosition.clone().add(forwardVector.multiplyScalar(5.0));

  // 停用当前控制
  isAutoRotate.value = false;
  if (viewer.controls) viewer.controls.enabled = false;

  const startPos = cam.position.clone();
  const startQuat = cam.quaternion.clone();
  const animState = { t: 0 };

  gsap.killTweensOf(cam.position);
  gsap.killTweensOf(cam.quaternion);
  gsap.killTweensOf(animState);

  // 开始丝滑运镜
  gsap.to(animState, {
    t: 1.0,
    duration: 1.5,
    ease: "power3.inOut",
    onUpdate: () => {
      cam.position.lerpVectors(startPos, targetPosition, animState.t);
      cam.quaternion.slerpQuaternions(startQuat, targetQuaternion, animState.t);
    },
    onComplete: () => {
      // 记录初始飞到后的欧拉角
      const euler = new THREE.Euler().setFromQuaternion(cam.quaternion, 'YXZ');
      arrivalEuler.value = {
        x: (euler.x * 180 / Math.PI).toFixed(1),
        y: (euler.y * 180 / Math.PI).toFixed(1),
        z: (euler.z * 180 / Math.PI).toFixed(1)
      };
      rotationDelta.value = { x: 0, y: 0 }; // 飞跃新镜头时，重置手动偏差
      updateDebugInfo();

      if (viewer.controls) {
        viewer.controls.target.copy(targetLookAt);
        viewer.controls.update();
        viewer.controls.enabled = true;
      }
    }
  });
};

const getViewerConfig = () => {
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  return {
    'rootElement': containerRef.value,
    'cameraUp': [0, 1, 0],
    'initialCameraPosition': [0, 0, 5],
    'initialCameraLookAt': [0, 0, 0],
    'useBuiltInControls': false,
    'gpuAcceleratedSort': false,
    'webXRMode': GaussianSplats3D.WebXRMode.None,
    'sharedMemoryForWorkers': false,
    'antialiased': !isMobile,
  };
};

// 当前加载的 PLY 和位姿的 URL（供外部通过 loadModelFromFlutter 传入）
let currentPlyUrl = '/models/scene_auto_sync.ply';
let currentPosesUrl = '/models/webgl_poses_with_tags.json';

const initViewer = async (plyUrl, posesUrl, initialPoseMatrix) => {
  if (isLoading.value) return;
  isLoading.value = true;

  // 更新 URL（如果有新传入的值）
  if (plyUrl) currentPlyUrl = plyUrl;
  if (posesUrl) currentPosesUrl = posesUrl;

  try {
    if (viewer) {
      viewer.renderer.setAnimationLoop(null);
      if (viewer.dispose) await viewer.dispose();
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

    // 加载模型（优先使用外部传入的云端 URL，缺省使用本地路径）
    console.log(`[Viewer] 加载模型: ${currentPlyUrl}`);
    await viewer.addSplatScene(currentPlyUrl, {
      'showLoadingUI': true,
        'progressiveLoad': false,
        'rotation': [0, 0, 0, 1] // [x, y, z, w] Identity Quaternion (No global rotation)
      });

    
    // 告诉 Flutter：模型加载完成
    isLoading.value = false;
    if (window.BrainDanceChannel) {
      window.BrainDanceChannel.postMessage(JSON.stringify({ status: 'success', msg: '模型加载完成' }));
    }

    // 加载相机位姿（支持本地路径与云端 URL）
    console.log(`[Viewer] 加载位姿: ${currentPosesUrl}`);
    fetch(currentPosesUrl)
      .then(res => res.json())
      .then(data => {
        // 数据适配
        if (data.frames) {
          sceneMetadata.value = {
            w: data.w,
            h: data.h,
            fl_x: data.fl_x,
            fl_y: data.fl_y
          };
          manualFocalPx.value = Number((data.fl_y || 0).toFixed(1));
          cameraPoses.value = data.frames.map(frame => {
            let imgUrl = frame.image_url;
            if (imgUrl && !imgUrl.startsWith('http')) {
              if (currentPosesUrl.startsWith('http')) {
                // Determine base path from currentPosesUrl
                const baseUrl = currentPosesUrl.substring(0, currentPosesUrl.lastIndexOf('/'));
                let relPath = imgUrl;
                const imagesIndex = relPath.indexOf('images/');
                if (imagesIndex !== -1) {
                  relPath = relPath.substring(imagesIndex); // Extracts 'images/frame_xxx.jpg' and drops any redundant parent dirs
                } else if (relPath.startsWith('/models/')) {
                  relPath = relPath.substring('/models/'.length);
                } else if (relPath.startsWith('/')) {
                  relPath = relPath.substring(1);
                }
                imgUrl = `${baseUrl}/${relPath}`;
              }
            }
            return {
              id: frame.id,
              matrix: frame.matrix,
              image_url: imgUrl,
              tag: frame.tag,
              fl_x: frame.fl_x,
              fl_y: frame.fl_y,
              w: frame.w || data.w,
              h: frame.h || data.h
            };
          });
          // 首次加载按拍摄焦距初始化查看相机
          if (sceneMetadata.value.fl_y && sceneMetadata.value.h) {
            applyFocalLengthPx(sceneMetadata.value.fl_y);
          }
        } else {
          cameraPoses.value = data; // 兼容旧格式
        }
      })
      .catch(err => console.error("加载位姿失败:", err));

    const splatMesh = viewer.getSplatMesh();
    splatMesh.visible = false;

    setTimeout(() => {
      if (splatMesh) {
        // 先生成粒子系统，这会计算出 uCenter 和 uMaxRadius
        createParticleSystem(splatMesh);
        // 然后应用 Shader
        applyAdvancedShader(splatMesh);
        
          if (initialPoseMatrix) {
            setTimeout(() => { flyToImage({ matrix: initialPoseMatrix }); }, 50);
          }

        animationState.lastFrameTime = Date.now();
        animationState.startTime = Date.now();
        animationState.isLoaded = true;
      }
    }, 200);
    // --- 5. 动画循环 (120 FPS 上限) ---
    let lastDrawTime = performance.now();
    const fpsInterval = 1000 / 120; // 目标 120 帧
    let framesThisSecond = 0;
    let lastFpsTime = performance.now();
    viewer.renderer.setAnimationLoop(() => {
      const nowPerf = performance.now();
      const elapsedSinceDraw = nowPerf - lastDrawTime;

      // 帧率限制：如果离上一帧不足 1/120 秒，放弃当前帧渲染
      if (elapsedSinceDraw < fpsInterval) return;

      // 更新上一帧时间，这会保留超出的那一点时间以防长期的漂移
      lastDrawTime = nowPerf - (elapsedSinceDraw % fpsInterval);

      viewer.update();
      viewer.render();

      // FPS 计算：累加帧数，如果过了1秒，就更新显示并清零
      framesThisSecond++;
      if (nowPerf - lastFpsTime >= 1000) {
        currentFps.value = framesThisSecond;
        framesThisSecond = 0;
        lastFpsTime = nowPerf;
      }

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
          if (particleSystem) particleSystem.visible = false;
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
    loadError.value = (error && (error.message || String(error))) || '模型加载失败，请检查模型 URL 是否正确可访问';
  } finally {
    isLoading.value = false;
  }
};

const setupDesktopControls = () => {
  if (!viewer) return;
  // 清理现有控制器
  if (viewer.controls) { viewer.controls.dispose(); viewer.controls = null; }

  // [DEBUG] 暂时禁用控制器
  /*
  const controls = new ArcballControls(viewer.camera, viewer.renderer.domElement, viewer.threeScene);
  controls.setGizmosVisible(false);
  controls.enableDamping = true;
  viewer.controls = controls;
  */
  console.log("Controls explicitly disabled for debugging");
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
  refreshCurrentFocalInfo();
};

const onSessionStarted = (session) => {
  isVRMode.value = true;
  if (viewer && viewer.controls) { viewer.controls.dispose(); viewer.controls = null; }
  session.addEventListener('end', onSessionEnded);
};
const onSessionEnded = () => { isVRMode.value = false; setupDesktopControls(); };
const toggleVRMode = async () => {
  if (!isSecureContext.value) { alert("需HTTPS"); return; }
  if (isVRMode.value) { if (viewer.xr) viewer.xr.exitVR(); isVRMode.value = false; }
  else { if (viewer.xr) viewer.xr.enterVR(); isVRMode.value = true; }
};
const toggleAutoRotate = () => { isAutoRotate.value = !isAutoRotate.value; };
const checkProtocol = () => {
  const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  const isHttps = window.location.protocol === 'https:';
  isSecureContext.value = isLocal || isHttps;
};

const isDragging = ref(false);
const lastMouse = { x: 0, y: 0 };
// const rotationDelta removed here

// --- 简单拖拽微调逻辑 ---
const onMouseDown = (e) => {
  isDragging.value = true;
  lastMouse.x = e.clientX;
  lastMouse.y = e.clientY;
};

const onMouseMove = (e) => {
  if (!isDragging.value || !viewer || !viewer.camera) return;

  const dx = e.clientX - lastMouse.x;
  const dy = e.clientY - lastMouse.y;
  const sensitivity = 0.2;

  // 计算增量
  const deltaPitch = dy * sensitivity; // 上下反转: -dy 变成 dy
  const panSensitivity = 0.01; // 平移灵敏度，根据模型大小可能需要调整

  // X轴旋转 (俯仰) - 本地轴
  viewer.camera.rotateX(deltaPitch * Math.PI / 180);

  // 左右平移 (移动视角左右而不是旋转)
  viewer.camera.translateX(-dx * panSensitivity);

  viewer.camera.updateProjectionMatrix();
  updateDebugInfo();

  lastMouse.x = e.clientX;
  lastMouse.y = e.clientY;
};

const onMouseUp = () => { isDragging.value = false; };

// --- 移动端 Touch 事件支持 ---
const onTouchStart = (e) => {
  if (e.touches.length > 0) {
    isDragging.value = true;
    lastMouse.x = e.touches[0].clientX;
    lastMouse.y = e.touches[0].clientY;
  }
};

const onTouchMove = (e) => {
  if (!isDragging.value || !viewer || !viewer.camera || e.touches.length === 0) return;

  const dx = e.touches[0].clientX - lastMouse.x;
  const dy = e.touches[0].clientY - lastMouse.y;
  const sensitivity = 0.2;

  const deltaPitch = dy * sensitivity; // 上下反转: -dy 变成 dy
  const panSensitivity = 0.01; // 平移灵敏度

  rotationDelta.value.x += deltaPitch;

  viewer.camera.rotateX(deltaPitch * Math.PI / 180);
  // 左右平移 (移动视角左右而不是旋转)
  viewer.camera.translateX(-dx * panSensitivity);

  viewer.camera.updateProjectionMatrix();
  updateDebugInfo();

  lastMouse.x = e.touches[0].clientX;
  lastMouse.y = e.touches[0].clientY;
};

const onTouchEnd = () => { isDragging.value = false; };

onMounted(() => {
  if (containerRef.value) {
    checkProtocol();

    // 注册供Flutter调用的全局函数
    // 支持两种调用方式：
    // 1. loadModelFromFlutter(plyUrl)              -- 只传模型URL，兼容旧版
    // 2. loadModelFromFlutter({ply: url, poses: url}) -- 同时传模型和位姿URL
    window.loadModelFromFlutter = (input) => {
      console.log('[Flutter->WebGL] 收到加载请求:', input);
      if (typeof input === 'string') {
        // 旧版兼容：只传了 PLY URL，位姿使用默认本地路径
        initViewer(input, null, null);
      } else if (typeof input === 'object' && input !== null) {
        // 新版：同时传 PLY URL、poses URL 与 初始矩阵
        initViewer(input.ply || null, input.poses || null, input.matrix || null);
      } else {
        initViewer(null, null, null);
      }
    };

    // 通知 Flutter 页面已就绪
    if (window.BrainDanceChannel) {
      window.BrainDanceChannel.postMessage(JSON.stringify({ status: 'ready' }));
    } else {
      // 非 Flutter 环境（浏览器直接打开），用默认本地文件初始化
      initViewer(null, null);
    }

    // 绑定原生事件用于调试拖拽
    window.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  }
});

onBeforeUnmount(async () => {
  window.removeEventListener('mousedown', onMouseDown);
  window.removeEventListener('mousemove', onMouseMove);
  window.removeEventListener('mouseup', onMouseUp);

  if (viewer) {
    viewer.renderer.setAnimationLoop(null);
    await viewer.dispose();
  }
});
</script>

<template>
  <div class="app-container" @mousedown="onMouseDown" @mousemove="onMouseMove" @mouseup="onMouseUp"
    @mouseleave="onMouseUp" @touchstart="onTouchStart" @touchmove.prevent="onTouchMove" @touchend="onTouchEnd"
    @touchcancel="onTouchEnd">
    <div ref="containerRef" class="viewer-container"></div>
    <div v-if="isLoading" class="loading-overlay">正在处理...</div>
    <div class="fps-counter" v-if="currentFps > 0">FPS: {{ currentFps }}</div>
    <div class="controls-ui" v-if="false">
      <button v-if="isSecureContext" @click="toggleVRMode" :class="{ active: isVRMode }">
        {{ isVRMode ? '退出 VR' : '进入 VR' }}
      </button>
      <button @click="toggleAutoRotate" :class="{ active: isAutoRotate }">
        {{ isAutoRotate ? '停止旋转' : '自动旋转' }}
      </button>
    </div>

    <!-- 搜索功能 -->
    <div class="search-panel">
      <input type="text" v-model="searchQuery" @keyup.enter="searchAndFly" placeholder="搜索想要的视角 (如: 正面特写...)"
        class="search-input" />
      <button @click="searchAndFly" class="search-btn">🔍 搜索视角</button>
    </div>

    <button class="focal-settings-toggle" @click="toggleFocalSettings"
      @mousedown.stop @touchstart.stop @touchend.stop>焦距设置</button>
    <div class="focal-settings-panel" v-if="showFocalSettings"
      @mousedown.stop @touchstart.stop @touchmove.stop @touchend.stop @touchcancel.stop>
      <div class="focal-title">镜头焦距</div>
      <input type="range" v-model.number="manualFocalPx" :min="focalMin" :max="focalMax" step="1"
        @input="onManualFocalChange" />
      <div class="focal-row">
        <input class="focal-number-input" type="number" v-model.number="manualFocalPx" :min="focalMin" :max="focalMax"
          step="1" @change="onManualFocalChange" />
        <span>px</span>
      </div>
      <div class="focal-row">
        <span>当前 FOV: {{ currentViewFov.toFixed(1) }}°</span>
      </div>
      <div class="focal-row">
        <span>当前焦距: {{ currentViewFocalPx.toFixed(1) }} px</span>
      </div>
      <button class="focal-reset-btn" @click="resetFocalToCapture">恢复拍摄焦距</button>
    </div>

    <!-- 调试面板 - 已注释 -->
    <!--
    <div class="debug-panel" v-if="cameraPoses.length > 0">
      <div class="debug-title">镜头调试器</div>
      <div class="debug-row">飞越起点: {{ arrivalEuler.x }}, {{ arrivalEuler.y }}, {{ arrivalEuler.z }}</div>
      <div class="debug-row">当前视角: <b>{{ debugInfo.x }}, {{ debugInfo.y }}, {{ debugInfo.z }}</b></div>
      <div class="debug-row" style="color:#ffcc00;">手动修正: <b>X:{{ rotationDelta.x.toFixed(1) }}°, Y:{{ rotationDelta.y.toFixed(1) }}°</b></div>
      <hr style="border:0; border-top:1px solid #333; margin:8px 0;" />
      <div class="debug-title">旋转控制 (Rotation)</div>      <div class="debug-controls">
        <button class="mini-btn" @click="manualRotate('x', 5)">X+5 (俯仰)</button>
        <button class="mini-btn" @click="manualRotate('y', 5)">Y+5 (偏航)</button>
        <button class="mini-btn" @click="manualRotate('z', 5)">Z+5 (滚转)</button>
        <button class="mini-btn" @click="manualRotate('x', -5)">X-5</button>
        <button class="mini-btn" @click="manualRotate('y', -5)">Y-5</button>
        <button class="mini-btn" @click="manualRotate('z', -5)">Z-5</button>
        <button class="mini-btn" @click="manualRotate('x', 90)">X+90</button>
        <button class="mini-btn" @click="manualRotate('y', 90)">Y+90</button>
        <button class="mini-btn" @click="manualRotate('z', 90)">Z+90</button>
        <button class="mini-btn" @click="manualRotate('x', -90)">X-90</button>
        <button class="mini-btn" @click="manualRotate('y', -90)">Y-90</button>
        <button class="mini-btn" @click="manualRotate('z', -90)">Z-90</button>
      </div>

      <div class="debug-title" style="margin-top:10px;">移动控制 (Translation)</div>
      <div class="debug-controls">
        <button class="mini-btn" @click="manualMove('x', 0.1)">X 右移</button>
        <button class="mini-btn" @click="manualMove('y', 0.1)">Y 上移</button>
        <button class="mini-btn" @click="manualMove('z', 0.1)">Z 后退</button>
        <button class="mini-btn" @click="manualMove('x', -0.1)">X 左移</button>
        <button class="mini-btn" @click="manualMove('y', -0.1)">Y 下移</button>
        <button class="mini-btn" @click="manualMove('z', -0.1)">Z 前进</button>
      </div>
      <div style="margin-top:10px; display:flex; gap:5px;">
        <button class="mini-btn" style="flex:1; border-color:#0f0;" @click="copyMatrixToClipboard">复制最终矩阵</button>
      </div>
      <div style="font-size:9px; color:#666; margin-top:5px;">对齐后点击复制，发送给我</div>
    </div>
    -->

    <!-- 镜头轨道小功能 -->
    <div class="camera-track" v-if="filteredPoses.length > 0" @mousedown.stop @touchstart.stop @touchmove.stop
      @touchend.stop>
      <div v-for="(pose, index) in filteredPoses" :key="pose.id" class="camera-btn"
        :class="{ active: activeImage === pose.image_url }" @click.stop="flyToImage(pose)">
        <img v-if="pose.image_url" :src="pose.image_url" class="btn-thumb" />
        <div v-if="pose.tag" class="camera-tag-overlay">
          <div class="camera-title-mini">镜 {{ pose.id.split('.')[0].replace('frame_', '') }}</div>
          <div class="camera-tag-text">{{ pose.tag }}</div>
        </div>
        <span v-else-if="!pose.image_url">镜头 {{ index + 1 }}</span>
      </div>
    </div>

    <!-- 参考图对比悬浮窗 -->
    <div class="reference-overlay" v-if="activeImage" @click="activeImage = ''; activeTag = ''">
      <div class="ref-title">参考原图</div>
      <img :src="activeImage" class="ref-img" />
      <div class="ref-info" v-if="activeTag">
        <span class="info-tag" style="color: #4CAF50;">{{ activeTag }}</span>
      </div>
      <div class="ref-info" v-if="sceneMetadata.fl_y">
        <span class="info-tag">焦距: {{ (sceneMetadata.fl_y).toFixed(1) }} px</span>
        <span class="info-tag">FOV: {{ (2 * Math.atan(sceneMetadata.h / (2 * sceneMetadata.fl_y)) * (180 /
          Math.PI)).toFixed(1) }}°</span>
        <span class="info-tag">分辨率: {{ sceneMetadata.w }}x{{ sceneMetadata.h }}</span>
      </div>
      <div class="ref-hint">点击关闭对比</div>
    </div>
  </div>
</template>

<style scoped>
.app-container {
  position: relative;
  width: 100vw;
  height: 100vh;
  background-color: #000000;
  overflow: hidden;
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
}

.loading-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.8);
  color: white;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 200;
  font-size: 20px;
}

.error-overlay {
  position: absolute;
  inset: 0;
  background: rgba(10, 10, 10, 0.92);
  color: white;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  z-index: 210;
  padding: 24px;
  text-align: center;
}

.error-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

.error-title {
  font-size: 20px;
  font-weight: bold;
  margin-bottom: 8px;
  color: #ff6b6b;
}

.error-msg {
  font-size: 13px;
  color: #ccc;
  max-width: 320px;
  word-break: break-all;
  margin-bottom: 20px;
}

.error-retry {
  background: #333;
  color: white;
  border: 1px solid #555;
  padding: 8px 24px;
  border-radius: 20px;
  cursor: pointer;
  font-size: 14px;
}

.error-retry:hover {
  background: #555;
}

button {
  background: rgba(0, 0, 0, 0.6);
  color: white;
  border: 1px solid rgba(255, 255, 255, 0.3);
  padding: 10px 20px;
  border-radius: 20px;
  cursor: pointer;
  transition: 0.3s;
}

button.active {
  background: #71838F;
  border-color: #71838F;
}

/* 镜头轨道样式 */
.camera-track {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 16px;
  overflow-x: auto;
  max-width: 90vw;
  padding: 16px 20px;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(12px);
  border-radius: 16px;
  z-index: 100;
  border: 1px solid rgba(255, 255, 255, 1);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
}

.camera-btn {
  width: 100px;
  height: 70px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 12px;
  cursor: pointer;
  overflow: hidden;
  border: 2px solid transparent;
  transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #333;
  position: relative;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
}

.camera-btn.active {
  border-color: #71838F;
  transform: translateY(-4px);
  box-shadow: 0 8px 16px rgba(113, 131, 143, 0.25);
}

.btn-thumb {
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0.85;
}

.camera-btn:hover .btn-thumb {
  opacity: 1;
}

.camera-btn.active .btn-thumb {
  opacity: 1;
}

/* 悬浮标签文字 */
.camera-tag-overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(4px);
  color: #fff;
  display: flex;
  flex-direction: column;
  padding: 4px 0;
  align-items: center;
  pointer-events: none;
}

.camera-title-mini {
  font-size: 10px;
  opacity: 0.8;
}

.camera-tag-text {
  font-size: 12px;
  font-weight: bold;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 90%;
}

/* 搜索栏样式 */
.search-panel {
  position: absolute;
  top: 80px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 10px;
  z-index: 100;
  background: rgba(0, 0, 0, 0.6);
  padding: 10px;
  border-radius: 12px;
  backdrop-filter: blur(10px);
}

.search-input {
  width: 250px;
  padding: 10px 15px;
  border: none;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.9);
  outline: none;
  font-size: 14px;
}

.search-btn {
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  background: #71838F;
  color: white;
  cursor: pointer;
  font-weight: bold;
}

.search-btn:hover {
  background: #5A6A74;
}

.focal-settings-toggle {
  position: absolute;
  top: 20px;
  right: 20px;
  z-index: 120;
  padding: 8px 14px;
  border-radius: 10px;
  background: rgba(0, 0, 0, 0.65);
}

.focal-settings-panel {
  position: absolute;
  top: 62px;
  right: 20px;
  z-index: 120;
  width: 220px;
  background: rgba(0, 0, 0, 0.75);
  color: #fff;
  border: 1px solid rgba(255, 255, 255, 0.25);
  border-radius: 10px;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.focal-title {
  font-size: 13px;
  font-weight: 700;
}

.focal-row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}

.focal-number-input {
  width: 100px;
  border-radius: 6px;
  border: 1px solid #555;
  padding: 4px 6px;
  background: rgba(255, 255, 255, 0.95);
}

.focal-reset-btn {
  border-radius: 8px;
  padding: 6px 10px;
  background: #71838F;
  border: none;
}

/* 参考图浮窗 */
.reference-overlay {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  right: 16px;
  width: 28vw;
  min-width: 110px;
  max-width: 180px;
  background: rgba(0, 0, 0, 0.7);
  padding: 8px;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  z-index: 150;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.ref-title {
  font-size: 10px;
  color: #aaa;
  margin-bottom: 6px;
  text-align: center;
}

.ref-img {
  width: 100%;
  border-radius: 4px;
  border: 1px solid #444;
  margin-bottom: 6px;
}

.ref-info {
  font-size: 9px;
  color: #ddd;
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  justify-content: center;
  margin-bottom: 4px;
}

.info-tag {
  background: rgba(255, 255, 255, 0.1);
  padding: 2px 4px;
  border-radius: 4px;
}

.ref-hint {
  font-size: 8px;
  color: #666;
  text-align: center;
  margin-top: 4px;
}

/* 调试面板 */
.debug-panel {
  position: absolute;
  top: 100px;
  right: 320px;
  background: rgba(0, 0, 0, 0.8);
  padding: 10px;
  border-radius: 8px;
  color: #0f0;
  font-family: monospace;
  font-size: 11px;
  z-index: 150;
  border: 1px solid #33cc33;
}

.debug-title {
  margin-bottom: 5px;
  color: #fff;
  font-weight: bold;
  font-size: 13px;
}

.debug-row {
  margin-bottom: 4px;
  color: #0f0;
}

.debug-controls {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 4px;
}

.mini-btn {
  padding: 4px;
  font-size: 10px;
  border-radius: 4px;
  border: 1px solid #444;
  background: #222;
  color: white;
  cursor: pointer;
}

.mini-btn:hover {
  background: #444;
}

/* FPS 计数器 */
.fps-counter {
  position: absolute;
  top: 10px;
  left: 10px;
  color: #d8f4ff;
  background: rgba(0, 0, 0, 0.55);
  border: 1px solid rgba(216, 244, 255, 0.35);
  border-radius: 6px;
  padding: 3px 7px;
  font-family: monospace;
  font-size: 12px;
  z-index: 1000;
  pointer-events: none;
}
</style>
