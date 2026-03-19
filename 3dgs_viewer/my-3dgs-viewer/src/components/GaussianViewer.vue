<script setup>
import { onMounted, onBeforeUnmount, ref, computed } from 'vue';
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import gsap from 'gsap';

const containerRef = ref(null);
const isVRMode = ref(false);
const isAutoRotate = ref(false);
const isLoading = ref(false);
const isSecureContext = ref(false);
const VIEW_MODE = {
  FREE: 'free',
  ORBIT: 'orbit'
};
const currentViewMode = ref(VIEW_MODE.FREE);
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
const DEFAULT_FOCAL_PX = 380; // 无位姿元数据时使用更广一点的默认焦距
const DRAG_ROTATE_SENSITIVITY = 0.065;
const DRAG_PAN_SENSITIVITY = 0.0022;
const WHEEL_ZOOM_STEP = 0.08;
const PINCH_ZOOM_STEP = 1.0;
const ORBIT_YAW_SENSITIVITY = 0.0055;
const ORBIT_PITCH_SENSITIVITY = 0.0042;
const ORBIT_ROLL_SENSITIVITY = 1.0;
const ORBIT_DOLLY_FACTOR = 0.35;

const isOrbitMode = computed(() => currentViewMode.value === VIEW_MODE.ORBIT);

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
const worldUp = new THREE.Vector3(0, 1, 0);
let pendingInitialTarget = null;
let didApplyInitialTarget = false;
let posesFetchSettled = false;

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

const clampFocalPx = (focalPx) => {
  if (!Number.isFinite(focalPx)) return null;
  return Math.min(focalMax.value, Math.max(focalMin.value, focalPx));
};

const getActiveFocalPx = () => {
  const focal = Number(
    manualFocalPx.value || currentViewFocalPx.value || sceneMetadata.value.fl_y || DEFAULT_FOCAL_PX
  );
  return clampFocalPx(focal);
};

const zoomByFocalScale = (scaleFactor) => {
  if (!viewer || !viewer.camera || !Number.isFinite(scaleFactor) || scaleFactor <= 0) return;
  const currentFocal = getActiveFocalPx();
  if (!currentFocal) return;

  const nextFocal = clampFocalPx(currentFocal * scaleFactor);
  if (!nextFocal) return;

  manualFocalPx.value = Number(nextFocal.toFixed(1));
  applyFocalLengthPx(nextFocal);
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
      (currentViewFocalPx.value || sceneMetadata.value.fl_y || DEFAULT_FOCAL_PX).toFixed(1)
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

const getModelWorldCenter = () => globalUniforms.uCenter.value.clone();

const syncOrbitTarget = () => {};

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

const normalizeMatrixArray = (input) => {
  if (!Array.isArray(input)) return null;

  if (input.length === 16) {
    const flat = input.map(value => Number(value));
    return flat.every(Number.isFinite) ? flat : null;
  }

  if (input.length === 4 && input.every(row => Array.isArray(row) && row.length === 4)) {
    const flat = input.flat().map(value => Number(value));
    return flat.every(Number.isFinite) ? flat : null;
  }

  return null;
};

const normalizeImageId = (value) => {
  if (value == null) return '';

  let text = String(value).trim();
  if (!text) return '';

  try {
    text = decodeURIComponent(text);
  } catch (_) {}

  text = text.replace(/\\/g, '/');
  const parts = text.split('/');
  return (parts[parts.length - 1] || '').trim().toLowerCase();
};

const getPoseImageId = (pose) => {
  if (!pose) return '';
  const directId = pose.id || pose.image_id || pose.imageId;
  if (directId) return normalizeImageId(directId);

  const imageUrl = pose.image_url;
  if (typeof imageUrl !== 'string' || imageUrl.length === 0) return '';

  const cleanUrl = imageUrl.split('?')[0];
  return normalizeImageId(cleanUrl);
};

const findPoseByInitialTarget = (target) => {
  if (!target || cameraPoses.value.length === 0) return null;

  const targetImageId = normalizeImageId(target.imageId);
  if (targetImageId) {
    const matchedPose = cameraPoses.value.find((pose) => getPoseImageId(pose) === targetImageId);
    if (matchedPose) return matchedPose;
  }

  const targetMatrix = normalizeMatrixArray(target.matrix);
  if (!targetMatrix) return null;

  let bestPose = null;
  let bestDiff = Number.POSITIVE_INFINITY;

  for (const pose of cameraPoses.value) {
    const poseMatrix = normalizeMatrixArray(pose.matrix);
    if (!poseMatrix) continue;

    let maxDiff = 0;
    for (let i = 0; i < 16; i += 1) {
      const diff = Math.abs(poseMatrix[i] - targetMatrix[i]);
      if (diff > maxDiff) maxDiff = diff;
      if (maxDiff >= bestDiff) break;
    }

    if (maxDiff < bestDiff) {
      bestDiff = maxDiff;
      bestPose = pose;
    }
  }

  return bestDiff <= 1e-4 ? bestPose : null;
};

const maybeApplyInitialTarget = (forceFallback = false) => {
  if (!pendingInitialTarget || didApplyInitialTarget) return;

  const resolvedPose = findPoseByInitialTarget(pendingInitialTarget);
  if (resolvedPose) {
    didApplyInitialTarget = true;
    flyToImage(resolvedPose);
    return;
  }

  if (!forceFallback) return;
  if (pendingInitialTarget.imageId && !posesFetchSettled) return;

  const fallbackMatrix = normalizeMatrixArray(pendingInitialTarget.matrix);
  if (!fallbackMatrix) return;

  didApplyInitialTarget = true;
  flyToImage({
    matrix: fallbackMatrix,
    image_url: pendingInitialTarget.imageId || '',
  });
};

// --- 5. 初始化 ---
const flyToImage = (poseData) => {
  if (!viewer || !viewer.camera) return;
  const normalizedMatrix = normalizeMatrixArray(poseData?.matrix);
  if (!normalizedMatrix) {
    console.warn('[Viewer] Skip invalid pose matrix:', poseData);
    return;
  }

  const cam = viewer.camera;
  const splatMesh = viewer.getSplatMesh(); // 获取当前加载的高斯模型

  // 更新参考图
  activeImage.value = poseData.image_url || getPoseImageId(poseData);
  activeTag.value = poseData.tag || '';

  // 1. 读取原始矩阵 (假设后端传来的是按列优先的 16 位数组)
  const rawMatrix = new THREE.Matrix4().fromArray(normalizedMatrix);

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
      orbitTouchState.roll = 0;
      updateDebugInfo();

      if (viewer.controls) viewer.controls.enabled = true;
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
let hasInitializedFromExternalInput = false;

const parseInitialInputFromUrl = () => {
  const params = new URLSearchParams(window.location.search);
  const payload = params.get('payload');
  if (payload) {
    try {
      const decoded = JSON.parse(decodeURIComponent(payload));
      return {
        ply: decoded.ply || null,
        poses: decoded.poses || null,
        matrix: decoded.matrix || null,
        imageId: decoded.imageId || null
      };
    } catch (error) {
      console.warn('[Viewer] 无法解析 payload 查询参数:', error);
    }
  }

  const ply = params.get('ply');
  const poses = params.get('poses');
  const matrix = params.get('matrix');
  const imageId = params.get('imageId');

  let parsedMatrix = null;
  if (matrix) {
    try {
      parsedMatrix = JSON.parse(decodeURIComponent(matrix));
    } catch (error) {
      console.warn('[Viewer] 无法解析 matrix 查询参数:', error);
    }
  }

  if (ply || poses || parsedMatrix) {
    return {
      ply: ply || null,
      poses: poses || null,
      matrix: parsedMatrix,
      imageId: imageId || null
    };
  }

  return null;
};

const initViewer = async (plyUrl, posesUrl, initialTarget) => {
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
    pendingInitialTarget = null;
    didApplyInitialTarget = false;
    posesFetchSettled = false;

    const config = getViewerConfig();
    viewer = new GaussianSplats3D.Viewer(config);
    window.viewer = viewer;
    manualFocalPx.value = DEFAULT_FOCAL_PX;

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
        posesFetchSettled = true;
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
          } else {
            applyFocalLengthPx(DEFAULT_FOCAL_PX);
          }
          maybeApplyInitialTarget(true);
        } else {
          cameraPoses.value = data; // 兼容旧格式
          applyFocalLengthPx(DEFAULT_FOCAL_PX);
          maybeApplyInitialTarget(true);
        }
      })
      .catch(err => {
        posesFetchSettled = true;
        console.error("加载位姿失败:", err);
        applyFocalLengthPx(DEFAULT_FOCAL_PX);
        maybeApplyInitialTarget(true);
      });

    const splatMesh = viewer.getSplatMesh();
    splatMesh.visible = false;

    setTimeout(() => {
      if (splatMesh) {
        // 先生成粒子系统，这会计算出 uCenter 和 uMaxRadius
        createParticleSystem(splatMesh);
        // 然后应用 Shader
        applyAdvancedShader(splatMesh);
        
          if (initialTarget && (initialTarget.matrix || initialTarget.imageId)) {
            pendingInitialTarget = {
              matrix: initialTarget.matrix || null,
              imageId: initialTarget.imageId || null
            };
            maybeApplyInitialTarget(posesFetchSettled);
            setTimeout(() => { maybeApplyInitialTarget(false); }, 50);
            if (!initialTarget.imageId) {
              setTimeout(() => { maybeApplyInitialTarget(true); }, 800);
            }
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

    applyViewMode();

  } catch (error) {
    console.error("error:", error);
    loadError.value = (error && (error.message || String(error))) || '模型加载失败，请检查模型 URL 是否正确可访问';
  } finally {
    isLoading.value = false;
  }
};

const disposeControls = () => {
  if (!viewer || !viewer.controls) return;
  viewer.controls.dispose();
  viewer.controls = null;
};

const renderCameraUpdate = () => {
  if (!viewer || !viewer.camera) return;
  viewer.camera.updateProjectionMatrix();
  refreshCurrentFocalInfo();
  updateDebugInfo();
  try { viewer.update(); viewer.render(); } catch (_) {}
};

const orbitRotate = (deltaYaw, deltaPitch) => {
  if (!viewer || !viewer.camera) return;
  viewer.camera.rotateOnWorldAxis(worldUp, -deltaYaw);
  viewer.camera.rotateX(-deltaPitch);
  renderCameraUpdate();
};

const orbitRoll = (deltaAngleRad) => {
  if (!viewer || !viewer.camera || !Number.isFinite(deltaAngleRad)) return;
  viewer.camera.rotateZ(deltaAngleRad * ORBIT_ROLL_SENSITIVITY);
  renderCameraUpdate();
};

const orbitZoom = (zoomFactor) => {
  if (!viewer || !viewer.camera || !Number.isFinite(zoomFactor) || zoomFactor <= 0) return;
  const sceneDistance = Math.max(0.3, viewer.camera.position.distanceTo(getModelWorldCenter()));
  const deltaZ = THREE.MathUtils.clamp(
    (1 - zoomFactor) * sceneDistance * ORBIT_DOLLY_FACTOR,
    -sceneDistance * 0.25,
    sceneDistance * 0.25
  );
  viewer.camera.translateZ(deltaZ);
  renderCameraUpdate();
};

const getTouchAngle = (touchA, touchB) => {
  return Math.atan2(touchB.clientY - touchA.clientY, touchB.clientX - touchA.clientX);
};

const normalizeTouchAngleDelta = (delta) => {
  if (delta > Math.PI) return delta - (Math.PI * 2);
  if (delta < -Math.PI) return delta + (Math.PI * 2);
  return delta;
};

const setupFreeControls = () => {
  if (!viewer) return;
  disposeControls();
};

const setupOrbitControls = () => {
  if (!viewer) return;
  disposeControls();
  orbitTouchState.roll = 0;
};

const applyViewMode = () => {
  if (!viewer) return;

  if (isOrbitMode.value) {
    setupOrbitControls();
  } else {
    setupFreeControls();
  }
};

const switchViewMode = (mode) => {
  if (mode !== VIEW_MODE.FREE && mode !== VIEW_MODE.ORBIT) return;
  if (currentViewMode.value === mode) return;

  currentViewMode.value = mode;
  applyViewMode();

  if (isOrbitMode.value) {
    syncOrbitTarget();
  }
};

// 修改后的 adjustControlsToModel，直接使用预计算好的值
const adjustControlsToModel = () => {
  if (isVRMode.value) return;

  // createParticleSystem 已经计算了最准确的 uCenter 和 uMaxRadius，直接用
  const worldCenter = globalUniforms.uCenter.value;
  const maxDim = globalUniforms.uMaxRadius.value / 0.7; // 还原回实际尺寸估计
  const distance = maxDim * 2.0;

  viewer.camera.position.set(worldCenter.x, worldCenter.y, worldCenter.z + distance);
  viewer.camera.lookAt(worldCenter);
  syncOrbitTarget(worldCenter);
  refreshCurrentFocalInfo();
};

const onSessionStarted = (session) => {
  isVRMode.value = true;
  if (viewer && viewer.controls) { viewer.controls.dispose(); viewer.controls = null; }
  session.addEventListener('end', onSessionEnded);
};
const onSessionEnded = () => { isVRMode.value = false; applyViewMode(); };
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
const pinchState = {
  active: false,
  distance: 0
};
const orbitTouchState = {
  active: false,
  angle: 0,
  roll: 0
};
// const rotationDelta removed here

const getTouchDistance = (touchA, touchB) => {
  const dx = touchA.clientX - touchB.clientX;
  const dy = touchA.clientY - touchB.clientY;
  return Math.hypot(dx, dy);
};

// --- 简单拖拽微调逻辑 ---
const onMouseDown = (e) => {
  if (isOrbitMode.value) {
    if (e.button !== 0) return;
    isDragging.value = true;
    pinchState.active = false;
    orbitTouchState.active = false;
    lastMouse.x = e.clientX;
    lastMouse.y = e.clientY;
    return;
  }
  isDragging.value = true;
  pinchState.active = false;
  lastMouse.x = e.clientX;
  lastMouse.y = e.clientY;
};

const onMouseMove = (e) => {
  if (isOrbitMode.value) {
    if (!isDragging.value || !viewer || !viewer.camera) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    orbitRotate(dx * ORBIT_YAW_SENSITIVITY, dy * ORBIT_PITCH_SENSITIVITY);
    lastMouse.x = e.clientX;
    lastMouse.y = e.clientY;
    return;
  }
  if (!isDragging.value || !viewer || !viewer.camera) return;

  const dx = e.clientX - lastMouse.x;
  const dy = e.clientY - lastMouse.y;

  // 计算增量
  const deltaPitch = dy * DRAG_ROTATE_SENSITIVITY;

  // X轴旋转 (俯仰) - 本地轴
  viewer.camera.rotateX(deltaPitch * Math.PI / 180);

  // 左右平移 (移动视角左右而不是旋转)
  viewer.camera.translateX(-dx * DRAG_PAN_SENSITIVITY);

  viewer.camera.updateProjectionMatrix();
  updateDebugInfo();

  lastMouse.x = e.clientX;
  lastMouse.y = e.clientY;
};

const onMouseUp = () => {
  if (isOrbitMode.value) {
    isDragging.value = false;
    pinchState.active = false;
    orbitTouchState.active = false;
    return;
  }
  isDragging.value = false;
  pinchState.active = false;
};

const onWheel = (e) => {
  if (!viewer || !viewer.camera) return;
  if (isOrbitMode.value) {
    const zoomFactor = e.deltaY < 0 ? (1 + WHEEL_ZOOM_STEP) : (1 / (1 + WHEEL_ZOOM_STEP));
    orbitZoom(zoomFactor);
    return;
  }
  const direction = e.deltaY < 0 ? (1 + WHEEL_ZOOM_STEP) : (1 / (1 + WHEEL_ZOOM_STEP));
  zoomByFocalScale(direction);
};

// --- 移动端 Touch 事件支持 ---
const onTouchStart = (e) => {
  if (isOrbitMode.value) {
    if (e.touches.length >= 2) {
      isDragging.value = false;
      pinchState.active = true;
      pinchState.distance = getTouchDistance(e.touches[0], e.touches[1]);
      orbitTouchState.active = true;
      orbitTouchState.angle = getTouchAngle(e.touches[0], e.touches[1]);
      return;
    }

    pinchState.active = false;
    orbitTouchState.active = false;
    if (e.touches.length === 1) {
      isDragging.value = true;
      lastMouse.x = e.touches[0].clientX;
      lastMouse.y = e.touches[0].clientY;
    }
    return;
  }
  if (e.touches.length >= 2) {
    isDragging.value = false;
    pinchState.active = true;
    pinchState.distance = getTouchDistance(e.touches[0], e.touches[1]);
    return;
  }

  pinchState.active = false;
  if (e.touches.length === 1) {
    isDragging.value = true;
    lastMouse.x = e.touches[0].clientX;
    lastMouse.y = e.touches[0].clientY;
  }
};

const onTouchMove = (e) => {
  if (isOrbitMode.value) {
    if (!viewer || !viewer.camera || e.touches.length === 0) return;

    if (e.touches.length >= 2) {
      const nextDistance = getTouchDistance(e.touches[0], e.touches[1]);
      const nextAngle = getTouchAngle(e.touches[0], e.touches[1]);

      if (pinchState.active && pinchState.distance > 0 && nextDistance > 0) {
        orbitZoom(nextDistance / pinchState.distance);
      }
      if (orbitTouchState.active) {
        orbitRoll(normalizeTouchAngleDelta(nextAngle - orbitTouchState.angle));
      }

      pinchState.active = true;
      pinchState.distance = nextDistance;
      orbitTouchState.active = true;
      orbitTouchState.angle = nextAngle;
      isDragging.value = false;
      return;
    }

    if (!isDragging.value) return;

    const dx = e.touches[0].clientX - lastMouse.x;
    const dy = e.touches[0].clientY - lastMouse.y;
    orbitRotate(dx * ORBIT_YAW_SENSITIVITY, dy * ORBIT_PITCH_SENSITIVITY);
    lastMouse.x = e.touches[0].clientX;
    lastMouse.y = e.touches[0].clientY;
    return;
  }
  if (!viewer || !viewer.camera || e.touches.length === 0) return;

  if (e.touches.length >= 2) {
    const nextDistance = getTouchDistance(e.touches[0], e.touches[1]);
    if (pinchState.active && pinchState.distance > 0 && nextDistance > 0) {
      const scale = nextDistance / pinchState.distance;
      zoomByFocalScale(1 + ((scale - 1) * PINCH_ZOOM_STEP));
    }
    pinchState.active = true;
    pinchState.distance = nextDistance;
    isDragging.value = false;
    return;
  }

  if (!isDragging.value) return;

  const dx = e.touches[0].clientX - lastMouse.x;
  const dy = e.touches[0].clientY - lastMouse.y;

  const deltaPitch = dy * DRAG_ROTATE_SENSITIVITY;

  rotationDelta.value.x += deltaPitch;

  viewer.camera.rotateX(deltaPitch * Math.PI / 180);
  // 左右平移 (移动视角左右而不是旋转)
  viewer.camera.translateX(-dx * DRAG_PAN_SENSITIVITY);

  viewer.camera.updateProjectionMatrix();
  updateDebugInfo();

  lastMouse.x = e.touches[0].clientX;
  lastMouse.y = e.touches[0].clientY;
};

const onTouchEnd = (e) => {
  if (isOrbitMode.value) {
    if (e.touches.length >= 2) {
      pinchState.active = true;
      pinchState.distance = getTouchDistance(e.touches[0], e.touches[1]);
      orbitTouchState.active = true;
      orbitTouchState.angle = getTouchAngle(e.touches[0], e.touches[1]);
      isDragging.value = false;
      return;
    }

    pinchState.active = false;
    pinchState.distance = 0;
    orbitTouchState.active = false;
    orbitTouchState.angle = 0;
    isDragging.value = false;

    if (e.touches.length === 1) {
      lastMouse.x = e.touches[0].clientX;
      lastMouse.y = e.touches[0].clientY;
      isDragging.value = true;
    }
    return;
  }
  if (e.touches.length >= 2) {
    pinchState.active = true;
    pinchState.distance = getTouchDistance(e.touches[0], e.touches[1]);
    isDragging.value = false;
    return;
  }

  pinchState.active = false;
  pinchState.distance = 0;
  isDragging.value = false;

  if (e.touches.length === 1) {
    lastMouse.x = e.touches[0].clientX;
    lastMouse.y = e.touches[0].clientY;
    isDragging.value = true;
  }
};

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
        // 新版：同时传 PLY URL、poses URL 与 初始目标
        initViewer(input.ply || null, input.poses || null, {
          matrix: input.matrix || null,
          imageId: input.imageId || null
        });
      } else {
        initViewer(null, null, null);
      }
    };

    // 通知 Flutter 页面已就绪
    if (window.BrainDanceChannel) {
      window.BrainDanceChannel.postMessage(JSON.stringify({ status: 'ready' }));
    } else {
      // 非 Flutter 环境（浏览器直接打开），优先使用 URL 参数启动
      const initialInput = parseInitialInputFromUrl();
      if (initialInput && !hasInitializedFromExternalInput) {
        hasInitializedFromExternalInput = true;
        initViewer(initialInput.ply, initialInput.poses, {
          matrix: initialInput.matrix || null,
          imageId: initialInput.imageId || null
        });
      } else {
        initViewer(null, null);
      }
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
    @wheel.prevent="onWheel"
    @mouseleave="onMouseUp" @touchstart="onTouchStart" @touchmove.prevent="onTouchMove" @touchend="onTouchEnd"
    @touchcancel="onTouchEnd">
    <div ref="containerRef" class="viewer-container"></div>
    <div class="viewer-vignette"></div>

    <div class="top-hud">
      <div class="search-panel archive-card" @mousedown.stop @touchstart.stop @touchmove.stop @touchend.stop>
        <input type="text" v-model="searchQuery" @keyup.enter="searchAndFly" placeholder="例如：门口、桌面左侧、正面特写"
          class="search-input" />
        <button @click="searchAndFly" class="archive-btn archive-btn--solid search-btn">检索视角</button>
      </div>

      <div class="top-actions">
        <div class="view-mode-switch archive-card" @mousedown.stop @touchstart.stop @touchmove.stop @touchend.stop>
          <button class="mode-chip" :class="{ active: currentViewMode === VIEW_MODE.FREE }"
            @click="switchViewMode(VIEW_MODE.FREE)">
            自由模式
          </button>
          <button class="mode-chip" :class="{ active: currentViewMode === VIEW_MODE.ORBIT }"
            @click="switchViewMode(VIEW_MODE.ORBIT)">
            Orbit 模式
          </button>
        </div>
        <button class="archive-btn archive-btn--ghost focal-settings-toggle" @click="toggleFocalSettings"
          @mousedown.stop @touchstart.stop @touchend.stop>
          {{ showFocalSettings ? '收起焦距' : '焦距设置' }}
        </button>
        <div class="fps-counter" v-if="currentFps > 0">FPS {{ currentFps }}</div>
      </div>
    </div>

    <div v-if="isLoading" class="loading-overlay">
      <div class="loading-card">
        <div class="loading-dot"></div>
        <div class="loading-title">场景正在展开</div>
        <div class="loading-copy">模型与参考镜头正在同步到工作台。</div>
      </div>
    </div>

    <div v-if="loadError" class="error-overlay">
      <div class="error-card">
        <div class="eyebrow">Load Failed</div>
        <div class="error-title">模型未能正常打开</div>
        <div class="error-msg">{{ loadError }}</div>
        <button class="archive-btn archive-btn--solid" @click="initViewer(currentPlyUrl, currentPosesUrl, null)">
          重新载入
        </button>
      </div>
    </div>

    <div class="controls-ui" v-if="false">
      <button v-if="isSecureContext" @click="toggleVRMode" :class="{ active: isVRMode }">
        {{ isVRMode ? '退出 VR' : '进入 VR' }}
      </button>
      <button @click="toggleAutoRotate" :class="{ active: isAutoRotate }">
        {{ isAutoRotate ? '停止旋转' : '自动旋转' }}
      </button>
    </div>

    <div class="focal-settings-panel" v-if="showFocalSettings"
      @mousedown.stop @touchstart.stop @touchmove.stop @touchend.stop @touchcancel.stop>
      <div class="eyebrow">Lens Control</div>
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
      <button class="archive-btn archive-btn--solid focal-reset-btn" @click="resetFocalToCapture">恢复拍摄焦距</button>
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
    <div class="camera-track" v-if="!isOrbitMode && filteredPoses.length > 0" @mousedown.stop @touchstart.stop @touchmove.stop
      @touchend.stop>
      <div class="camera-track-header">
        <div class="eyebrow">Shot Strip</div>
        <div class="camera-track-copy">{{ searchQuery ? '按当前检索结果排序' : '优先显示已打标签镜头' }}</div>
      </div>
      <div v-for="(pose, index) in filteredPoses" :key="pose.id" class="camera-btn"
        :class="{ active: activeImage === pose.image_url }" @click.stop="flyToImage(pose)">
        <img v-if="pose.image_url" :src="pose.image_url" class="btn-thumb" />
        <div v-if="pose.tag" class="camera-tag-overlay">
          <div class="camera-tag-text">{{ pose.tag }}</div>
        </div>
        <span v-else-if="!pose.image_url">未命名视角</span>
      </div>
    </div>

    <!-- 参考图对比悬浮窗 -->
    <div class="reference-overlay" v-if="activeImage" @click="activeImage = ''; activeTag = ''">
      <div class="eyebrow">Reference Still</div>
      <div class="ref-title">参考原图</div>
      <img :src="activeImage" class="ref-img" />
      <div class="ref-info" v-if="activeTag">
        <span class="info-tag info-tag--accent">{{ activeTag }}</span>
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
  background:
    radial-gradient(circle at top left, rgba(228, 232, 237, 0.16), transparent 24%),
    radial-gradient(circle at top right, rgba(107, 122, 143, 0.14), transparent 28%),
    linear-gradient(180deg, #f4f3ee 0%, #e6e3db 100%);
  overflow: hidden;
  color: #1e1e20;
  font-family: 'HarmonyOS Sans SC', 'Microsoft YaHei', 'PingFang SC', sans-serif;
}

.viewer-container {
  width: 100%;
  height: 100%;
}

.viewer-vignette {
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(180deg, rgba(30, 30, 32, 0.12), transparent 18%, transparent 78%, rgba(30, 30, 32, 0.2)),
    radial-gradient(circle at center, transparent 55%, rgba(30, 30, 32, 0.14) 100%);
  z-index: 1;
}

.eyebrow {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #6b7a8f;
}

.archive-card {
  background: rgba(249, 249, 248, 0.84);
  border: 1px solid rgba(107, 122, 143, 0.16);
  border-radius: 22px;
  box-shadow: 0 10px 26px rgba(0, 0, 0, 0.06);
  backdrop-filter: blur(18px);
}

.top-hud {
  position: absolute;
  top: 18px;
  left: 18px;
  right: 18px;
  z-index: 120;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 12px;
}

.top-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 0 0 auto;
  align-self: flex-end;
  justify-content: flex-end;
  flex-wrap: wrap;
}

.view-mode-switch {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px;
  border-radius: 18px;
}

.mode-chip {
  appearance: none;
  border: 0;
  background: transparent;
  color: #6b7280;
  padding: 8px 12px;
  border-radius: 12px;
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.2s ease, color 0.2s ease, transform 0.2s ease;
}

.mode-chip.active {
  background: #1e1e20;
  color: #f5f4ef;
  box-shadow: 0 8px 18px rgba(30, 30, 32, 0.16);
}

.mode-chip:not(.active):hover {
  background: rgba(107, 122, 143, 0.12);
  color: #273142;
}

.archive-btn {
  appearance: none;
  border-radius: 14px;
  border: 1px solid rgba(107, 122, 143, 0.2);
  padding: 10px 14px;
  cursor: pointer;
  transition:
    transform 180ms ease-out,
    background-color 180ms ease-out,
    border-color 180ms ease-out,
    box-shadow 180ms ease-out;
  font-size: 13px;
  font-weight: 600;
}

.archive-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 18px rgba(0, 0, 0, 0.06);
}

.archive-btn--ghost {
  background: rgba(249, 249, 248, 0.84);
  color: #1e1e20;
}

.archive-btn--solid {
  background: #6b7a8f;
  border-color: #6b7a8f;
  color: #f9f9f8;
}

.archive-btn--solid:hover {
  background: #5e6d81;
  border-color: #5e6d81;
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
  background: rgba(30, 30, 32, 0.24);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 200;
  backdrop-filter: blur(8px);
}

.loading-card,
.error-card {
  min-width: min(84vw, 320px);
  padding: 22px 20px;
  border-radius: 24px;
  background: rgba(249, 249, 248, 0.92);
  border: 1px solid rgba(107, 122, 143, 0.18);
  box-shadow: 0 18px 34px rgba(0, 0, 0, 0.08);
  text-align: center;
}

.loading-dot {
  width: 12px;
  height: 12px;
  margin: 0 auto 14px;
  border-radius: 999px;
  background: #6d8260;
  box-shadow: 0 0 0 10px rgba(109, 130, 96, 0.12);
  animation: pulse 1.8s ease-in-out infinite;
}

.loading-title {
  font-size: 20px;
  font-weight: 600;
}

.loading-copy {
  margin-top: 6px;
  font-size: 13px;
  color: rgba(30, 30, 32, 0.66);
}

.error-overlay {
  position: absolute;
  inset: 0;
  background: rgba(30, 30, 32, 0.18);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 210;
  padding: 24px;
  backdrop-filter: blur(10px);
}

.error-title {
  font-size: 20px;
  font-weight: 600;
  margin: 8px 0;
  color: #8b4747;
}

.error-msg {
  font-size: 13px;
  color: rgba(30, 30, 32, 0.68);
  max-width: 320px;
  word-break: break-all;
  margin-bottom: 20px;
}

button {
  font-family: inherit;
}

button.active {
  background: #71838F;
  border-color: #71838F;
}

/* 镜头轨道样式 */
.camera-track {
  position: absolute;
  bottom: 18px;
  left: 18px;
  right: 18px;
  display: flex;
  align-items: stretch;
  gap: 16px;
  overflow-x: auto;
  padding: 16px 18px;
  background: rgba(249, 249, 248, 0.84);
  backdrop-filter: blur(12px);
  border-radius: 22px;
  z-index: 100;
  border: 1px solid rgba(107, 122, 143, 0.16);
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.06);
}

.camera-track-header {
  min-width: 144px;
  padding-right: 4px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.camera-track-copy {
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.5;
  color: rgba(30, 30, 32, 0.68);
}

.camera-btn {
  width: 100px;
  height: 70px;
  background: rgba(255, 255, 255, 0.72);
  border-radius: 16px;
  cursor: pointer;
  overflow: hidden;
  border: 1px solid rgba(107, 122, 143, 0.12);
  transition: all 0.25s cubic-bezier(0.22, 1, 0.36, 1);
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #333;
  position: relative;
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.04);
  outline: none;
  user-select: none;
  -webkit-tap-highlight-color: transparent;
}

.camera-btn.active {
  border-color: rgba(107, 122, 143, 0.12);
  transform: translateY(-3px);
  box-shadow: 0 10px 20px rgba(107, 122, 143, 0.12);
}

.camera-btn:focus,
.camera-btn:focus-visible,
.camera-btn:active {
  outline: none;
  box-shadow: 0 10px 20px rgba(107, 122, 143, 0.12);
}

.btn-thumb {
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0.88;
  display: block;
  user-select: none;
  -webkit-user-drag: none;
  -webkit-tap-highlight-color: transparent;
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
  background: linear-gradient(180deg, transparent, rgba(30, 30, 32, 0.72));
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
  color: #fff;
  display: flex;
  flex-direction: column;
  padding: 10px 8px 8px;
  align-items: center;
  pointer-events: none;
  border-radius: 0 0 16px 16px;
  overflow: hidden;
  clip-path: inset(0 round 0 0 16px 16px);
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
  position: static;
  z-index: auto;
  width: min(560px, 100%);
  max-width: 560px;
  display: flex;
  flex: 1 1 auto;
  min-width: 0;
  flex-direction: row;
  align-items: center;
  gap: 8px;
  padding: 8px;
  border-radius: 18px;
}

.search-input {
  flex: 1 1 auto;
  width: auto;
  min-width: 0;
  padding: 10px 12px;
  border: 1px solid rgba(107, 122, 143, 0.14);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.72);
  outline: none;
  font-size: 13px;
  color: #1e1e20;
}

.search-input:focus {
  border-color: rgba(107, 122, 143, 0.5);
  box-shadow: 0 0 0 4px rgba(107, 122, 143, 0.08);
}

.search-btn {
  flex: 0 0 auto;
  padding: 10px 12px;
  border-radius: 12px;
  white-space: nowrap;
}

.focal-settings-toggle {
  position: static;
  z-index: auto;
}

.focal-settings-panel {
  position: absolute;
  top: 122px;
  right: 18px;
  z-index: 120;
  width: 236px;
  background: rgba(249, 249, 248, 0.9);
  color: #1e1e20;
  border: 1px solid rgba(107, 122, 143, 0.16);
  border-radius: 20px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  box-shadow: 0 16px 28px rgba(0, 0, 0, 0.08);
  backdrop-filter: blur(16px);
}

.focal-title {
  font-size: 15px;
  font-weight: 700;
}

.focal-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  font-size: 12px;
  color: rgba(30, 30, 32, 0.72);
}

.focal-number-input {
  width: 100px;
  border-radius: 10px;
  border: 1px solid rgba(107, 122, 143, 0.16);
  padding: 8px 10px;
  background: rgba(255, 255, 255, 0.86);
}

.focal-reset-btn {
  width: 100%;
}

/* 参考图浮窗 */
.reference-overlay {
  position: absolute;
  top: 76px;
  right: 18px;
  width: min(22vw, 148px);
  min-width: 112px;
  background: rgba(249, 249, 248, 0.9);
  padding: 8px;
  border-radius: 16px;
  border: 1px solid rgba(107, 122, 143, 0.14);
  z-index: 150;
  cursor: pointer;
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.08);
  backdrop-filter: blur(16px);
}

.ref-title {
  font-size: 12px;
  color: #1e1e20;
  margin: 2px 0 6px;
  font-weight: 600;
}

.ref-img {
  width: 100%;
  border-radius: 10px;
  border: 1px solid rgba(107, 122, 143, 0.12);
  margin-bottom: 6px;
}

.ref-info {
  font-size: 9px;
  color: rgba(30, 30, 32, 0.7);
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 4px;
}

.info-tag {
  background: rgba(228, 232, 237, 0.78);
  padding: 3px 6px;
  border-radius: 999px;
}

.info-tag--accent {
  color: #6d8260;
}

.ref-hint {
  font-size: 9px;
  color: rgba(30, 30, 32, 0.48);
  margin-top: 2px;
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
  color: #1e1e20;
  background: rgba(249, 249, 248, 0.84);
  border: 1px solid rgba(107, 122, 143, 0.16);
  border-radius: 12px;
  padding: 8px 10px;
  font-family: monospace;
  font-size: 12px;
  pointer-events: none;
}

input[type='range'] {
  accent-color: #6b7a8f;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.18);
    opacity: 0.75;
  }
}

@media (max-width: 768px) {
  .top-hud {
    top: 12px;
    left: 12px;
    right: 12px;
    gap: 8px;
  }

  .top-actions {
    align-self: stretch;
    justify-content: space-between;
    gap: 8px;
  }

  .view-mode-switch {
    padding: 4px;
    gap: 4px;
  }

  .mode-chip {
    padding: 8px 10px;
    font-size: 12px;
  }

  .search-panel {
    width: 100%;
    max-width: none;
    padding: 6px;
    gap: 6px;
  }

  .search-input {
    padding: 9px 10px;
    font-size: 12px;
  }

  .search-btn {
    padding: 9px 10px;
    font-size: 12px;
  }

  .reference-overlay {
    top: 60px;
    right: 12px;
    width: 112px;
    min-width: 112px;
    padding: 7px;
  }

  .camera-track {
    padding-top: 14px;
  }

  .camera-track-header {
    min-width: 116px;
  }

  .camera-btn {
    width: 92px;
    height: 66px;
  }
}
</style>
