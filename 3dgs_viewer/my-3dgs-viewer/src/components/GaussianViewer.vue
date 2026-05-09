<script setup>
import { onMounted, onBeforeUnmount, ref, computed } from 'vue';
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import gsap from 'gsap';
import BottomSelector from './BottomSelector.vue';

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
const activePoseId = ref(''); // 当前高亮的镜头项，不一定同步刷新右侧参考图
const sceneMetadata = ref({}); // 存储 FOV 等元数据
const debugInfo = ref({ x: 0, y: 0, z: 0 }); // 调试用的旋转信息
const arrivalEuler = ref({ x: 0, y: 0, z: 0 }); // 刚飞到时的欧拉角
const loadError = ref(''); // 添加错误状态
const loadingProgress = ref(0);
const loadingStatusText = ref('准备加载模型');
const currentFps = ref(0); // 实时帧数
const showFocalSettings = ref(false); // 焦距设置面板
const currentViewFov = ref(0); // 当前相机FOV
const currentViewFocalPx = ref(0); // 当前相机等效焦距（像素）
const manualFocalPx = ref(null); // 手动焦距输入
const cinematicSpeed = ref(1);
const cinematicProgress = ref(0);
const cinematicLoop = ref(true);
const isCinematicPlaying = ref(false);
const isCinematicPaused = ref(false);
const cinematicSmoothness = ref(0.68);
const cinematicSubjectLock = ref(true);
const showCinematicPanel = ref(false);
const modelList = ref([]);
const activeModelId = ref('');
const showBottomSelector = computed(() => modelList.value.length > 1 || (!isOrbitMode.value && filteredPoses.value.length > 0));
const hasModelTab = computed(() => modelList.value.length > 1);
const hasPoseTab = computed(() => !isOrbitMode.value && filteredPoses.value.length > 0);
const DEFAULT_FOCAL_PX = 380; // 无位姿元数据时使用更广一点的默认焦距
const FREE_LOOK_SENSITIVITY = 0.0048;
const WHEEL_ZOOM_STEP = 0.08;
const PINCH_ZOOM_STEP = 1.8;
const ORBIT_YAW_SENSITIVITY = 0.0048;
const ORBIT_PITCH_SENSITIVITY = 0.0048;
const ORBIT_ROLL_SENSITIVITY = 1.0;
const CINEMATIC_MIN_LOOK_AHEAD = 1.2;
const CINEMATIC_MAX_LOOK_AHEAD = 8.0;
const CINEMATIC_PATH_BLEND = 0.72;
const CINEMATIC_CAMERA_DAMPING_FAST = 0.26;
const CINEMATIC_CAMERA_DAMPING_SLOW = 0.1;
const CINEMATIC_MAX_KEYFRAMES = 18;
const CINEMATIC_MIN_KEYFRAMES = 6;
const CINEMATIC_UP_ALIGNMENT_MIN = 0.45;
const ORBIT_RECENTER_DURATION = 1.5;
const OPTIMIZED_MODEL_EXTENSIONS = ['.ksplat', '.splat'];
const SAME_ORIGIN_MODEL_HEAD_TIMEOUT_MS = 1200;
const DESKTOP_INTRO_PARTICLE_BUDGET = 120000;
const MOBILE_INTRO_PARTICLE_BUDGET = 45000;
const INTRO_DURATION_MS = 6500;
const INTRO_ORBIT_AXIS = new THREE.Vector3(0, 0, 1);

const isOrbitMode = computed(() => currentViewMode.value === VIEW_MODE.ORBIT);

const isMobileDevice = () => (
  /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
);

const getFileExtension = (url) => {
  const path = String(url || '').split('?')[0].split('#')[0];
  const match = path.match(/\.[^./\\]+$/);
  return match ? match[0].toLowerCase() : '';
};

const replaceModelExtension = (url, extension) => {
  const value = String(url || '');
  const queryIndex = value.search(/[?#]/);
  const base = queryIndex === -1 ? value : value.slice(0, queryIndex);
  const suffix = queryIndex === -1 ? '' : value.slice(queryIndex);
  return `${base.replace(/\.(ply|splat|ksplat)$/i, extension)}${suffix}`;
};

const isSameOriginUrl = (url) => {
  try {
    return new URL(url, window.location.href).origin === window.location.origin;
  } catch (_) {
    return false;
  }
};

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
const centerModeUp = new THREE.Vector3(0, 0, 1);
let activeCameraTween = null;
let activeCameraFlightId = 0;
let isOrbitRecenterFlightActive = false;
let pendingInitialTarget = null;
let didApplyInitialTarget = false;
let didApplyDefaultPose = false;
let posesFetchSettled = false;
let cinematicFrameHandle = 0;
let interactionFrameHandle = 0;
let renderRequested = false;

const cinematicState = {
  trajectory: null,
  phase: 'main',
  startTimeMs: 0,
  elapsedMs: 0,
  lastNearestPoseIndex: -1,
  filteredSample: null,
};

const interactionState = {
  freeVelocityYaw: 0,
  freeVelocityPitch: 0,
  orbitVelocityYaw: 0,
  orbitVelocityPitch: 0,
  orbitZoomVelocity: 0,
  freeInertiaActive: false,
  orbitInertiaActive: false,
  zoomInertiaActive: false,
  lastFrameTime: 0,
};

const orbitState = {
  center: new THREE.Vector3(0, 0, 0),
  yaw: 0,
  pitch: 0,
  radius: 3,
  targetYaw: 0,
  targetPitch: 0,
  targetRadius: 3,
};
let orbitNeedsRecenterAfterPoseFlight = false;

const reusableYawQuat = new THREE.Quaternion();
const reusablePitchQuat = new THREE.Quaternion();

const rotationDelta = ref({ x: 0, y: 0 }); // 记录用户微调了多少度
const canPlayCinematic = computed(() => cameraPoses.value.length >= 2);
const cinematicButtonLabel = computed(() => {
  if (isCinematicPlaying.value) return '暂停运镜';
  if (isCinematicPaused.value) return '继续运镜';
  return '开始运镜';
});

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

const requestRender = () => {
  if (renderRequested) return;
  renderRequested = true;

  requestAnimationFrame(() => {
    renderRequested = false;
    if (!viewer) return;
    try {
      viewer.update();
      viewer.render();
    } catch (_) {}
  });
};

const hasModelResource = async (url) => {
  if (!isSameOriginUrl(url)) return false;

  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), SAME_ORIGIN_MODEL_HEAD_TIMEOUT_MS);
  try {
    const response = await fetch(url, {
      method: 'HEAD',
      cache: 'no-store',
      signal: controller.signal,
    });
    return response.ok;
  } catch (_) {
    return false;
  } finally {
    window.clearTimeout(timeout);
  }
};

const resolvePreferredModelUrl = async (modelUrl) => {
  const sourceUrl = modelUrl || currentPlyUrl;
  const ext = getFileExtension(sourceUrl);
  if (!sourceUrl || ext !== '.ply') return sourceUrl;

  for (const candidateExt of OPTIMIZED_MODEL_EXTENSIONS) {
    const candidate = replaceModelExtension(sourceUrl, candidateExt);
    if (candidate === sourceUrl) continue;
    if (await hasModelResource(candidate)) {
      console.info(`[Viewer] 检测到优化模型格式，优先加载: ${candidate}`);
      return candidate;
    }
  }

  return sourceUrl;
};

const addSplatSceneWithFormatFallback = async (sourceUrl) => {
  const preferredUrl = await resolvePreferredModelUrl(sourceUrl);
  const candidates = [preferredUrl];
  if (preferredUrl !== sourceUrl) candidates.push(sourceUrl);

  let lastError = null;
  for (const candidate of candidates) {
    try {
      console.log(`[Viewer] 加载模型: ${candidate}`);
      loadingStatusText.value = '下载模型数据';
      await viewer.addSplatScene(candidate, {
        'showLoadingUI': false,
        'progressiveLoad': false,
        'optimizeSplatData': true,
        'freeIntermediateSplatData': true,
        'onProgress': (percentComplete, percentCompleteLabel, loaderStatus) => {
          const percent = Number(percentComplete);
          if (Number.isFinite(percent)) {
            const normalized = THREE.MathUtils.clamp(percent / 100, 0, 1);
            loadingProgress.value = loaderStatus === 1
              ? THREE.MathUtils.clamp(0.96 + normalized * 0.04, loadingProgress.value, 1)
              : THREE.MathUtils.clamp(normalized * 0.96, loadingProgress.value, 0.96);
          }
          if (loaderStatus === 1) {
            loadingStatusText.value = '解析并构建高斯数据';
          } else {
            loadingStatusText.value = percentCompleteLabel
              ? `下载模型数据 ${percentCompleteLabel}`
              : '下载模型数据';
          }
        },
        'rotation': [0, 0, 0, 1] // [x, y, z, w] Identity Quaternion (No global rotation)
      });
      currentPlyUrl = candidate;
      loadingProgress.value = 1;
      loadingStatusText.value = '模型加载完成，准备入场动画';
      return candidate;
    } catch (error) {
      lastError = error;
      if (candidate !== sourceUrl) {
        console.warn(`[Viewer] 优化格式加载失败，回退原始模型: ${sourceUrl}`, error);
        continue;
      }
      throw error;
    }
  }

  throw lastError || new Error('模型加载失败');
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
        // 将投影更新合并到下一帧，避免与 viewer 自身循环重复渲染。
        requestRender();
        refreshCurrentFocalInfo();
      }
    });
  } else {
    cam.fov = targetFov;
    cam.updateProjectionMatrix();
    requestRender();
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
const getSceneRadius = () => {
  const radius = Number(globalUniforms.uMaxRadius.value || 0);
  return radius > 0 ? radius : 1;
};

const clampOrbitPitch = (pitch) => THREE.MathUtils.clamp(
  pitch,
  THREE.MathUtils.degToRad(-86),
  THREE.MathUtils.degToRad(86)
);

const getOrbitMinRadius = () => Math.max(getSceneRadius() * 0.18, 0.08);
const getOrbitMaxRadius = () => Math.max(getSceneRadius() * 12, getOrbitMinRadius() * 6);

const syncOrbitTarget = (center = getModelWorldCenter()) => {
  if (!viewer || !viewer.camera) return;

  orbitState.center.copy(center);
  const offset = viewer.camera.position.clone().sub(orbitState.center);
  let radius = offset.length();
  if (!Number.isFinite(radius) || radius < getOrbitMinRadius()) {
    radius = Math.max(getSceneRadius() * 2.6, 1.5);
    offset.set(0, 0, radius);
  }

  orbitState.radius = THREE.MathUtils.clamp(radius, getOrbitMinRadius(), getOrbitMaxRadius());
  orbitState.targetRadius = orbitState.radius;
  orbitState.yaw = Math.atan2(offset.y, offset.x);
  orbitState.pitch = clampOrbitPitch(Math.asin(THREE.MathUtils.clamp(offset.z / radius, -1, 1)));
  orbitState.targetYaw = orbitState.yaw;
  orbitState.targetPitch = orbitState.pitch;
};

const getOrbitCameraState = () => {
  const cosPitch = Math.cos(orbitState.targetPitch);
  const position = orbitState.center.clone().add(new THREE.Vector3(
    Math.cos(orbitState.targetYaw) * cosPitch * orbitState.targetRadius,
    Math.sin(orbitState.targetYaw) * cosPitch * orbitState.targetRadius,
    Math.sin(orbitState.targetPitch) * orbitState.targetRadius,
  ));
  const lookMatrix = new THREE.Matrix4().lookAt(position, orbitState.center, centerModeUp);
  return {
    position,
    quaternion: new THREE.Quaternion().setFromRotationMatrix(lookMatrix),
  };
};

const buildCameraBezierCurve = (startPos, endPos, targetCenter) => {
  const sceneRadius = getSceneRadius();
  const distance = startPos.distanceTo(endPos);
  const lift = centerModeUp.clone().multiplyScalar(Math.max(sceneRadius * 0.18, distance * 0.14, 0.08));
  const startForward = targetCenter.clone().sub(startPos).normalize();
  const endBack = endPos.clone().sub(targetCenter).normalize();
  const handleDistance = Math.max(distance * 0.35, sceneRadius * 0.35, 0.15);

  return new THREE.CubicBezierCurve3(
    startPos.clone(),
    startPos.clone().add(startForward.multiplyScalar(handleDistance)).add(lift),
    endPos.clone().add(endBack.multiplyScalar(handleDistance * 0.28)).add(lift.multiplyScalar(0.45)),
    endPos.clone()
  );
};

const cancelCinematicFrame = () => {
  if (cinematicFrameHandle) {
    cancelAnimationFrame(cinematicFrameHandle);
    cinematicFrameHandle = 0;
  }
};

const stopCameraTweens = () => {
  if (!viewer || !viewer.camera) return;
  activeCameraFlightId += 1;
  if (activeCameraTween) {
    activeCameraTween.kill();
    activeCameraTween = null;
  }
  isOrbitRecenterFlightActive = false;
  orbitNeedsRecenterAfterPoseFlight = false;
  gsap.killTweensOf(viewer.camera.position);
  gsap.killTweensOf(viewer.camera.quaternion);
  gsap.killTweensOf(viewer.camera);
};

const interruptCameraFlight = () => {
  const hadActiveFlight = Boolean(activeCameraTween);
  stopCameraTweens();
  if (!hadActiveFlight || !viewer || !viewer.camera) return;

  // 用户输入优先级高于自动飞行，取消后保留当前帧姿态作为新的交互起点。
  if (viewer.controls) viewer.controls.enabled = true;
  if (isOrbitMode.value) syncOrbitTarget();
  renderCameraUpdate();
};

const interruptCameraFlightFromUserInput = () => {
  if (!activeCameraTween) return;
  interruptCameraFlight();
};

const isCameraFlightLocked = () => Boolean(activeCameraTween);

const setActivePosePresentation = (poseData) => {
  setActivePosePresentationState(poseData);
};

const getPosePresentationId = (poseData) => {
  if (!poseData) return '';
  return String(
    poseData.id
    || poseData.image_id
    || poseData.imageId
    || getPoseImageId(poseData)
    || JSON.stringify(normalizeMatrixArray(poseData.matrix) || [])
  );
};

const setActivePosePresentationState = (poseData, options = {}) => {
  activePoseId.value = getPosePresentationId(poseData);

  if (options.updateReference === false) return;

  activeImage.value = poseData?.image_url || getPoseImageId(poseData);
  activeTag.value = poseData?.tag || '';
};

const stopCinematicPlayback = (options = {}) => {
  cancelCinematicFrame();
  cinematicState.trajectory = null;
  cinematicState.startTimeMs = 0;
  cinematicState.elapsedMs = 0;
  cinematicState.lastNearestPoseIndex = -1;
  cinematicState.filteredSample = null;
  isCinematicPlaying.value = false;
  isCinematicPaused.value = false;
  if (options.resetProgress !== false) {
    cinematicProgress.value = 0;
  }
};

const interruptCinematicPlayback = () => {
  if (!isCinematicPlaying.value && !isCinematicPaused.value) return;
  stopCinematicPlayback({ resetProgress: false });
  stopInteractionInertia();
};

const smoothVectorSeries = (vectors, amount) => {
  if (!Array.isArray(vectors) || vectors.length < 3) return vectors.map(vec => vec.clone());

  const strength = THREE.MathUtils.clamp(Number(amount) || 0, 0, 1);
  const passes = Math.max(1, Math.round(1 + strength * 3));
  const blend = 0.12 + strength * 0.26;
  let result = vectors.map(vec => vec.clone());

  for (let pass = 0; pass < passes; pass += 1) {
    const nextSeries = result.map((vec, index) => {
      if (index === 0 || index === result.length - 1) return vec.clone();
      const blended = result[index - 1].clone()
        .add(result[index].clone().multiplyScalar(2))
        .add(result[index + 1])
        .multiplyScalar(0.25);
      return vec.clone().lerp(blended, blend);
    });
    result = nextSeries;
  }

  return result;
};

const smoothScalarSeries = (values, amount) => {
  if (!Array.isArray(values) || values.length < 3) return values.slice();

  const strength = THREE.MathUtils.clamp(Number(amount) || 0, 0, 1);
  const passes = Math.max(1, Math.round(1 + strength * 2));
  const blend = 0.1 + strength * 0.28;
  let result = values.slice();

  for (let pass = 0; pass < passes; pass += 1) {
    const nextSeries = result.map((value, index) => {
      if (index === 0 || index === result.length - 1) return value;
      const averaged = (result[index - 1] + result[index] * 2 + result[index + 1]) / 4;
      return THREE.MathUtils.lerp(value, averaged, blend);
    });
    result = nextSeries;
  }

  return result;
};

const makeLookQuaternion = (position, target) => {
  const forward = target.clone().sub(position);
  if (forward.lengthSq() < 1e-8) return new THREE.Quaternion();

  const lookAtMatrix = new THREE.Matrix4().lookAt(position, target, worldUp);
  return new THREE.Quaternion().setFromRotationMatrix(lookAtMatrix);
};

const makeUprightQuaternion = (position, quaternion, fallbackTarget) => {
  const forward = getCameraForward(quaternion);
  let target = position.clone().add(forward);

  if (fallbackTarget && forward.lengthSq() < 1e-8) {
    target = fallbackTarget.clone();
  }

  return makeLookQuaternion(position, target);
};

const ensureQuaternionContinuity = (quaternions) => {
  if (!Array.isArray(quaternions) || quaternions.length === 0) return [];

  const result = [quaternions[0].clone().normalize()];
  for (let i = 1; i < quaternions.length; i += 1) {
    const next = quaternions[i].clone().normalize();
    if (result[i - 1].dot(next) < 0) {
      next.x *= -1;
      next.y *= -1;
      next.z *= -1;
      next.w *= -1;
    }
    result.push(next);
  }

  return result;
};

const smoothQuaternionSeries = (quaternions, amount) => {
  if (!Array.isArray(quaternions) || quaternions.length < 3) {
    return ensureQuaternionContinuity(quaternions || []);
  }

  const strength = THREE.MathUtils.clamp(Number(amount) || 0, 0, 1);
  const passes = Math.max(1, Math.round(1 + strength * 2));
  const blend = 0.16 + strength * 0.22;
  let result = ensureQuaternionContinuity(quaternions);

  for (let pass = 0; pass < passes; pass += 1) {
    const nextSeries = result.map((quat, index) => {
      if (index === 0 || index === result.length - 1) return quat.clone();

      const prev = result[index - 1].clone();
      const curr = result[index].clone();
      const next = result[index + 1].clone();
      const averaged = prev.slerp(next, 0.5);
      return curr.slerp(averaged, blend).normalize();
    });
    result = ensureQuaternionContinuity(nextSeries);
  }

  return result;
};

const getCameraUpAlignment = (quaternion) => {
  if (!quaternion) return 0;
  const cameraUp = new THREE.Vector3(0, 1, 0).applyQuaternion(quaternion).normalize();
  return Math.abs(cameraUp.dot(worldUp));
};

const getCameraForward = (quaternion) => {
  return new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion).normalize();
};

const selectStableCinematicKeyframes = (keyframes) => {
  if (!Array.isArray(keyframes) || keyframes.length <= CINEMATIC_MAX_KEYFRAMES) {
    return keyframes;
  }

  const filtered = keyframes.filter((frame) => {
    const alignment = getCameraUpAlignment(frame.quaternion);
    return alignment >= CINEMATIC_UP_ALIGNMENT_MIN;
  });

  const pool = filtered.length >= CINEMATIC_MIN_KEYFRAMES ? filtered : keyframes.slice();
  if (pool.length <= CINEMATIC_MAX_KEYFRAMES) return pool;

  const scored = pool.map((frame, index, arr) => {
    const prev = arr[Math.max(0, index - 1)];
    const next = arr[Math.min(arr.length - 1, index + 1)];
    const upAlignment = getCameraUpAlignment(frame.quaternion);
    const prevDistance = index > 0 ? frame.position.distanceTo(prev.position) : 0;
    const nextDistance = index < arr.length - 1 ? frame.position.distanceTo(next.position) : 0;
    const avgDistance = (prevDistance + nextDistance) * 0.5;
    const prevForward = getCameraForward(prev.quaternion);
    const currForward = getCameraForward(frame.quaternion);
    const nextForward = getCameraForward(next.quaternion);
    const directionalContinuity = index > 0 && index < arr.length - 1
      ? Math.max(0, prevForward.dot(currForward)) * 0.5 + Math.max(0, currForward.dot(nextForward)) * 0.5
      : 1;

    return {
      frame,
      index,
      score: upAlignment * 2.2 + directionalContinuity * 1.4 + Math.min(avgDistance, 1.5) * 0.4,
    };
  });

  const forcedIndices = new Set([0, pool.length - 1]);
  const targetCount = Math.max(CINEMATIC_MIN_KEYFRAMES, Math.min(CINEMATIC_MAX_KEYFRAMES, pool.length));
  const selected = scored
    .filter(({ index }) => forcedIndices.has(index))
    .map(({ frame }) => frame);

  const remaining = scored
    .filter(({ index }) => !forcedIndices.has(index))
    .sort((a, b) => b.score - a.score);

  for (const candidate of remaining) {
    if (selected.length >= targetCount) break;
    selected.push(candidate.frame);
  }

  selected.sort((a, b) => a.index - b.index);

  if (selected.length < CINEMATIC_MIN_KEYFRAMES) {
    const step = Math.max(1, Math.floor(pool.length / CINEMATIC_MIN_KEYFRAMES));
    for (let i = 0; i < pool.length && selected.length < CINEMATIC_MIN_KEYFRAMES; i += step) {
      const frame = pool[i];
      if (!selected.includes(frame)) selected.push(frame);
    }
    selected.sort((a, b) => a.index - b.index);
  }

  return selected;
};

const unwrapCircularAngles = (angles) => {
  if (!Array.isArray(angles) || angles.length === 0) return [];
  const sorted = angles
    .map((angle, index) => ({ angle, index }))
    .sort((a, b) => a.angle - b.angle);

  let largestGap = -1;
  let splitIndex = 0;
  for (let i = 0; i < sorted.length; i += 1) {
    const current = sorted[i].angle;
    const next = sorted[(i + 1) % sorted.length].angle + (i === sorted.length - 1 ? Math.PI * 2 : 0);
    const gap = next - current;
    if (gap > largestGap) {
      largestGap = gap;
      splitIndex = (i + 1) % sorted.length;
    }
  }

  const rotated = [];
  for (let i = 0; i < sorted.length; i += 1) {
    const item = sorted[(splitIndex + i) % sorted.length];
    let angle = item.angle;
    if (rotated.length > 0 && angle < rotated[rotated.length - 1].angle) {
      angle += Math.PI * 2;
    }
    rotated.push({ ...item, angle });
  }

  return rotated;
};

const computeRouteTransitionCost = (from, to, worldCenter) => {
  const distance = from.position.distanceTo(to.position);
  const fromForward = getCameraForward(from.quaternion);
  const toForward = getCameraForward(to.quaternion);
  const forwardMismatch = 1 - Math.max(-1, Math.min(1, fromForward.dot(toForward)));
  const fromTargetDir = worldCenter.clone().sub(from.position).normalize();
  const toTargetDir = worldCenter.clone().sub(to.position).normalize();
  const focusMismatch = 1 - Math.max(-1, Math.min(1, fromTargetDir.dot(toTargetDir)));
  const heightDelta = Math.abs(from.position.y - to.position.y);
  return distance * 1.25 + forwardMismatch * 1.4 + focusMismatch * 0.9 + heightDelta * 0.35;
};

const getDominantHorizontalAxis = (keyframes) => {
  let bestA = keyframes[0]?.position || new THREE.Vector3(1, 0, 0);
  let bestB = keyframes[keyframes.length - 1]?.position || new THREE.Vector3(0, 0, 0);
  let bestDistSq = -1;

  for (let i = 0; i < keyframes.length; i += 1) {
    for (let j = i + 1; j < keyframes.length; j += 1) {
      const distSq = keyframes[i].position.distanceToSquared(keyframes[j].position);
      if (distSq > bestDistSq) {
        bestDistSq = distSq;
        bestA = keyframes[i].position;
        bestB = keyframes[j].position;
      }
    }
  }

  const axis = bestB.clone().sub(bestA);
  axis.y = 0;
  if (axis.lengthSq() < 1e-6) axis.set(1, 0, 0);
  return axis.normalize();
};

const chooseLowerCostRouteDirection = (orderedKeyframes, worldCenter) => {
  if (!Array.isArray(orderedKeyframes) || orderedKeyframes.length < 3) return orderedKeyframes;

  const routeCost = (frames) => {
    let total = 0;
    for (let i = 1; i < frames.length; i += 1) {
      total += computeRouteTransitionCost(frames[i - 1], frames[i], worldCenter);
    }
    return total;
  };

  const forward = orderedKeyframes.slice();
  const reverse = orderedKeyframes.slice().reverse();
  return routeCost(forward) <= routeCost(reverse) ? forward : reverse;
};

const planSmartCinematicRoute = (keyframes, worldCenter) => {
  if (!Array.isArray(keyframes) || keyframes.length < 3) return keyframes;

  const horizontalDistances = keyframes.map((frame) => {
    const offset = frame.position.clone().sub(worldCenter);
    offset.y = 0;
    return offset.length();
  });
  const radiiMean = horizontalDistances.reduce((sum, value) => sum + value, 0) / horizontalDistances.length;
  const radiiVariance = horizontalDistances.reduce((sum, value) => sum + ((value - radiiMean) ** 2), 0) / horizontalDistances.length;
  const radiiStd = Math.sqrt(radiiVariance);
  const yValues = keyframes.map((frame) => frame.position.y);
  const heightSpread = Math.max(...yValues) - Math.min(...yValues);

  const angles = keyframes.map((frame) => {
    const offset = frame.position.clone().sub(worldCenter);
    return Math.atan2(offset.z, offset.x);
  });
  const unwrappedAngles = unwrapCircularAngles(angles);
  const angleSpread = unwrappedAngles.length > 1
    ? unwrappedAngles[unwrappedAngles.length - 1].angle - unwrappedAngles[0].angle
    : 0;

  let routeMode = 'dolly';
  if (angleSpread > 1.1 && radiiStd < Math.max(0.35, radiiMean * 0.28)) {
    routeMode = 'orbit';
  } else if (heightSpread > Math.max(0.8, radiiMean * 0.42)) {
    routeMode = 'crane';
  }

  let ordered = keyframes.slice();

  if (routeMode === 'orbit') {
    const byAngle = unwrappedAngles.map(({ index }) => keyframes[index]);
    ordered = chooseLowerCostRouteDirection(byAngle, worldCenter);
  } else if (routeMode === 'crane') {
    ordered = chooseLowerCostRouteDirection(
      keyframes.slice().sort((a, b) => a.position.y - b.position.y),
      worldCenter
    );
  } else {
    const axis = getDominantHorizontalAxis(keyframes);
    ordered = chooseLowerCostRouteDirection(
      keyframes.slice().sort((a, b) => a.position.dot(axis) - b.position.dot(axis)),
      worldCenter
    );
  }

  return ordered.map((frame, index) => ({
    ...frame,
    routeIndex: index,
    routeMode,
  }));
};

const buildCinematicSegment = ({
  keyframes,
  positions,
  targets,
  focals,
  durationMs,
}) => {
  const sceneRadius = getSceneRadius();
  const stabilizedQuaternions = smoothQuaternionSeries(
    positions.map((position, index) => makeUprightQuaternion(position, keyframes[index].quaternion, targets[index])),
    cinematicSmoothness.value
  );

  const preparedKeyframes = keyframes.map((frame, index) => ({
    ...frame,
    position: positions[index],
    target: targets[index],
    stabilizedQuaternion: stabilizedQuaternions[index],
    fl_y: focals[index] || frame.fl_y,
  }));

  const curve = new THREE.CatmullRomCurve3(
    preparedKeyframes.map(frame => frame.position.clone()),
    false,
    'centripetal'
  );
  const lookCurve = new THREE.CatmullRomCurve3(
    preparedKeyframes.map(frame => frame.target.clone()),
    false,
    'centripetal'
  );
  const cumulativeDistances = [0];
  for (let i = 1; i < preparedKeyframes.length; i += 1) {
    const prev = preparedKeyframes[i - 1];
    const next = preparedKeyframes[i];
    cumulativeDistances.push(
      cumulativeDistances[i - 1] + prev.position.distanceTo(next.position)
    );
  }

  return {
    keyframes: preparedKeyframes,
    curve,
    lookCurve,
    cumulativeDistances,
    totalDistance: Math.max(cumulativeDistances[cumulativeDistances.length - 1], 1e-5),
    durationMs,
    lookAheadDistance: THREE.MathUtils.clamp(
      sceneRadius * (0.4 + cinematicSmoothness.value * 0.45),
      CINEMATIC_MIN_LOOK_AHEAD,
      CINEMATIC_MAX_LOOK_AHEAD
    ),
  };
};

const buildLoopBridgeSegment = (mainSegment, worldCenter) => {
  if (!mainSegment?.keyframes || mainSegment.keyframes.length < 2) return null;

  const sceneRadius = getSceneRadius();
  const first = mainSegment.keyframes[0];
  const last = mainSegment.keyframes[mainSegment.keyframes.length - 1];
  const directDistance = last.position.distanceTo(first.position);
  if (directDistance < 1e-4) return null;

  const liftAmount = Math.max(sceneRadius * 0.55, directDistance * 0.22, 0.9);
  const radialPush = Math.max(sceneRadius * 0.18, directDistance * 0.08, 0.35);
  const startOut = last.position.clone().sub(worldCenter).setY(0);
  const endOut = first.position.clone().sub(worldCenter).setY(0);

  if (startOut.lengthSq() < 1e-6) startOut.set(1, 0, 0);
  if (endOut.lengthSq() < 1e-6) endOut.set(-1, 0, 0);

  startOut.normalize().multiplyScalar(radialPush);
  endOut.normalize().multiplyScalar(radialPush);

  const centerLift = worldCenter.clone().add(new THREE.Vector3(0, sceneRadius * 0.15, 0));
  const bridgePositions = [
    last.position.clone(),
    last.position.clone().add(new THREE.Vector3(0, liftAmount, 0)).add(startOut),
    first.position.clone().add(new THREE.Vector3(0, liftAmount * 0.86, 0)).add(endOut),
    first.position.clone(),
  ];
  const bridgeTargets = [
    last.target.clone().lerp(centerLift, 0.4),
    centerLift.clone(),
    centerLift.clone(),
    first.target.clone().lerp(centerLift, 0.28),
  ];
  const bridgeFocal = Math.max(
    0,
    Number(last.fl_y || first.fl_y || sceneMetadata.value.fl_y || DEFAULT_FOCAL_PX)
  );
  const bridgeDurationMs = THREE.MathUtils.clamp(
    directDistance * 1350 + 1800,
    2400,
    6200
  ) / cinematicSpeed.value;

  return buildCinematicSegment({
    keyframes: [
      { index: last.index, pose: last.pose, quaternion: last.stabilizedQuaternion || last.quaternion, fl_y: bridgeFocal, h: last.h },
      { index: last.index, pose: last.pose, quaternion: last.stabilizedQuaternion || last.quaternion, fl_y: bridgeFocal, h: last.h },
      { index: first.index, pose: first.pose, quaternion: first.stabilizedQuaternion || first.quaternion, fl_y: bridgeFocal, h: first.h },
      { index: first.index, pose: first.pose, quaternion: first.stabilizedQuaternion || first.quaternion, fl_y: bridgeFocal, h: first.h },
    ],
    positions: bridgePositions,
    targets: bridgeTargets,
    focals: [bridgeFocal, bridgeFocal, bridgeFocal, bridgeFocal],
    durationMs: bridgeDurationMs,
  });
};

const manualMove = (axis, dist) => {
  if (!viewer || !viewer.camera) return;
  if (isCameraFlightLocked()) return;
  interruptCinematicPlayback();
  interruptCameraFlight();
  if (viewer.controls) viewer.controls.enabled = false;

  if (axis === 'x') viewer.camera.translateX(dist);
  if (axis === 'y') viewer.camera.translateY(dist);
  if (axis === 'z') viewer.camera.translateZ(dist);

  viewer.camera.updateProjectionMatrix();
};

const manualRotate = (axis, angleDeg) => {
  if (!viewer || !viewer.camera) return;
  if (isCameraFlightLocked()) return;
  interruptCinematicPlayback();
  interruptCameraFlight();

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
  INTRO: 0,
  FINISHED: 1
};

const animationState = {
  isLoaded: false,
  lastFrameTime: 0,
  phase: PHASE.INTRO,
  introStartTime: 0,
  introDurationMs: INTRO_DURATION_MS,
  introCamera: null,
};

const globalUniforms = {
  uTime: { value: 0 },
  uCenter: { value: new THREE.Vector3(0, 0, 0) },
  uGeoRadius: { value: 0 },
  uColorRadius: { value: 0 },
  uMaxRadius: { value: 50 }, // 将由自适应逻辑动态更新
  uParticleProgress: { value: 0 },
  uRevealProgress: { value: 0 },
  uRevealFeather: { value: 0.22 },
  uIntroSplatAlpha: { value: 0 },
};

const normalizeColorChannel = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0.6;
  return THREE.MathUtils.clamp(numeric > 1 ? numeric / 255 : numeric, 0, 1);
};

const readSplatRgbColor = (splatMesh, index, outColor) => {
  outColor.set(0.6, 0.6, 0.6, 1);
  if (!splatMesh || typeof splatMesh.getSplatColor !== 'function') return outColor;

  try {
    const returnedColor = splatMesh.getSplatColor(index, outColor);
    const source = returnedColor || outColor;
    const r = source.x ?? source.r ?? source[0] ?? outColor.x;
    const g = source.y ?? source.g ?? source[1] ?? outColor.y;
    const b = source.z ?? source.b ?? source[2] ?? outColor.z;
    outColor.set(normalizeColorChannel(r), normalizeColorChannel(g), normalizeColorChannel(b), 1);
  } catch (error) {
    if (index === 0) console.warn('[Intro] 读取 splat 颜色失败，粒子颜色回退为灰色:', error);
  }

  return outColor;
};

// --- 2. 自适应粒子系统 (核心修改) ---
const createParticleSystem = (splatMesh) => {
  if (!viewer) return;

  const splatCount = splatMesh.getSplatCount();
  if (!Number.isFinite(splatCount) || splatCount <= 0) return;
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
  // 入场动画只承担过渡表达，按设备预算采样可避免大模型额外分配几十 MB 数组。
  const particleBudget = isMobileDevice() ? MOBILE_INTRO_PARTICLE_BUDGET : DESKTOP_INTRO_PARTICLE_BUDGET;
  const targetParticleCount = Math.min(splatCount, particleBudget);
  const step = Math.max(1, Math.ceil(splatCount / targetParticleCount));
  const sampledCount = Math.ceil(splatCount / step);

  // 2. 自适应粒子大小
  // 逻辑：模型越大，单个粒子在世界空间中应该越大才能被看见。
  // 系数 150.0 是经验值，表示将最大边长切分多少份。
  let adaptiveSize = (maxDim / 95.0) * window.devicePixelRatio;
  // 限制最小值，防止极小模型看不见
  const minParticleSize = isMobileDevice() ? 1.8 : 2.4;
  if (adaptiveSize < minParticleSize) adaptiveSize = minParticleSize;
  adaptiveSize = Math.min(adaptiveSize, isMobileDevice() ? 7.0 : 10.0);

  // 3. 自适应飞行距离
  // 粒子应该从包围盒外面飞进来
  const flyRadiusBase = maxDim * 1.0;

  console.log(`[Adaptive] MaxDim: ${maxDim.toFixed(2)}, Particles: ~${sampledCount}, Size: ${adaptiveSize.toFixed(2)}`);

  // === C. 生成几何体 ===
  const geometry = new THREE.BufferGeometry();
  const startPositions = new Float32Array(sampledCount * 3);
  const targetPositions = new Float32Array(sampledCount * 3);
  const colors = new Float32Array(sampledCount * 3);
  const randoms = new Float32Array(sampledCount);
  const tempColor = new THREE.Vector4(0.6, 0.6, 0.6, 1);
  let didLogSampleColor = false;
  let sampleIndex = 0;

  for (let i = 0; i < splatCount; i += step) {
    splatMesh.getSplatCenter(i, tempVec);
    tempVec.applyMatrix4(splatMesh.matrixWorld);

    const positionIndex = sampleIndex * 3;
    targetPositions[positionIndex] = tempVec.x;
    targetPositions[positionIndex + 1] = tempVec.y;
    targetPositions[positionIndex + 2] = tempVec.z;

    readSplatRgbColor(splatMesh, i, tempColor);
    colors[positionIndex] = tempColor.x;
    colors[positionIndex + 1] = tempColor.y;
    colors[positionIndex + 2] = tempColor.z;
    if (!didLogSampleColor) {
      console.log(
        `[Intro] Sample particle color: ${tempColor.x.toFixed(3)}, ${tempColor.y.toFixed(3)}, ${tempColor.z.toFixed(3)}`
      );
      didLogSampleColor = true;
    }

    // 随机分布在远处 (基于自适应的 maxDim)
    const r = flyRadiusBase + Math.random() * (maxDim * 0.5);
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);

    // 从中心点向外偏移
    const startX = centerX + r * Math.sin(phi) * Math.cos(theta);
    const startY = centerY + r * Math.sin(phi) * Math.sin(theta);
    const startZ = centerZ + r * Math.cos(phi);

    startPositions[positionIndex] = startX;
    startPositions[positionIndex + 1] = startY;
    startPositions[positionIndex + 2] = startZ;
    randoms[sampleIndex] = Math.random();
    sampleIndex += 1;
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(startPositions, 3));
  geometry.setAttribute('aTarget', new THREE.BufferAttribute(targetPositions, 3));
  geometry.setAttribute('aColor', new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute('aRandom', new THREE.BufferAttribute(randoms, 1));

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uProgress: globalUniforms.uParticleProgress,
      uSize: { value: adaptiveSize }, // 使用计算出的大小
    },
    vertexShader: `
      uniform float uProgress;
      uniform float uSize;
      attribute vec3 aTarget;
      attribute vec3 aColor;
      attribute float aRandom;
      varying vec3 vColor;
      
      float easeOutCubic(float x) { return 1.0 - pow(1.0 - x, 3.0); }
      
      void main() {
        float t = (uProgress - aRandom * 0.1) / 0.9;
        t = clamp(t, 0.0, 1.0);
        vec3 pos = mix(position, aTarget, easeOutCubic(t));
        vColor = aColor;
        
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        
        // 距离衰减 (20.0 是透视缩放因子，配合世界单位的 uSize 使用)
        gl_PointSize = uSize * (44.0 / max(-mvPosition.z, 0.001));
        if(gl_PointSize < 2.0) gl_PointSize = 2.0;
        if(gl_PointSize > 32.0) gl_PointSize = 32.0;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        if(length(coord) > 0.5) discard;
        gl_FragColor = vec4(vColor, 1.0);
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
  material.uniforms.uRevealProgress = globalUniforms.uRevealProgress;
  material.uniforms.uRevealFeather = globalUniforms.uRevealFeather;
  material.uniforms.uIntroSplatAlpha = globalUniforms.uIntroSplatAlpha;

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
    uniform float uRevealProgress;
    uniform float uRevealFeather;
    uniform float uIntroSplatAlpha;
    uniform vec3 uCenter;
    varying vec3 vWorldPosition;

    float introHash(vec3 p) {
      return fract(sin(dot(p, vec3(17.13, 43.71, 91.17))) * 43758.5453);
    }

    float introValueNoise(vec3 p) {
      vec3 i = floor(p);
      vec3 f = smoothstep(vec3(0.0), vec3(1.0), fract(p));
      float n000 = introHash(i + vec3(0.0, 0.0, 0.0));
      float n100 = introHash(i + vec3(1.0, 0.0, 0.0));
      float n010 = introHash(i + vec3(0.0, 1.0, 0.0));
      float n110 = introHash(i + vec3(1.0, 1.0, 0.0));
      float n001 = introHash(i + vec3(0.0, 0.0, 1.0));
      float n101 = introHash(i + vec3(1.0, 0.0, 1.0));
      float n011 = introHash(i + vec3(0.0, 1.0, 1.0));
      float n111 = introHash(i + vec3(1.0, 1.0, 1.0));
      float nx00 = mix(n000, n100, f.x);
      float nx10 = mix(n010, n110, f.x);
      float nx01 = mix(n001, n101, f.x);
      float nx11 = mix(n011, n111, f.x);
      float nxy0 = mix(nx00, nx10, f.y);
      float nxy1 = mix(nx01, nx11, f.y);
      return mix(nxy0, nxy1, f.z);
    }
  `;
  material.fragmentShader = commonFragment + material.fragmentShader;

  const fsEndIndex = material.fragmentShader.lastIndexOf('}');
  if (fsEndIndex !== -1) {
    const originalContent = material.fragmentShader.substring(0, fsEndIndex);
    const visualLogic = `
      vec3 centeredPos = vWorldPosition - uCenter;
      float distFromCenter = length(centeredPos);
      float normalizedOrder = clamp(distFromCenter / max(uMaxRadius, 0.0001), 0.0, 1.0);
      float waveNoise = introValueNoise(centeredPos * 2.2);
      float fineNoise = introValueNoise(centeredPos * 7.5 + 19.0);
      float noisyOrder = clamp(normalizedOrder + (waveNoise - 0.5) * 0.18 + (fineNoise - 0.5) * 0.06, 0.0, 1.0);
      float revealT = smoothstep(noisyOrder, noisyOrder + uRevealFeather, uRevealProgress);

      if (revealT <= 0.001 || uIntroSplatAlpha <= 0.001) discard;

      // 带噪声的空间波前让高斯从中心向外逐片恢复，而不是全局同时变形。
      float alphaClip = mix(0.95, 0.02, revealT);
      if (gl_FragColor.a < alphaClip) discard;
      gl_FragColor.a *= revealT * uIntroSplatAlpha;
    `;
    material.fragmentShader = originalContent + visualLogic + '}';
  }
  material.needsUpdate = true;
};

const smoothstep01 = (value) => {
  const t = THREE.MathUtils.clamp(value, 0, 1);
  return t * t * (3 - 2 * t);
};

const resetIntroUniforms = () => {
  globalUniforms.uParticleProgress.value = 0;
  globalUniforms.uRevealProgress.value = 0;
  globalUniforms.uIntroSplatAlpha.value = 0;
  globalUniforms.uGeoRadius.value = 0;
  globalUniforms.uColorRadius.value = 0;
  if (particleSystem) {
    particleSystem.geometry?.dispose?.();
    particleSystem.material?.dispose?.();
    if (viewer?.threeScene) viewer.threeScene.remove(particleSystem);
    particleSystem = null;
  }
};

const resetIntroAnimationVisuals = () => {
  globalUniforms.uParticleProgress.value = 0;
  globalUniforms.uRevealProgress.value = 0;
  globalUniforms.uIntroSplatAlpha.value = 0;
  globalUniforms.uGeoRadius.value = 0;
  globalUniforms.uColorRadius.value = 0;
  if (particleSystem) {
    particleSystem.visible = true;
    if (particleSystem.material) particleSystem.material.opacity = 1;
  }
};

const finalizeIntroAnimation = () => {
  const splatMesh = viewer?.getSplatMesh?.();
  if (splatMesh) splatMesh.visible = true;
  if (particleSystem) particleSystem.visible = false;
  globalUniforms.uParticleProgress.value = 1;
  globalUniforms.uRevealProgress.value = 1.5;
  globalUniforms.uIntroSplatAlpha.value = 1;
  globalUniforms.uGeoRadius.value = 99999;
  globalUniforms.uColorRadius.value = 99999;
  animationState.phase = PHASE.FINISHED;
  animationState.isLoaded = false;
  animationState.introCamera = null;
  if (viewer?.controls) viewer.controls.enabled = true;
  if (isOrbitMode.value) syncOrbitTarget();
  updateDebugInfo();
};

const getFarthestIntroStartPose = (targetPosition, targetPose = null) => {
  if (!viewer || !targetPosition || !Array.isArray(cameraPoses.value) || cameraPoses.value.length < 2) return null;

  const targetPoseId = getPosePresentationId(targetPose);
  let farthestState = null;
  let farthestDistanceSq = -1;

  for (const pose of cameraPoses.value) {
    if (targetPose && getPosePresentationId(pose) === targetPoseId) continue;
    const state = resolvePoseCameraState(pose);
    if (!state) continue;
    const distanceSq = state.position.distanceToSquared(targetPosition);
    if (distanceSq > farthestDistanceSq) {
      farthestDistanceSq = distanceSq;
      farthestState = state;
    }
  }

  return farthestState;
};

const buildIntroOrbitCamera = (targetPosition, targetQuaternion, targetPose = null) => {
  const center = getModelWorldCenter();
  const radius = getSceneRadius();
  const farthestStartState = getFarthestIntroStartPose(targetPosition, targetPose);
  let startPosition;
  let startQuaternion;

  if (farthestStartState) {
    startPosition = farthestStartState.position.clone();
    startQuaternion = makeLookQuaternion(startPosition, center);
  } else {
    const targetOffset = targetPosition.clone().sub(center);
    if (targetOffset.lengthSq() < 1e-8) targetOffset.set(Math.max(radius * 2.5, 1), 0, 0);

    const orbitAxis = INTRO_ORBIT_AXIS.clone().normalize();
    const startOffset = targetOffset.clone()
      .applyAxisAngle(orbitAxis, THREE.MathUtils.degToRad(115))
      .multiplyScalar(1.18);
    const lift = Math.max(radius * 0.22, targetOffset.length() * 0.08, 0.08);
    startOffset.z += lift;

    startPosition = center.clone().add(startOffset);
    startQuaternion = makeLookQuaternion(startPosition, center);
  }

  const curve = buildCameraBezierCurve(startPosition, targetPosition, center);

  return {
    center,
    curve,
    startPosition: startPosition.clone(),
    startQuaternion,
    targetPosition: targetPosition.clone(),
    targetQuaternion: targetQuaternion.clone(),
  };
};

const beginIntroAnimation = (targetCameraState = null, targetPose = null) => {
  if (!viewer || !viewer.camera) return;
  const splatMesh = viewer.getSplatMesh();
  if (!splatMesh) return;

  const targetPosition = targetCameraState?.position?.clone?.() || viewer.camera.position.clone();
  const targetQuaternion = targetCameraState?.quaternion?.clone?.() || viewer.camera.quaternion.clone();
  animationState.introCamera = buildIntroOrbitCamera(targetPosition, targetQuaternion, targetPose);
  animationState.introStartTime = performance.now();
  animationState.lastFrameTime = Date.now();
  animationState.phase = PHASE.INTRO;
  animationState.isLoaded = true;

  resetIntroAnimationVisuals();
  splatMesh.visible = false;
  viewer.camera.position.copy(animationState.introCamera.startPosition);
  viewer.camera.quaternion.copy(animationState.introCamera.startQuaternion);
  viewer.camera.updateProjectionMatrix();
  if (viewer.controls) viewer.controls.enabled = false;
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

const isRemoteHttpUrl = (value) => /^https?:\/\//i.test(String(value || ''));

const toViewerSafeAssetUrl = (value) => {
  if (typeof value !== 'string' || !value.trim()) return '';
  const raw = value.trim();
  if (!isRemoteHttpUrl(raw)) return raw;

  if (window.location.origin.startsWith('http://127.0.0.1:')) {
    return `${window.location.origin}/proxy/${encodeURIComponent(raw)}`;
  }

  return raw;
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

const resolveInitialTargetPose = (forceFallback = false) => {
  if (!pendingInitialTarget) return null;

  if (!pendingInitialTarget.imageId) {
    const preferredDefaultPose = getPreferredDefaultPose();
    if (preferredDefaultPose) return preferredDefaultPose;
  }

  const resolvedPose = findPoseByInitialTarget(pendingInitialTarget);
  if (resolvedPose) return resolvedPose;

  if (!forceFallback) return null;
  if (pendingInitialTarget.imageId && !posesFetchSettled) return null;

  const fallbackMatrix = normalizeMatrixArray(pendingInitialTarget.matrix);
  if (!fallbackMatrix) return null;

  return {
    matrix: fallbackMatrix,
    image_url: pendingInitialTarget.imageId || '',
  };
};

const maybeApplyInitialTarget = (forceFallback = false) => {
  if (!pendingInitialTarget || didApplyInitialTarget) return;
  const targetPose = resolveInitialTargetPose(forceFallback);
  if (!targetPose) return;

  didApplyInitialTarget = true;
  flyToImage(targetPose);
};

const hasUsablePoseImage = (pose) => {
  const imageUrl = pose?.image_url;
  return typeof imageUrl === 'string' && imageUrl.trim().length > 0;
};

const getPreferredDefaultPose = () => {
  if (!Array.isArray(cameraPoses.value) || cameraPoses.value.length === 0) return null;

  const taggedWithImage = cameraPoses.value.find((pose) => hasUsablePoseImage(pose) && pose.tag);
  if (taggedWithImage) return taggedWithImage;

  const firstWithImage = cameraPoses.value.find((pose) => hasUsablePoseImage(pose));
  if (firstWithImage) return firstWithImage;

  return cameraPoses.value[0] || null;
};

const getPreferredCinematicPoses = () => {
  if (!Array.isArray(filteredPoses.value) || filteredPoses.value.length === 0) {
    return cameraPoses.value;
  }

  const taggedFiltered = filteredPoses.value.filter((pose) => typeof pose?.tag === 'string' && pose.tag.trim().length > 0);
  if (taggedFiltered.length >= 2) return taggedFiltered.slice(0, 12);
  if (filteredPoses.value.length >= 2) return filteredPoses.value.slice(0, 12);

  const taggedAll = cameraPoses.value.filter((pose) => typeof pose?.tag === 'string' && pose.tag.trim().length > 0);
  if (taggedAll.length >= 2) return taggedAll.slice(0, 12);

  return cameraPoses.value.slice(0, 12);
};

const maybeApplyDefaultPose = () => {
  if (pendingInitialTarget || didApplyInitialTarget || didApplyDefaultPose) return;
  const defaultPose = getPreferredDefaultPose();
  if (!defaultPose) return;

  didApplyDefaultPose = true;
  flyToImage(defaultPose);
};

const resolveIntroTargetPose = () => {
  if (pendingInitialTarget && !didApplyInitialTarget) {
    const targetPose = resolveInitialTargetPose(true);
    if (targetPose) {
      didApplyInitialTarget = true;
      return targetPose;
    }
  }

  if (!pendingInitialTarget && !didApplyDefaultPose) {
    const defaultPose = getPreferredDefaultPose();
    if (defaultPose) {
      didApplyDefaultPose = true;
      return defaultPose;
    }
  }

  return null;
};

const beginIntroAnimationToResolvedPose = () => {
  const targetPose = resolveIntroTargetPose();
  const targetCameraState = targetPose ? resolvePoseCameraState(targetPose) : null;

  if (targetPose) setActivePosePresentation(targetPose);
  if (targetCameraState?.fl_y && targetCameraState?.h) {
    sceneMetadata.value.h = targetCameraState.h;
    manualFocalPx.value = Number(targetCameraState.fl_y.toFixed(1));
    applyFocalLengthPx(targetCameraState.fl_y);
  } else {
    applyFocalLengthPx(DEFAULT_FOCAL_PX);
  }

  beginIntroAnimation(targetCameraState, targetPose);
};

const resolvePoseCameraState = (poseData) => {
  if (!viewer || !viewer.camera) return null;
  const normalizedMatrix = normalizeMatrixArray(poseData?.matrix);
  if (!normalizedMatrix) return null;

  const splatMesh = viewer.getSplatMesh();
  const rawMatrix = new THREE.Matrix4().fromArray(normalizedMatrix);
  const finalMatrix = new THREE.Matrix4();

  if (splatMesh) {
    splatMesh.updateMatrixWorld();
    finalMatrix.copy(splatMesh.matrixWorld).multiply(rawMatrix);
  } else {
    finalMatrix.copy(rawMatrix);
  }

  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  finalMatrix.decompose(position, quaternion, scale);

  return {
    position,
    quaternion,
    fl_y: Number(poseData?.fl_y || sceneMetadata.value.fl_y || 0),
    h: Number(poseData?.h || sceneMetadata.value.h || 0),
  };
};

const buildCinematicTrajectory = () => {
  const sourcePoses = getPreferredCinematicPoses();
  if (!viewer || !Array.isArray(sourcePoses) || sourcePoses.length < 2) return null;
  const worldCenter = getModelWorldCenter();

  const keyframes = sourcePoses
    .map((pose, index) => {
      const cameraState = resolvePoseCameraState(pose);
      if (!cameraState) return null;
      const uprightQuaternion = makeUprightQuaternion(cameraState.position, cameraState.quaternion, worldCenter);
      return {
        index,
        pose,
        position: cameraState.position,
        quaternion: uprightQuaternion,
        fl_y: cameraState.fl_y,
        h: cameraState.h,
      };
    })
    .filter(Boolean);

  if (keyframes.length < 2) return null;

  const dedupedKeyframes = [keyframes[0]];
  for (let i = 1; i < keyframes.length; i += 1) {
    const prev = dedupedKeyframes[dedupedKeyframes.length - 1];
    const next = keyframes[i];
    const samePoint = prev.position.distanceToSquared(next.position) < 1e-6;
    const sameAngle = Math.abs(prev.quaternion.dot(next.quaternion)) > 0.999999;
    if (samePoint && sameAngle) continue;
    dedupedKeyframes.push(next);
  }

  if (dedupedKeyframes.length < 2) return null;

  const stableKeyframes = selectStableCinematicKeyframes(dedupedKeyframes);
  if (stableKeyframes.length < 2) return null;

  const orderedKeyframes = stableKeyframes;
  const rawPositions = orderedKeyframes.map(frame => frame.position.clone());
  const smoothedPositions = smoothVectorSeries(rawPositions, cinematicSmoothness.value);
  const smoothedFocals = smoothScalarSeries(
    orderedKeyframes.map(frame => frame.fl_y || 0),
    cinematicSmoothness.value
  );
  const lookTargets = orderedKeyframes.map((frame, index) => {
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(frame.quaternion).normalize();
    const frameDistanceToCenter = Math.max(
      0.8,
      rawPositions[index].distanceTo(worldCenter)
    );
    const forwardTarget = rawPositions[index].clone().add(
      forward.multiplyScalar(Math.max(2.2, frameDistanceToCenter * 0.9))
    );

    if (!cinematicSubjectLock.value) return forwardTarget;

    return forwardTarget.lerp(
      worldCenter,
      THREE.MathUtils.clamp(0.48 + cinematicSmoothness.value * 0.26, 0, 0.9)
    );
  });
  const smoothedLookTargets = smoothVectorSeries(lookTargets, cinematicSmoothness.value);
  let totalDistance = 0;
  for (let i = 1; i < smoothedPositions.length; i += 1) {
    totalDistance += smoothedPositions[i - 1].distanceTo(smoothedPositions[i]);
  }
  const segmentCount = orderedKeyframes.length - 1;
  const durationMs = THREE.MathUtils.clamp(
    totalDistance * 1600 + segmentCount * 260,
    7000,
    42000
  ) / cinematicSpeed.value;
  const mainSegment = buildCinematicSegment({
    keyframes: orderedKeyframes,
    positions: smoothedPositions,
    targets: smoothedLookTargets,
    focals: smoothedFocals,
    durationMs,
  });

  return {
    ...mainSegment,
    worldCenter: worldCenter.clone(),
    loopBridge: buildLoopBridgeSegment(mainSegment, worldCenter),
  };
};

const sampleCinematicTrajectory = (trajectory, normalizedT) => {
  if (!trajectory) return null;

  const t = THREE.MathUtils.clamp(normalizedT, 0, 1);
  const distanceAlongPath = trajectory.totalDistance * t;
  let segmentIndex = trajectory.keyframes.length - 2;

  for (let i = 0; i < trajectory.cumulativeDistances.length - 1; i += 1) {
    if (distanceAlongPath <= trajectory.cumulativeDistances[i + 1]) {
      segmentIndex = i;
      break;
    }
  }

  const startDistance = trajectory.cumulativeDistances[segmentIndex];
  const endDistance = trajectory.cumulativeDistances[segmentIndex + 1];
  const segmentLength = Math.max(endDistance - startDistance, 1e-5);
  const localT = THREE.MathUtils.smootherstep(
    (distanceAlongPath - startDistance) / segmentLength,
    0,
    1
  );
  const from = trajectory.keyframes[segmentIndex];
  const to = trajectory.keyframes[segmentIndex + 1];
  const position = trajectory.curve.getPointAt(t);
  const stabilizedQuaternion = from.stabilizedQuaternion
    .clone()
    .slerp(to.stabilizedQuaternion, localT)
    .normalize();
  const target = from.target.clone().lerp(to.target, localT);
  const quaternion = stabilizedQuaternion;

  return {
    position,
    quaternion,
    target,
    fl_y: from.fl_y && to.fl_y ? THREE.MathUtils.lerp(from.fl_y, to.fl_y, localT) : (from.fl_y || to.fl_y || 0),
    h: from.h || to.h || sceneMetadata.value.h || 0,
    nearestPoseIndex: localT < 0.5 ? from.index : to.index,
  };
};

const applyCinematicSample = (sample) => {
  if (!sample || !viewer || !viewer.camera) return;

  const dampingAlpha = THREE.MathUtils.lerp(
    CINEMATIC_CAMERA_DAMPING_FAST,
    CINEMATIC_CAMERA_DAMPING_SLOW,
    cinematicSmoothness.value
  );

  if (!cinematicState.filteredSample) {
    cinematicState.filteredSample = {
      position: sample.position.clone(),
      quaternion: sample.quaternion.clone(),
      fl_y: Number(sample.fl_y || 0),
      h: Number(sample.h || sceneMetadata.value.h || 0),
    };
  } else {
    cinematicState.filteredSample.position.lerp(sample.position, dampingAlpha);
    cinematicState.filteredSample.quaternion.slerp(sample.quaternion, dampingAlpha).normalize();
    if (sample.fl_y) {
      cinematicState.filteredSample.fl_y = THREE.MathUtils.lerp(
        cinematicState.filteredSample.fl_y || sample.fl_y,
        sample.fl_y,
        dampingAlpha * 0.85
      );
    }
    if (sample.h) {
      cinematicState.filteredSample.h = sample.h;
    }
  }

  const cam = viewer.camera;
  cam.position.copy(cinematicState.filteredSample.position);
  cam.quaternion.copy(cinematicState.filteredSample.quaternion);

  if (cinematicState.filteredSample.fl_y && cinematicState.filteredSample.h) {
    sceneMetadata.value.h = cinematicState.filteredSample.h;
    manualFocalPx.value = Number(cinematicState.filteredSample.fl_y.toFixed(1));
    applyFocalLengthPx(cinematicState.filteredSample.fl_y);
  } else {
    renderCameraUpdate();
  }

  if (sample.nearestPoseIndex !== cinematicState.lastNearestPoseIndex) {
    cinematicState.lastNearestPoseIndex = sample.nearestPoseIndex;
    const nearestPose = cameraPoses.value[sample.nearestPoseIndex];
    if (nearestPose) {
      setActivePosePresentationState(nearestPose, { updateReference: false });
    }
  }
};

const stepCinematicPlayback = (now) => {
  if (!cinematicState.trajectory || !viewer || !viewer.camera) {
    stopCinematicPlayback({ resetProgress: false });
    return;
  }

  const activeSegment = cinematicState.phase === 'loop-bridge' && cinematicState.trajectory.loopBridge
    ? cinematicState.trajectory.loopBridge
    : cinematicState.trajectory;
  const durationMs = Math.max(activeSegment.durationMs, 1);
  const elapsedMs = Math.max(0, now - cinematicState.startTimeMs);
  cinematicState.elapsedMs = elapsedMs;

  let normalizedT = elapsedMs / durationMs;
  if (normalizedT >= 1) {
    if (cinematicState.phase === 'loop-bridge') {
      cinematicState.startTimeMs = now;
      cinematicState.elapsedMs = 0;
      cinematicState.phase = 'main';
      cinematicState.lastNearestPoseIndex = -1;
      normalizedT = 0;
    } else if (cinematicLoop.value && cinematicState.trajectory.loopBridge) {
      cinematicState.startTimeMs = now;
      cinematicState.elapsedMs = 0;
      cinematicState.phase = 'loop-bridge';
      cinematicState.lastNearestPoseIndex = -1;
      normalizedT = 0;
    } else if (cinematicLoop.value) {
      cinematicState.startTimeMs = now;
      cinematicState.elapsedMs = 0;
      cinematicState.phase = 'main';
      cinematicState.lastNearestPoseIndex = -1;
      normalizedT = 0;
    } else {
      normalizedT = 1;
    }
  }

  cinematicProgress.value = cinematicState.phase === 'main' ? normalizedT : 1;
  applyCinematicSample(sampleCinematicTrajectory(activeSegment, normalizedT));

  if (!cinematicLoop.value && cinematicState.phase === 'main' && normalizedT >= 1) {
    stopCinematicPlayback({ resetProgress: false });
    cinematicProgress.value = 1;
    return;
  }

  cinematicFrameHandle = requestAnimationFrame(stepCinematicPlayback);
};

const startCinematicPlayback = (options = {}) => {
  if (!viewer || !viewer.camera) return;
  const trajectory = buildCinematicTrajectory();
  if (!trajectory) return;

  stopCameraTweens();
  cancelCinematicFrame();
  cinematicState.trajectory = trajectory;
  cinematicState.phase = 'main';
  cinematicState.filteredSample = null;
  cinematicState.elapsedMs = options.resume ? cinematicState.elapsedMs : 0;
  cinematicState.startTimeMs = performance.now() - cinematicState.elapsedMs;
  cinematicState.lastNearestPoseIndex = -1;
  isCinematicPlaying.value = true;
  isCinematicPaused.value = false;

  if (!options.resume) {
    cinematicProgress.value = 0;
    applyCinematicSample(sampleCinematicTrajectory(trajectory, 0));
  }

  cinematicFrameHandle = requestAnimationFrame(stepCinematicPlayback);
};

const pauseCinematicPlayback = () => {
  if (!isCinematicPlaying.value) return;
  cancelCinematicFrame();
  cinematicState.elapsedMs = Math.max(0, performance.now() - cinematicState.startTimeMs);
  isCinematicPlaying.value = false;
  isCinematicPaused.value = true;
};

const toggleCinematicPlayback = () => {
  if (!canPlayCinematic.value) return;
  if (isCinematicPlaying.value) {
    pauseCinematicPlayback();
    return;
  }
  startCinematicPlayback({ resume: isCinematicPaused.value });
};

const toggleCinematicPanel = () => {
  if (!canPlayCinematic.value) return;
  showCinematicPanel.value = !showCinematicPanel.value;
};

const rebuildCinematicAtCurrentProgress = () => {
  const nextTrajectory = buildCinematicTrajectory();
  if (!nextTrajectory) return;

  cinematicState.trajectory = nextTrajectory;
  cinematicState.phase = 'main';
  cinematicState.lastNearestPoseIndex = -1;
  applyCinematicSample(sampleCinematicTrajectory(nextTrajectory, cinematicProgress.value));

  if (isCinematicPlaying.value) {
    cinematicState.elapsedMs = nextTrajectory.durationMs * cinematicProgress.value;
    cinematicState.startTimeMs = performance.now() - cinematicState.elapsedMs;
  } else if (isCinematicPaused.value) {
    cinematicState.elapsedMs = nextTrajectory.durationMs * cinematicProgress.value;
  }
};

const onCinematicSpeedChange = () => {
  cinematicSpeed.value = Number(
    THREE.MathUtils.clamp(Number(cinematicSpeed.value) || 1, 0.25, 3).toFixed(2)
  );

  if (isCinematicPlaying.value || isCinematicPaused.value) rebuildCinematicAtCurrentProgress();
};

const onCinematicStyleChange = () => {
  cinematicSmoothness.value = Number(
    THREE.MathUtils.clamp(Number(cinematicSmoothness.value) || 0.68, 0, 1).toFixed(2)
  );
  if (isCinematicPlaying.value || isCinematicPaused.value) rebuildCinematicAtCurrentProgress();
};

// --- 5. 初始化 ---
const flyToImage = (poseData, options = {}) => {
  if (!viewer || !viewer.camera) return;
  const targetCameraState = resolvePoseCameraState(poseData);
  if (!targetCameraState) {
    console.warn('[Viewer] Skip invalid pose matrix:', poseData);
    return;
  }
  if (!options.keepCinematic) interruptCinematicPlayback();
  stopInteractionInertia();

  const cam = viewer.camera;
  const targetPosition = targetCameraState.position;
  const targetQuaternion = targetCameraState.quaternion;
  setActivePosePresentation(poseData);

  // === 核心修正 3：同步真实相机的视场角 (FOV) ===
  const fl_y = targetCameraState.fl_y;
  const h = targetCameraState.h;
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

  stopCameraTweens();
  gsap.killTweensOf(animState);
  activeCameraFlightId += 1;
  const flightId = activeCameraFlightId;

  // 开始丝滑运镜
  activeCameraTween = gsap.to(animState, {
    t: 1.0,
    duration: 1.5,
    ease: "power3.inOut",
    onUpdate: () => {
      if (flightId !== activeCameraFlightId) return;
      cam.position.lerpVectors(startPos, targetPosition, animState.t);
      cam.quaternion.slerpQuaternions(startQuat, targetQuaternion, animState.t);
    },
    onComplete: () => {
      if (flightId !== activeCameraFlightId) return;
      activeCameraTween = null;
      // 记录初始飞到后的欧拉角
      const euler = new THREE.Euler().setFromQuaternion(cam.quaternion, 'YXZ');
      arrivalEuler.value = {
        x: (euler.x * 180 / Math.PI).toFixed(1),
        y: (euler.y * 180 / Math.PI).toFixed(1),
        z: (euler.z * 180 / Math.PI).toFixed(1)
      };
      rotationDelta.value = { x: 0, y: 0 }; // 飞跃新镜头时，重置手动偏差
      orbitTouchState.roll = 0;
      if (isOrbitMode.value) {
        orbitNeedsRecenterAfterPoseFlight = true;
      }
      updateDebugInfo();

      if (viewer.controls) viewer.controls.enabled = true;
    }
  });
};

const getViewerConfig = () => {
  const isMobile = isMobileDevice();
  const canUseSharedMemory = window.crossOriginIsolated === true;
  return {
    'rootElement': containerRef.value,
    'cameraUp': [0, 1, 0],
    'initialCameraPosition': [0, 0, 5],
    'initialCameraLookAt': [0, 0, 0],
    'useBuiltInControls': false,
    'gpuAcceleratedSort': canUseSharedMemory,
    'webXRMode': GaussianSplats3D.WebXRMode.None,
    'sharedMemoryForWorkers': canUseSharedMemory,
    'integerBasedSort': true,
    'halfPrecisionCovariancesOnGPU': true,
    'dynamicScene': false,
    'sphericalHarmonicsDegree': 0,
    'enableOptionalEffects': false,
    'optimizeSplatData': true,
    'freeIntermediateSplatData': true,
    'antialiased': !isMobile,
  };
};

// 当前加载的模型和位姿的 URL（供外部通过 loadModelFromFlutter 传入）
let currentPlyUrl = '/models/scene_auto_sync_raw.ply';
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
  loadingProgress.value = 0;
  loadingStatusText.value = '准备加载模型';
  loadError.value = '';
  stopCinematicPlayback();

  // 清除旧模型的视角数据和 UI 状态，防止切换模型时残留
  cameraPoses.value = [];
  activePoseId.value = '';
  activeImage.value = '';
  activeTag.value = '';
  sceneMetadata.value = {};

  // 更新 URL（如果有新传入的值）
  if (plyUrl) currentPlyUrl = plyUrl;
  if (posesUrl) currentPosesUrl = posesUrl;

  try {
    if (viewer) {
      try {
        if (viewer.renderer) {
          viewer.renderer.setAnimationLoop(null);
        }
      } catch (e) { console.warn('[Viewer] renderer cleanup:', e); }
      try {
        if (viewer.dispose) await viewer.dispose();
      } catch (e) { console.warn('[Viewer] dispose:', e); }
      viewer = null;
    }
    if (containerRef.value) {
      while (containerRef.value.firstChild) {
        containerRef.value.removeChild(containerRef.value.firstChild);
      }
    }

    animationState.isLoaded = false;
    animationState.phase = PHASE.INTRO;
    animationState.introCamera = null;
    resetIntroUniforms();
    pendingInitialTarget = null;
    didApplyInitialTarget = false;
    didApplyDefaultPose = false;
    posesFetchSettled = false;

    const config = getViewerConfig();
    viewer = new GaussianSplats3D.Viewer(config);
    window.viewer = viewer;
    manualFocalPx.value = DEFAULT_FOCAL_PX;

    // 加载模型：同名 .ksplat/.splat 存在时优先使用，失败后回退原始 PLY。
    await addSplatSceneWithFormatFallback(currentPlyUrl);

    // 加载相机位姿（支持本地路径与云端 URL）
    console.log(`[Viewer] 加载位姿: ${currentPosesUrl}`);
    loadingStatusText.value = '加载参考镜头';
    try {
      const res = await fetch(currentPosesUrl);
      const data = await res.json();
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
          imgUrl = toViewerSafeAssetUrl(imgUrl);
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
      } else {
        cameraPoses.value = data; // 兼容旧格式
      }
    } catch (err) {
      posesFetchSettled = true;
      console.error("加载位姿失败:", err);
    }

    const splatMesh = viewer.getSplatMesh();
    splatMesh.visible = false;
    createParticleSystem(splatMesh);
    applyAdvancedShader(splatMesh);

    if (initialTarget && (initialTarget.matrix || initialTarget.imageId)) {
      pendingInitialTarget = {
        matrix: initialTarget.matrix || null,
        imageId: initialTarget.imageId || null
      };
    }

    isLoading.value = false;
    if (window.BrainDanceChannel) {
      window.BrainDanceChannel.postMessage(JSON.stringify({ status: 'success', msg: '模型加载完成' }));
    }

    beginIntroAnimationToResolvedPose();
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

      if (animationState.phase === PHASE.INTRO) {
        const rawT = THREE.MathUtils.clamp(
          (performance.now() - animationState.introStartTime) / animationState.introDurationMs,
          0,
          1
        );
        const t = smoothstep01(rawT);
        const introCamera = animationState.introCamera;

        if (introCamera) {
          if (introCamera.curve) {
            viewer.camera.position.copy(introCamera.curve.getPoint(t));
          } else {
            viewer.camera.position.lerpVectors(introCamera.startPosition, introCamera.targetPosition, t);
          }
          viewer.camera.quaternion.slerpQuaternions(introCamera.startQuaternion, introCamera.targetQuaternion, t);
        }

        const pointT = smoothstep01(rawT / 0.45);
        const revealT = smoothstep01((rawT - 0.24) / 0.72);
        const splatAlpha = smoothstep01((rawT - 0.30) / 0.56);
        globalUniforms.uParticleProgress.value = pointT;
        globalUniforms.uRevealProgress.value = revealT * 1.22;
        globalUniforms.uIntroSplatAlpha.value = splatAlpha;
        globalUniforms.uGeoRadius.value = globalUniforms.uRevealProgress.value * globalUniforms.uMaxRadius.value;
        globalUniforms.uColorRadius.value = globalUniforms.uGeoRadius.value;

        const splatMesh = viewer.getSplatMesh();
        if (splatMesh && rawT >= 0.25) splatMesh.visible = true;
        if (particleSystem && particleSystem.material) {
          particleSystem.material.opacity = THREE.MathUtils.clamp(1 - ((rawT - 0.38) / 0.34), 0, 1);
          particleSystem.visible = rawT < 0.78;
        }

        if (rawT >= 1) {
          finalizeIntroAnimation();
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
  requestRender();
};

const applyFreeLookDelta = (deltaYaw, deltaPitch) => {
  if (!viewer || !viewer.camera) return;
  if (isCameraFlightLocked()) return;
  interruptCameraFlightFromUserInput();
  const cam = viewer.camera;
  reusableYawQuat.setFromAxisAngle(centerModeUp, deltaYaw);
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(cam.quaternion).normalize();
  reusablePitchQuat.setFromAxisAngle(right, deltaPitch);
  // 自由模式只改变相机朝向，不改变相机位置和模型姿态，符合第一人称查看手感。
  cam.quaternion.premultiply(reusableYawQuat).premultiply(reusablePitchQuat).normalize();
  renderCameraUpdate();
};

const startOrbitRecenterFlight = () => {
  if (!viewer || !viewer.camera || !isOrbitMode.value || !orbitNeedsRecenterAfterPoseFlight) return false;

  interruptCinematicPlayback();
  stopInteractionInertia();
  orbitNeedsRecenterAfterPoseFlight = false;
  if (viewer.controls) viewer.controls.enabled = false;

  const cam = viewer.camera;
  const targetCenter = orbitState.center.clone();
  const targetState = getOrbitCameraState();
  const startPos = cam.position.clone();
  const startQuat = cam.quaternion.clone();
  const curve = buildCameraBezierCurve(startPos, targetState.position, targetCenter);
  const animState = { t: 0 };

  stopCameraTweens();
  isOrbitRecenterFlightActive = true;
  orbitNeedsRecenterAfterPoseFlight = false;
  activeCameraFlightId += 1;
  const flightId = activeCameraFlightId;

  activeCameraTween = gsap.to(animState, {
    t: 1,
    duration: ORBIT_RECENTER_DURATION,
    ease: 'power3.inOut',
    onUpdate: () => {
      if (flightId !== activeCameraFlightId) return;
      cam.position.copy(curve.getPoint(animState.t));
      cam.quaternion.slerpQuaternions(startQuat, targetState.quaternion, animState.t).normalize();
      renderCameraUpdate();
    },
    onComplete: () => {
      if (flightId !== activeCameraFlightId) return;
      activeCameraTween = null;
      isOrbitRecenterFlightActive = false;
      cam.position.copy(targetState.position);
      cam.quaternion.copy(targetState.quaternion);
      syncOrbitTarget(targetCenter);
      applyOrbitCamera(true);
      orbitTouchState.roll = 0;
      if (viewer.controls) viewer.controls.enabled = true;
    }
  });

  return true;
};

const applyOrbitCamera = (immediate = false) => {
  if (!viewer || !viewer.camera) return;

  if (immediate) {
    orbitState.yaw = orbitState.targetYaw;
    orbitState.pitch = orbitState.targetPitch;
    orbitState.radius = orbitState.targetRadius;
  }

  const cosPitch = Math.cos(orbitState.pitch);
  const offset = new THREE.Vector3(
    Math.cos(orbitState.yaw) * cosPitch * orbitState.radius,
    Math.sin(orbitState.yaw) * cosPitch * orbitState.radius,
    Math.sin(orbitState.pitch) * orbitState.radius,
  );

  viewer.camera.position.copy(orbitState.center).add(offset);
  viewer.camera.up.copy(centerModeUp);
  viewer.camera.lookAt(orbitState.center);
  renderCameraUpdate();
};

const scheduleInteractionFrame = () => {
  if (interactionFrameHandle) return;
  interactionState.lastFrameTime = performance.now();
  interactionFrameHandle = requestAnimationFrame(stepInteractionInertia);
};

const stepInteractionInertia = (now) => {
  interactionFrameHandle = 0;
  if (!viewer || !viewer.camera) return;

  const dt = Math.min(Math.max((now - interactionState.lastFrameTime) / 1000, 1 / 120), 0.05);
  interactionState.lastFrameTime = now;
  let needsNextFrame = false;

  if (interactionState.freeInertiaActive) {
    if (
      Math.abs(interactionState.freeVelocityYaw) > 0.00001 ||
      Math.abs(interactionState.freeVelocityPitch) > 0.00001
    ) {
      applyFreeLookDelta(interactionState.freeVelocityYaw, interactionState.freeVelocityPitch);
      needsNextFrame = true;
    } else {
      interactionState.freeInertiaActive = false;
    }
  }

  if (interactionState.orbitInertiaActive) {
    if (
      Math.abs(interactionState.orbitVelocityYaw) > 0.00001 ||
      Math.abs(interactionState.orbitVelocityPitch) > 0.00001
    ) {
      orbitState.targetYaw += interactionState.orbitVelocityYaw;
      orbitState.targetPitch = clampOrbitPitch(orbitState.targetPitch + interactionState.orbitVelocityPitch);
      needsNextFrame = true;
    } else {
      interactionState.orbitInertiaActive = false;
    }
  }

  if (interactionState.zoomInertiaActive) {
    if (Math.abs(interactionState.orbitZoomVelocity) > 0.0002) {
      orbitState.targetRadius = THREE.MathUtils.clamp(
        orbitState.targetRadius * Math.exp(interactionState.orbitZoomVelocity),
        getOrbitMinRadius(),
        getOrbitMaxRadius()
      );
      needsNextFrame = true;
    } else {
      interactionState.zoomInertiaActive = false;
    }
  }

  if (isOrbitMode.value) {
    const alpha = 1 - Math.exp(-dt * 18);
    orbitState.yaw = THREE.MathUtils.lerp(orbitState.yaw, orbitState.targetYaw, alpha);
    orbitState.pitch = THREE.MathUtils.lerp(orbitState.pitch, orbitState.targetPitch, alpha);
    orbitState.radius = THREE.MathUtils.lerp(orbitState.radius, orbitState.targetRadius, alpha);
    applyOrbitCamera();
    if (
      Math.abs(orbitState.yaw - orbitState.targetYaw) > 0.00001 ||
      Math.abs(orbitState.pitch - orbitState.targetPitch) > 0.00001 ||
      Math.abs(orbitState.radius - orbitState.targetRadius) > 0.0002
    ) {
      needsNextFrame = true;
    }
  }

  const decay = Math.exp(-dt * 7.5);
  interactionState.freeVelocityYaw *= decay;
  interactionState.freeVelocityPitch *= decay;
  interactionState.orbitVelocityYaw *= decay;
  interactionState.orbitVelocityPitch *= decay;
  interactionState.orbitZoomVelocity *= decay;

  if (needsNextFrame) scheduleInteractionFrame();
};

const stopInteractionInertia = () => {
  interactionState.freeInertiaActive = false;
  interactionState.orbitInertiaActive = false;
  interactionState.zoomInertiaActive = false;
  interactionState.freeVelocityYaw = 0;
  interactionState.freeVelocityPitch = 0;
  interactionState.orbitVelocityYaw = 0;
  interactionState.orbitVelocityPitch = 0;
  interactionState.orbitZoomVelocity = 0;
  if (interactionFrameHandle) {
    cancelAnimationFrame(interactionFrameHandle);
    interactionFrameHandle = 0;
  }
};

const orbitRotate = (deltaYaw, deltaPitch) => {
  if (!viewer || !viewer.camera) return;
  if (isCameraFlightLocked()) return;
  if (isOrbitRecenterFlightActive) return;
  interruptCameraFlightFromUserInput();
  interactionState.orbitVelocityYaw = deltaYaw;
  interactionState.orbitVelocityPitch = deltaPitch;
  interactionState.orbitInertiaActive = false;
  orbitState.targetYaw += deltaYaw;
  orbitState.targetPitch = clampOrbitPitch(orbitState.targetPitch + deltaPitch);
  scheduleInteractionFrame();
};

const orbitRoll = (deltaAngleRad) => {
  if (!viewer || !viewer.camera || !Number.isFinite(deltaAngleRad)) return;
  if (isCameraFlightLocked()) return;
  if (isOrbitRecenterFlightActive) return;
  interruptCameraFlightFromUserInput();
  viewer.camera.rotateOnWorldAxis(centerModeUp, deltaAngleRad * ORBIT_ROLL_SENSITIVITY);
  syncOrbitTarget();
  renderCameraUpdate();
};

const orbitZoom = (zoomFactor) => {
  if (!viewer || !viewer.camera || !Number.isFinite(zoomFactor) || zoomFactor <= 0) return;
  if (isCameraFlightLocked()) return;
  if (isOrbitRecenterFlightActive) return;
  interruptCameraFlightFromUserInput();
  const delta = -Math.log(zoomFactor);
  interactionState.orbitZoomVelocity = delta;
  interactionState.zoomInertiaActive = false;
  orbitState.targetRadius = THREE.MathUtils.clamp(
    orbitState.targetRadius * Math.exp(delta),
    getOrbitMinRadius(),
    getOrbitMaxRadius()
  );
  scheduleInteractionFrame();
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
  if (viewer.camera) viewer.camera.up.copy(worldUp);
};

const setupOrbitControls = () => {
  if (!viewer) return;
  disposeControls();
  orbitTouchState.roll = 0;
  orbitNeedsRecenterAfterPoseFlight = false;
  syncOrbitTarget();
  applyOrbitCamera(true);
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

  stopInteractionInertia();
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
  viewer.camera.up.copy(centerModeUp);
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

const resetManualCameraInputState = () => {
  isDragging.value = false;
  pinchState.active = false;
  pinchState.distance = 0;
  orbitTouchState.active = false;
  orbitTouchState.angle = 0;
};

const handleFreePinchMove = (touches) => {
  if (!touches || touches.length < 2) return false;

  const nextDistance = getTouchDistance(touches[0], touches[1]);
  if (!Number.isFinite(nextDistance) || nextDistance <= 0) return true;

  if (pinchState.active && pinchState.distance > 0) {
    const distanceDelta = nextDistance - pinchState.distance;
    const normalizedDelta = distanceDelta / Math.max(pinchState.distance, 80);
    const scaleFactor = THREE.MathUtils.clamp(
      Math.exp(normalizedDelta * PINCH_ZOOM_STEP),
      0.72,
      1.38
    );
    zoomByFocalScale(scaleFactor);
  }

  // Flutter WebView 有时不会可靠派发“第二根手指按下”的 touchstart，
  // 因此双指 touchmove 必须也能兜底进入 pinch 状态。
  pinchState.active = true;
  pinchState.distance = nextDistance;
  isDragging.value = false;
  return true;
};

// --- 简单拖拽微调逻辑 ---
const onMouseDown = (e) => {
  if (isCameraFlightLocked()) {
    resetManualCameraInputState();
    return;
  }
  if (isOrbitMode.value && startOrbitRecenterFlight()) return;
  interruptCinematicPlayback();
  interruptCameraFlight();
  stopInteractionInertia();
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
  if (isCameraFlightLocked()) return;
  if (isOrbitRecenterFlightActive) return;
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

  const deltaYaw = dx * FREE_LOOK_SENSITIVITY;
  const deltaPitch = -dy * FREE_LOOK_SENSITIVITY;
  interactionState.freeVelocityYaw = deltaYaw;
  interactionState.freeVelocityPitch = deltaPitch;
  applyFreeLookDelta(deltaYaw, deltaPitch);

  lastMouse.x = e.clientX;
  lastMouse.y = e.clientY;
};

const onMouseUp = () => {
  if (isCameraFlightLocked()) {
    resetManualCameraInputState();
    return;
  }
  if (isOrbitRecenterFlightActive) return;
  if (isOrbitMode.value) {
    if (isDragging.value) {
      interactionState.orbitInertiaActive = true;
      scheduleInteractionFrame();
    }
    isDragging.value = false;
    pinchState.active = false;
    orbitTouchState.active = false;
    return;
  }
  if (isDragging.value) {
    interactionState.freeInertiaActive = true;
    scheduleInteractionFrame();
  }
  isDragging.value = false;
  pinchState.active = false;
};

const onWheel = (e) => {
  if (!viewer || !viewer.camera) return;
  if (isCameraFlightLocked()) return;
  if (isOrbitMode.value && startOrbitRecenterFlight()) return;
  interruptCinematicPlayback();
  interruptCameraFlight();
  if (isOrbitMode.value) {
    const zoomFactor = e.deltaY < 0 ? (1 + WHEEL_ZOOM_STEP) : (1 / (1 + WHEEL_ZOOM_STEP));
    orbitZoom(zoomFactor);
    interactionState.zoomInertiaActive = true;
    scheduleInteractionFrame();
    return;
  }
  const direction = e.deltaY < 0 ? (1 + WHEEL_ZOOM_STEP) : (1 / (1 + WHEEL_ZOOM_STEP));
  zoomByFocalScale(direction);
};

// --- 移动端 Touch 事件支持 ---
const onTouchStart = (e) => {
  if (isCameraFlightLocked()) {
    resetManualCameraInputState();
    return;
  }
  if (isOrbitMode.value && startOrbitRecenterFlight()) return;
  interruptCinematicPlayback();
  interruptCameraFlight();
  stopInteractionInertia();
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
  if (isCameraFlightLocked()) return;
  if (isOrbitRecenterFlightActive) return;
  if (isOrbitMode.value) {
    if (!viewer || !viewer.camera || e.touches.length === 0) return;
    if (isOrbitRecenterFlightActive || startOrbitRecenterFlight()) return;

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
    handleFreePinchMove(e.touches);
    return;
  }

  if (!isDragging.value) return;

  const dx = e.touches[0].clientX - lastMouse.x;
  const dy = e.touches[0].clientY - lastMouse.y;

  const deltaYaw = dx * FREE_LOOK_SENSITIVITY;
  const deltaPitch = -dy * FREE_LOOK_SENSITIVITY;
  interactionState.freeVelocityYaw = deltaYaw;
  interactionState.freeVelocityPitch = deltaPitch;
  rotationDelta.value.x += THREE.MathUtils.radToDeg(deltaPitch);
  rotationDelta.value.y += THREE.MathUtils.radToDeg(deltaYaw);
  applyFreeLookDelta(deltaYaw, deltaPitch);

  lastMouse.x = e.touches[0].clientX;
  lastMouse.y = e.touches[0].clientY;
};

const onTouchEnd = (e) => {
  if (isCameraFlightLocked()) {
    resetManualCameraInputState();
    return;
  }
  if (isOrbitRecenterFlightActive) return;
  if (isOrbitMode.value) {
    const wasPinching = pinchState.active;
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
    if (isDragging.value) {
      interactionState.orbitInertiaActive = true;
      scheduleInteractionFrame();
    }
    if (wasPinching) {
      interactionState.zoomInertiaActive = true;
      scheduleInteractionFrame();
    }
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
  if (isDragging.value) {
    interactionState.freeInertiaActive = true;
    scheduleInteractionFrame();
  }
  isDragging.value = false;

  if (e.touches.length === 1) {
    lastMouse.x = e.touches[0].clientX;
    lastMouse.y = e.touches[0].clientY;
    isDragging.value = true;
  }
};

const onCapturedUserCameraInput = () => {
  if (isCameraFlightLocked()) {
    resetManualCameraInputState();
    return;
  }
  if (isOrbitRecenterFlightActive) return;
  interruptCameraFlightFromUserInput();
};

function onTimePeelingSelect(model) {
  activeModelId.value = model.id;
  // 通知 Flutter 切换模型（Flutter 负责下载后回调 loadModelFromFlutter）
  if (window.BrainDanceChannel) {
    window.BrainDanceChannel.postMessage(JSON.stringify({
      action: 'switchModel',
      modelId: model.id,
      ply: model.ply || '',
      poses: model.poses || '',
    }));
  } else {
    // 非 Flutter 环境，直接加载
    isLoading.value = false;
    stopCinematicPlayback();
    initViewer(model.ply || null, model.poses || null, null);
  }
}

onMounted(() => {
  if (containerRef.value) {
    checkProtocol();

    // 注册供Flutter调用的模型列表设置函数
    window.setModelListForTimePeeling = (list, currentId) => {
      console.log('[Flutter->WebGL] 收到模型列表:', list, '当前模型:', currentId);
      if (Array.isArray(list)) {
        modelList.value = list;
        if (currentId) {
          activeModelId.value = currentId;
        } else if (list.length > 0 && !activeModelId.value) {
          activeModelId.value = list[0].id || '';
        }
      }
    };

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

    // 注册供Flutter调用的主题切换函数
    window.setThemeFromFlutter = (theme) => {
      const container = document.querySelector('.app-container');
      if (container) {
        container.setAttribute('data-theme', theme);
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
    window.addEventListener('pointerdown', onCapturedUserCameraInput, true);
    window.addEventListener('pointermove', onCapturedUserCameraInput, true);
    window.addEventListener('touchstart', onCapturedUserCameraInput, true);
    window.addEventListener('touchmove', onCapturedUserCameraInput, true);
    window.addEventListener('wheel', onCapturedUserCameraInput, true);
  }
});

onBeforeUnmount(async () => {
  window.removeEventListener('mousedown', onMouseDown);
  window.removeEventListener('mousemove', onMouseMove);
  window.removeEventListener('mouseup', onMouseUp);
  window.removeEventListener('pointerdown', onCapturedUserCameraInput, true);
  window.removeEventListener('pointermove', onCapturedUserCameraInput, true);
  window.removeEventListener('touchstart', onCapturedUserCameraInput, true);
  window.removeEventListener('touchmove', onCapturedUserCameraInput, true);
  window.removeEventListener('wheel', onCapturedUserCameraInput, true);
  stopInteractionInertia();
  stopCinematicPlayback();

  if (viewer) {
    try {
      if (viewer.renderer) viewer.renderer.setAnimationLoop(null);
    } catch (_) {}
    try {
      await viewer.dispose();
    } catch (_) {}
    viewer = null;
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

    <BottomSelector
      v-if="showBottomSelector"
      :models="modelList"
      :activeModelId="activeModelId"
      :poses="filteredPoses"
      :activePoseId="activePoseId"
      :searchQuery="searchQuery"
      :getPosePresentationId="getPosePresentationId"
      :hasModels="hasModelTab"
      :hasPoses="hasPoseTab"
      @selectModel="onTimePeelingSelect"
      @selectPose="flyToImage"
    />

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
            中心模式
          </button>
        </div>
        <button class="archive-btn archive-btn--ghost focal-settings-toggle" @click="toggleFocalSettings"
          @mousedown.stop @touchstart.stop @touchend.stop>
          {{ showFocalSettings ? '收起焦距' : '焦距设置' }}
        </button>
        <button v-if="canPlayCinematic" class="cinematic-trigger archive-btn archive-btn--ghost"
          :class="{ active: showCinematicPanel }" @click="toggleCinematicPanel"
          @mousedown.stop @touchstart.stop @touchend.stop>
          <span class="cinematic-trigger-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" focusable="false">
              <path
                d="M4 7.5a1.5 1.5 0 0 1 1.5-1.5h7A1.5 1.5 0 0 1 14 7.5v9a1.5 1.5 0 0 1-1.5 1.5h-7A1.5 1.5 0 0 1 4 16.5v-9Zm11 2.1 4.83-2.76A.75.75 0 0 1 21 7.5v9a.75.75 0 0 1-1.17.66L15 14.4V9.6Z" />
            </svg>
          </span>
          <span>运镜</span>
        </button>
        <div class="cinematic-panel archive-card" v-if="canPlayCinematic && showCinematicPanel"
          @mousedown.stop @touchstart.stop @touchmove.stop @touchend.stop @touchcancel.stop>
          <div class="cinematic-head">
            <div>
              <div class="eyebrow">Camera Move</div>
              <div class="cinematic-title">自动运镜</div>
            </div>
            <div class="cinematic-head-actions">
              <label class="cinematic-loop-toggle">
                <input type="checkbox" v-model="cinematicLoop" />
                <span>循环</span>
              </label>
              <button class="cinematic-close" @click="showCinematicPanel = false" aria-label="收起运镜面板">
                ×
              </button>
            </div>
          </div>
          <div class="cinematic-actions">
            <button class="archive-btn archive-btn--solid cinematic-primary" @click="toggleCinematicPlayback">
              {{ cinematicButtonLabel }}
            </button>
            <button class="archive-btn archive-btn--ghost cinematic-secondary"
              @click="stopCinematicPlayback()"
              :disabled="!isCinematicPlaying && !isCinematicPaused && cinematicProgress === 0">
              停止
            </button>
          </div>
          <div class="cinematic-progress-row">
            <span>进度</span>
            <span>{{ Math.round(cinematicProgress * 100) }}%</span>
          </div>
          <input class="cinematic-progress" type="range" :value="cinematicProgress * 100" min="0" max="100"
            step="1" disabled />
          <div class="cinematic-progress-row">
            <span>速度</span>
            <span>{{ cinematicSpeed.toFixed(2) }}x</span>
          </div>
          <input class="cinematic-speed" type="range" v-model.number="cinematicSpeed" min="0.25" max="3" step="0.05"
            @input="onCinematicSpeedChange" />
          <div class="cinematic-progress-row">
            <span>平滑</span>
            <span>{{ Math.round(cinematicSmoothness * 100) }}%</span>
          </div>
          <input class="cinematic-speed" type="range" v-model.number="cinematicSmoothness" min="0" max="1"
            step="0.05" @input="onCinematicStyleChange" />
          <label class="cinematic-focus-toggle">
            <input type="checkbox" v-model="cinematicSubjectLock" @change="onCinematicStyleChange" />
            <span>主体锁定</span>
          </label>
        </div>
        <div class="fps-counter" v-if="currentFps > 0">FPS {{ currentFps }}</div>
      </div>
    </div>

    <div v-if="isLoading" class="loading-overlay">
      <div class="loading-card">
        <div class="loading-dot"></div>
        <div class="loading-title">场景正在展开</div>
        <div class="loading-copy">{{ loadingStatusText }}</div>
        <div class="loading-progress" aria-hidden="true">
          <div class="loading-progress-fill" :style="{ width: `${Math.round(loadingProgress * 100)}%` }"></div>
        </div>
        <div class="loading-percent">{{ Math.round(loadingProgress * 100) }}%</div>
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
  --flutter-safe-top: 92px;
  --flutter-safe-left: 14px;
  --flutter-safe-right: 154px;

  --bg-gradient-1: rgba(228, 232, 237, 0.16);
  --bg-gradient-2: rgba(107, 122, 143, 0.14);
  --bg-start: #f4f3ee;
  --bg-end: #e6e3db;
  --text-primary: #1e1e20;
  --text-secondary: rgba(30, 30, 32, 0.72);
  --text-muted: rgba(30, 30, 32, 0.48);
  --card-bg: rgba(249, 249, 248, 0.84);
  --card-border: rgba(107, 122, 143, 0.16);
  --card-shadow: rgba(0, 0, 0, 0.06);
  --input-bg: rgba(255, 255, 255, 0.72);
  --input-border: rgba(107, 122, 143, 0.14);
  --input-focus-border: rgba(107, 122, 143, 0.5);
  --input-focus-ring: rgba(107, 122, 143, 0.08);
  --btn-ghost-bg: rgba(249, 249, 248, 0.84);
  --btn-solid-bg: #6b7a8f;
  --btn-solid-hover: #5e6d81;
  --btn-solid-text: #f9f9f8;
  --chip-active-bg: #1e1e20;
  --chip-active-text: #f5f4ef;
  --chip-hover-bg: rgba(107, 122, 143, 0.12);
  --chip-hover-text: #273142;
  --eyebrow-color: #6b7a8f;
  --accent: #6d8260;
  --accent-ring: rgba(109, 130, 96, 0.12);
  --error-title: #8b4747;
  --overlay-bg: rgba(30, 30, 32, 0.24);
  --range-accent: #6b7a8f;
  --vignette-color: rgba(30, 30, 32, 0.12);
  --info-tag-bg: rgba(228, 232, 237, 0.78);
  --fps-bg: rgba(249, 249, 248, 0.84);
  --close-btn-bg: rgba(107, 122, 143, 0.1);
  --cinematic-loop-text: rgba(30, 30, 32, 0.7);
  --cinematic-focus-text: rgba(30, 30, 32, 0.78);
  --loading-copy-text: rgba(30, 30, 32, 0.66);
  --error-msg-text: rgba(30, 30, 32, 0.68);
  --chip-inactive-text: #6b7280;

  position: relative;
  width: 100vw;
  height: 100vh;
  background:
    radial-gradient(circle at top left, var(--bg-gradient-1), transparent 24%),
    radial-gradient(circle at top right, var(--bg-gradient-2), transparent 28%),
    linear-gradient(180deg, var(--bg-start) 0%, var(--bg-end) 100%);
  overflow: hidden;
  color: var(--text-primary);
  font-family: 'HarmonyOS Sans SC', 'Microsoft YaHei', 'PingFang SC', sans-serif;
}

.app-container[data-theme="dark"] {
  --bg-gradient-1: rgba(30, 35, 45, 0.3);
  --bg-gradient-2: rgba(50, 60, 80, 0.2);
  --bg-start: #101014;
  --bg-end: #18181c;
  --text-primary: #f5f7fa;
  --text-secondary: rgba(245, 247, 250, 0.72);
  --text-muted: rgba(245, 247, 250, 0.48);
  --card-bg: rgba(30, 30, 34, 0.84);
  --card-border: rgba(174, 186, 204, 0.12);
  --card-shadow: rgba(0, 0, 0, 0.22);
  --input-bg: rgba(35, 35, 42, 0.72);
  --input-border: rgba(174, 186, 204, 0.14);
  --input-focus-border: rgba(174, 186, 204, 0.5);
  --input-focus-ring: rgba(174, 186, 204, 0.08);
  --btn-ghost-bg: rgba(30, 30, 34, 0.84);
  --btn-solid-bg: #aebacc;
  --btn-solid-hover: #9aa8bc;
  --btn-solid-text: #101014;
  --chip-active-bg: #f5f7fa;
  --chip-active-text: #101014;
  --chip-hover-bg: rgba(174, 186, 204, 0.12);
  --chip-hover-text: #d0d8e4;
  --eyebrow-color: #aebacc;
  --accent: #8fae7f;
  --accent-ring: rgba(143, 174, 127, 0.12);
  --error-title: #ff6b6b;
  --overlay-bg: rgba(0, 0, 0, 0.5);
  --range-accent: #aebacc;
  --vignette-color: rgba(0, 0, 0, 0.2);
  --info-tag-bg: rgba(50, 55, 65, 0.78);
  --fps-bg: rgba(30, 30, 34, 0.84);
  --close-btn-bg: rgba(174, 186, 204, 0.12);
  --cinematic-loop-text: rgba(245, 247, 250, 0.7);
  --cinematic-focus-text: rgba(245, 247, 250, 0.78);
  --loading-copy-text: rgba(245, 247, 250, 0.66);
  --error-msg-text: rgba(245, 247, 250, 0.68);
  --chip-inactive-text: #9ca3af;
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
    linear-gradient(180deg, var(--vignette-color), transparent 18%, transparent 78%, var(--vignette-color)),
    radial-gradient(circle at center, transparent 55%, var(--vignette-color) 100%);
  z-index: 1;
}

.eyebrow {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--eyebrow-color);
}

.archive-card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 22px;
  box-shadow: 0 10px 26px var(--card-shadow);
  backdrop-filter: blur(18px);
}

.top-hud {
  position: absolute;
  top: calc(var(--flutter-safe-top) + 56px);
  left: var(--flutter-safe-left);
  right: auto;
  width: min(520px, calc(100vw - var(--flutter-safe-left) - var(--flutter-safe-right)));
  z-index: 120;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 12px;
}

.top-actions {
  display: flex;
  width: auto;
  max-width: 100%;
  align-items: center;
  gap: 8px;
  flex: 0 0 auto;
  align-self: flex-start;
  justify-content: flex-start;
  flex-wrap: wrap;
}

.view-mode-switch {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px;
  border-radius: 18px;
}

.cinematic-trigger {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-width: auto;
  padding-inline: 12px;
}

.cinematic-trigger.active {
  background: var(--chip-active-bg);
  color: var(--chip-active-text);
  border-color: var(--card-border);
}

.cinematic-trigger-icon {
  display: inline-flex;
  width: 16px;
  height: 16px;
}

.cinematic-trigger-icon svg {
  width: 100%;
  height: 100%;
  fill: currentColor;
}

.cinematic-panel {
  width: min(84vw, 280px);
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.cinematic-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.cinematic-head-actions {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.cinematic-title {
  font-size: 15px;
  font-weight: 700;
}

.cinematic-loop-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--cinematic-loop-text);
}

.cinematic-close {
  appearance: none;
  border: 0;
  background: var(--close-btn-bg);
  color: var(--text-primary);
  width: 26px;
  height: 26px;
  border-radius: 999px;
  cursor: pointer;
  font-size: 18px;
  line-height: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.cinematic-close:hover {
  background: var(--chip-hover-bg);
}

.cinematic-focus-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--cinematic-focus-text);
}

.cinematic-actions {
  display: flex;
  gap: 8px;
}

.cinematic-primary,
.cinematic-secondary {
  flex: 1 1 0;
  justify-content: center;
}

.cinematic-progress-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  color: var(--text-secondary);
}

.cinematic-progress,
.cinematic-speed {
  width: 100%;
  accent-color: var(--accent);
}

.cinematic-progress[disabled] {
  opacity: 0.8;
}

.mode-chip {
  appearance: none;
  border: 0;
  background: transparent;
  color: var(--chip-inactive-text);
  padding: 8px 12px;
  border-radius: 12px;
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.2s ease, color 0.2s ease, transform 0.2s ease;
}

.mode-chip.active {
  background: var(--chip-active-bg);
  color: var(--chip-active-text);
  box-shadow: 0 8px 18px var(--card-shadow);
}

.mode-chip:not(.active):hover {
  background: var(--chip-hover-bg);
  color: var(--chip-hover-text);
}

.archive-btn {
  appearance: none;
  border-radius: 14px;
  border: 1px solid var(--card-border);
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
  box-shadow: 0 8px 18px var(--card-shadow);
}

.archive-btn--ghost {
  background: var(--btn-ghost-bg);
  color: var(--text-primary);
}

.archive-btn--solid {
  background: var(--btn-solid-bg);
  border-color: var(--btn-solid-bg);
  color: var(--btn-solid-text);
}

.archive-btn--solid:hover {
  background: var(--btn-solid-hover);
  border-color: var(--btn-solid-hover);
}

.controls-ui {
  position: absolute;
  top: calc(var(--flutter-safe-top) + 48px);
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 15px;
  z-index: 100;
}

.loading-overlay {
  position: absolute;
  inset: 0;
  background: var(--overlay-bg);
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
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  box-shadow: 0 18px 34px var(--card-shadow);
  text-align: center;
}

.loading-dot {
  width: 12px;
  height: 12px;
  margin: 0 auto 14px;
  border-radius: 999px;
  background: var(--accent);
  box-shadow: 0 0 0 10px var(--accent-ring);
  animation: pulse 1.8s ease-in-out infinite;
}

.loading-title {
  font-size: 20px;
  font-weight: 600;
}

.loading-copy {
  margin-top: 6px;
  font-size: 13px;
  color: var(--loading-copy-text);
}

.loading-progress {
  width: 100%;
  height: 6px;
  margin-top: 14px;
  border-radius: 999px;
  background: var(--input-bg);
  overflow: hidden;
}

.loading-progress-fill {
  height: 100%;
  width: 0;
  border-radius: inherit;
  background: var(--accent);
  transition: width 160ms ease-out;
}

.loading-percent {
  margin-top: 8px;
  font-size: 12px;
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}

.error-overlay {
  position: absolute;
  inset: 0;
  background: var(--overlay-bg);
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
  color: var(--error-title);
}

.error-msg {
  font-size: 13px;
  color: var(--error-msg-text);
  max-width: 320px;
  word-break: break-all;
  margin-bottom: 20px;
}

button {
  font-family: inherit;
}

button.active {
  background: var(--btn-solid-bg);
  border-color: var(--btn-solid-bg);
}

/* 搜索栏样式 */
.search-panel {
  position: static;
  z-index: auto;
  width: 100%;
  max-width: none;
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
  border: 1px solid var(--input-border);
  border-radius: 12px;
  background: var(--input-bg);
  outline: none;
  font-size: 13px;
  color: var(--text-primary);
}

.search-input:focus {
  border-color: var(--input-focus-border);
  box-shadow: 0 0 0 4px var(--input-focus-ring);
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
  top: calc(var(--flutter-safe-top) + 130px);
  right: var(--flutter-safe-right);
  z-index: 120;
  width: 236px;
  background: var(--card-bg);
  color: var(--text-primary);
  border: 1px solid var(--card-border);
  border-radius: 20px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  box-shadow: 0 16px 28px var(--card-shadow);
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
  color: var(--text-secondary);
}

.focal-number-input {
  width: 100px;
  border-radius: 10px;
  border: 1px solid var(--card-border);
  padding: 8px 10px;
  background: var(--input-bg);
}

.focal-reset-btn {
  width: 100%;
}

/* 参考图浮窗 */
.reference-overlay {
  position: absolute;
  top: calc(var(--flutter-safe-top) + 56px);
  right: 14px;
  width: min(22vw, 148px);
  min-width: 112px;
  background: var(--card-bg);
  padding: 8px;
  border-radius: 16px;
  border: 1px solid var(--card-border);
  z-index: 150;
  cursor: pointer;
  box-shadow: 0 12px 24px var(--card-shadow);
  backdrop-filter: blur(16px);
}

.ref-title {
  font-size: 12px;
  color: var(--text-primary);
  margin: 2px 0 6px;
  font-weight: 600;
}

.ref-img {
  width: 100%;
  border-radius: 10px;
  border: 1px solid var(--card-border);
  margin-bottom: 6px;
}

.ref-info {
  font-size: 9px;
  color: var(--text-secondary);
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 4px;
}

.info-tag {
  background: var(--info-tag-bg);
  padding: 3px 6px;
  border-radius: 999px;
}

.info-tag--accent {
  color: var(--accent);
}

.ref-hint {
  font-size: 9px;
  color: var(--text-muted);
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
  color: var(--text-primary);
  background: var(--fps-bg);
  border: 1px solid var(--card-border);
  border-radius: 12px;
  padding: 8px 10px;
  font-family: monospace;
  font-size: 12px;
  pointer-events: none;
}

input[type='range'] {
  accent-color: var(--range-accent);
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
  .app-container {
    --flutter-safe-top: 84px;
    --flutter-safe-left: 12px;
    --flutter-safe-right: 144px;
  }

  .top-hud {
    left: var(--flutter-safe-left);
    right: auto;
    width: min(520px, calc(100vw - var(--flutter-safe-left) - var(--flutter-safe-right)));
    gap: 8px;
  }

  .top-actions {
    width: auto;
    max-width: 100%;
    align-self: flex-start;
    justify-content: flex-start;
    gap: 8px;
  }

  .view-mode-switch {
    padding: 4px;
    gap: 4px;
  }

  .cinematic-panel {
    width: 100%;
  }

  .cinematic-trigger {
    justify-content: center;
  }

  .mode-chip {
    padding: 8px 10px;
    font-size: 12px;
  }

  .search-panel {
    width: 100%;
    max-width: calc(100vw - var(--flutter-safe-left) - var(--flutter-safe-right));
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
    top: calc(var(--flutter-safe-top) + 48px);
    right: 12px;
    width: 112px;
    min-width: 112px;
    padding: 7px;
  }

  .focal-settings-panel {
    top: calc(var(--flutter-safe-top) + 122px);
  }
}

@media (prefers-color-scheme: dark) {
  .app-container:not([data-theme]) {
    --bg-gradient-1: rgba(30, 35, 45, 0.3);
    --bg-gradient-2: rgba(50, 60, 80, 0.2);
    --bg-start: #101014;
    --bg-end: #18181c;
    --text-primary: #f5f7fa;
    --text-secondary: rgba(245, 247, 250, 0.72);
    --text-muted: rgba(245, 247, 250, 0.48);
    --card-bg: rgba(30, 30, 34, 0.84);
    --card-border: rgba(174, 186, 204, 0.12);
    --card-shadow: rgba(0, 0, 0, 0.22);
    --input-bg: rgba(35, 35, 42, 0.72);
    --input-border: rgba(174, 186, 204, 0.14);
    --input-focus-border: rgba(174, 186, 204, 0.5);
    --input-focus-ring: rgba(174, 186, 204, 0.08);
    --btn-ghost-bg: rgba(30, 30, 34, 0.84);
    --btn-solid-bg: #aebacc;
    --btn-solid-hover: #9aa8bc;
    --btn-solid-text: #101014;
    --chip-active-bg: #f5f7fa;
    --chip-active-text: #101014;
    --chip-hover-bg: rgba(174, 186, 204, 0.12);
    --chip-hover-text: #d0d8e4;
    --eyebrow-color: #aebacc;
    --accent: #8fae7f;
    --accent-ring: rgba(143, 174, 127, 0.12);
    --error-title: #ff6b6b;
    --overlay-bg: rgba(0, 0, 0, 0.5);
    --range-accent: #aebacc;
    --vignette-color: rgba(0, 0, 0, 0.2);
    --info-tag-bg: rgba(50, 55, 65, 0.78);
    --fps-bg: rgba(30, 30, 34, 0.84);
    --close-btn-bg: rgba(174, 186, 204, 0.12);
    --cinematic-loop-text: rgba(245, 247, 250, 0.7);
    --cinematic-focus-text: rgba(245, 247, 250, 0.78);
    --loading-copy-text: rgba(245, 247, 250, 0.66);
    --error-msg-text: rgba(245, 247, 250, 0.68);
    --chip-inactive-text: #9ca3af;
  }
}
</style>
