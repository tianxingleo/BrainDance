<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import * as THREE from 'three';
import gsap from 'gsap';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { BrainDanceCameraRig } from '../lib/interaction/BrainDanceCameraRig';
import { classifyInteractionProfile } from '../lib/interaction/classifySceneProfile';
import { PoseGraph } from '../lib/interaction/poseGraph';
import { GestureHandler } from '../lib/interaction/gestures';
import TopHud from './TopHud.vue';
import FocalPanel from './FocalPanel.vue';
import StatusRibbon from './StatusRibbon.vue';
import CameraTrack from './CameraTrack.vue';
import ReferenceCard from './ReferenceCard.vue';
import {
  DEFAULT_FOCAL_PX,
  DEFAULT_SCENE_RADIUS,
  calcFocalFromFov,
  calcFovFromFocal,
  clampFocalPx,
} from '../lib/cameraMath';
import {
  deriveHighlightPointFromPose,
  findPoseByInitialTarget,
  normalizeMatrixArray,
  parseInitialInputFromUrl,
  resolveImageUrl,
} from '../lib/poseUtils';
import {
  createClipPlaneEffect,
  createSphereHighlightEffect,
  updateClipPlaneEffect,
  updateSphereHighlight,
} from '../lib/sparkEffects';
import { notifyFlutter } from '../lib/viewerBridge';

const containerRef = ref(null);
const cameraPoses = ref([]);
const searchQuery = ref('');
const activeImage = ref('');
const activeTag = ref('');
const sceneMetadata = ref({});
const loadError = ref('');
const isLoading = ref(false);
const currentFps = ref(0);
const showFocalSettings = ref(false);
const currentViewFov = ref(0);
const currentViewFocalPx = ref(0);
const manualFocalPx = ref(null);
const highlightEnabled = ref(true);
const highlightStatus = ref('待命');
const clipEnabled = ref(false);
const clipOffset = ref(0);
const currentModelUrl = ref('./models/scene_auto_sync_raw.ply');
const currentPosesPath = ref('/models/webgl_poses_with_tags.json');

// ==================== Orbit 相机模式 ====================
// Orbit 模式：相机绕模型中心自动旋转（圆周运动），不改变现有手动控制逻辑
const orbitEnabled = ref(false);    // 是否开启 orbit 模式
const orbitPaused = ref(false);     // 是否暂停旋转
const orbitSpeed = ref(20);         // 旋转速度，单位：度/秒
const orbitDirection = ref(1);      // 旋转方向：1=逆时针(CCW)，-1=顺时针(CW)
const orbitRadius = ref(0);         // 旋转半径：0=自动计算（保持当前距离），>0 使用指定值

const filteredPoses = computed(() => {
  const query = searchQuery.value.trim().toLowerCase();
  if (!query) {
    const withTags = cameraPoses.value.filter((pose) => pose.tag);
    return withTags.length > 0 ? withTags : cameraPoses.value.slice(0, 60);
  }
  return cameraPoses.value.filter((pose) => {
    const tag = typeof pose.tag === 'string' ? pose.tag.toLowerCase() : '';
    return tag.includes(query);
  });
});

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

let scene = null;
let camera = null;
let renderer = null;
let spark = null;
let splatMesh = null;
let highlightEffect = null;
let clipEffect = null;
let sceneCenter = new THREE.Vector3(0, 0, 0);
let sceneRadius = DEFAULT_SCENE_RADIUS;
let resizeHandler = null;
let pendingInitialTarget = null;
let didApplyInitialTarget = false;
let posesFetchSettled = false;
let hasInitializedFromExternalInput = false;
let fpsFrames = 0;
let fpsTimestamp = 0;
let clock = new THREE.Clock();

// ── Spark 2.0 interaction state ──
let cameraRig = null;
let poseGraph = null;
let gestureHandler = null;
const interactionProfile = ref('hybrid');
const profileConfidence = ref(0);
const focusPointArr = ref([]);
const qualityMode = ref('standard');
const currentPoseIndex = ref(-1);

// ===== Orbit 运行时变量（非响应式，避免不必要的 Vue 重渲染） =====
let orbitAngle = 0;           // 当前旋转角度（弧度）
let orbitY = 0;               // 相机 Y 坐标（保持 orbit 高度不变）
let autoOrbitRadius = 0;      // 自动计算的轨道半径（从当前相机距离获取）
let orbitLastFrameTime = performance.now();
let isOrbitDragging = false;  // 用户正在手动拖拽（临时中断 orbit）

const refreshCurrentFocalInfo = () => {
  if (!camera) return;
  const h = sceneMetadata.value.h || containerRef.value?.clientHeight || window.innerHeight;
  currentViewFov.value = Number(camera.fov || 0);
  if (h && currentViewFov.value > 0 && currentViewFov.value < 179) {
    const focal = calcFocalFromFov(currentViewFov.value, h);
    currentViewFocalPx.value = focal ? Number(focal.toFixed(1)) : 0;
  }
};

const renderOnce = () => {
  if (!renderer || !scene || !camera) return;
  renderer.render(scene, camera);
  refreshCurrentFocalInfo();
};

const syncHighlight = (point, radius = null) => {
  updateSphereHighlight(highlightEffect, {
    enabled: highlightEnabled.value,
    point,
    radius,
  });
};

const syncClipPlane = () => {
  if (!clipEffect) return;
  const normal = new THREE.Vector3(1, 0, 0);
  const point = sceneCenter.clone().add(new THREE.Vector3(sceneRadius * clipOffset.value, 0, 0));
  updateClipPlaneEffect(clipEffect, {
    enabled: clipEnabled.value,
    point,
    normal,
  });
};

// ==================== Orbit 相机模式：核心函数 ====================

// 从当前相机位置同步 orbit 参数（角度、高度、半径）
// 用于：(1) 开启 orbit 时初始化 (2) 手动拖拽松开后恢复 (3) 模型加载后重新锚定
const syncOrbitFromCamera = () => {
  if (!camera || !sceneCenter) return;
  const dx = camera.position.x - sceneCenter.x;
  const dz = camera.position.z - sceneCenter.z;
  orbitAngle = Math.atan2(dz, dx);
  orbitY = camera.position.y;
  autoOrbitRadius = Math.sqrt(dx * dx + dz * dz);
};

// 开启 orbit 模式：从当前相机位置初始化轨道参数并开始旋转
const startOrbit = () => {
  if (!sceneCenter) return;
  syncOrbitFromCamera();
  // 如果自动半径过小（相机太靠近中心），使用默认距离
  if (autoOrbitRadius < 0.1) {
    autoOrbitRadius = sceneRadius * 2.4;
  }
  orbitPaused.value = false;
  orbitEnabled.value = true;
};

// 关闭 orbit 模式：相机停留在当前位置，恢复原有手动控制
const stopOrbit = () => {
  orbitEnabled.value = false;
  orbitPaused.value = false;
};

// 暂停/恢复 orbit 旋转
const toggleOrbitPause = () => {
  if (!orbitEnabled.value) return;
  if (orbitPaused.value) {
    // 恢复时从当前相机位置重新同步轨道参数，确保无缝衔接
    syncOrbitFromCamera();
    orbitPaused.value = false;
  } else {
    orbitPaused.value = true;
  }
};

// 切换旋转方向（顺时针/逆时针）
const toggleOrbitDirection = () => {
  orbitDirection.value *= -1;
};

const applyFocalLengthPx = (focalPx, options = {}) => {
  if (!camera) return;
  const h = sceneMetadata.value.h || containerRef.value?.clientHeight || window.innerHeight;
  if (!h || !focalPx) return;

  const targetFov = calcFovFromFocal(focalPx, h);
  if (!targetFov || !Number.isFinite(targetFov)) return;

  const duration = options.duration ?? 0;
  if (duration > 0) {
    gsap.to(camera, {
      fov: targetFov,
      duration,
      ease: options.ease || 'power2.out',
      onUpdate: () => {
        camera.updateProjectionMatrix();
        renderOnce();
      },
    });
    return;
  }

  camera.fov = targetFov;
  camera.updateProjectionMatrix();
  renderOnce();
};

const onManualFocalChange = () => {
  const value = clampFocalPx(Number(manualFocalPx.value), focalMin.value, focalMax.value);
  if (!value) return;
  manualFocalPx.value = Number(value.toFixed(1));
  applyFocalLengthPx(value);
};

const resetFocalToCapture = () => {
  const captureFocal = Number(sceneMetadata.value.fl_y || 0);
  if (!captureFocal) return;
  manualFocalPx.value = Number(captureFocal.toFixed(1));
  applyFocalLengthPx(captureFocal, { duration: 0.45, ease: 'power2.inOut' });
};

const toggleFocalSettings = () => {
  showFocalSettings.value = !showFocalSettings.value;
  if (showFocalSettings.value && !manualFocalPx.value) {
    manualFocalPx.value = Number(
      (currentViewFocalPx.value || sceneMetadata.value.fl_y || DEFAULT_FOCAL_PX).toFixed(1),
    );
  }
};

const frameScene = () => {
  if (!camera) return;
  const distance = Math.max(sceneRadius * 2.4, 2.5);
  const pos = sceneCenter.clone().add(new THREE.Vector3(0, sceneRadius * 0.3, distance));

  if (cameraRig) {
    cameraRig.targetPosition.copy(pos);
    cameraRig.targetYaw = 0;
    cameraRig.targetPitch = -Math.atan2(sceneRadius * 0.3, distance);
    cameraRig.position.copy(pos);
    cameraRig.yaw = 0;
    cameraRig.pitch = cameraRig.targetPitch;
    cameraRig.pivot.copy(sceneCenter);
    cameraRig.distance = distance;
    cameraRig.targetDistance = distance;
  } else {
    camera.position.copy(pos);
    camera.lookAt(sceneCenter);
  }

  camera.updateProjectionMatrix();
  refreshCurrentFocalInfo();
  syncHighlight(sceneCenter, Math.max(sceneRadius * 0.16, 0.08));
  syncClipPlane();
};

const flyToImage = (poseData) => {
  if (!camera) return;

  const normalizedMatrix = normalizeMatrixArray(poseData?.matrix);
  if (!normalizedMatrix) {
    console.warn('[SparkViewer] Skip invalid pose matrix:', poseData);
    return;
  }

  const targetMatrix = new THREE.Matrix4().fromArray(normalizedMatrix);
  const targetPosition = new THREE.Vector3();
  const targetQuaternion = new THREE.Quaternion();
  const targetScale = new THREE.Vector3();
  targetMatrix.decompose(targetPosition, targetQuaternion, targetScale);

  if (cameraRig) {
    // 底部镜头代表真实采集相机，跳转后必须按第一人称相机继续交互。
    cameraRig.flyToPose(targetPosition, targetQuaternion);
    notifyFlutter({
      status: 'info',
      msg: `Spark rollfix active: mode=${cameraRig.mode}, roll=${THREE.MathUtils.radToDeg(cameraRig.targetRoll).toFixed(1)}deg`,
    });
  } else {
    // Fallback: direct GSAP tween
    gsap.killTweensOf(camera.position);
    gsap.killTweensOf(camera.quaternion);
    gsap.to(camera.position, {
      x: targetPosition.x,
      y: targetPosition.y,
      z: targetPosition.z,
      duration: 0.9,
      ease: 'power2.inOut',
      onUpdate: renderOnce,
    });
    gsap.to(camera.quaternion, {
      x: targetQuaternion.x,
      y: targetQuaternion.y,
      z: targetQuaternion.z,
      w: targetQuaternion.w,
      duration: 0.9,
      ease: 'power2.inOut',
      onUpdate: renderOnce,
    });
  }

  activeImage.value = poseData.image_url || '';
  activeTag.value = poseData.tag || '';

  const focal = Number(poseData.fl_y || sceneMetadata.value.fl_y || 0);
  if (focal > 0) {
    manualFocalPx.value = Number(focal.toFixed(1));
    applyFocalLengthPx(focal, { duration: 0.65, ease: 'power2.out' });
  }

  syncHighlight(
    deriveHighlightPointFromPose(normalizedMatrix, sceneRadius),
    Math.max(sceneRadius * 0.12, 0.08),
  );

  // ── Focus-area LoD boost: temporarily boost quality around search hit ──
  boostLodAroundPoint(
    deriveHighlightPointFromPose(normalizedMatrix, sceneRadius),
    Math.max(sceneRadius * 0.3, 0.2),
  );

  highlightStatus.value = activeTag.value ? `高亮镜头: ${activeTag.value}` : '高亮当前视角区域';
};

/**
 * Boost LoD quality around a specific point for a short duration.
 * Uses Spark 2.0 lodSplatScale to increase quality in the focus area.
 */
const boostLodAroundPoint = (point, radius) => {
  if (spark && 'lodSplatScale' in spark) {
    // Temporarily boost quality
    spark.lodSplatScale = 1.3;

    // Gradually return to normal after flight settles
    setTimeout(() => {
      if (spark && 'lodSplatScale' in spark) {
        spark.lodSplatScale = 1.0;
      }
    }, 2000);
  }
};

const searchAndFly = () => {
  if (filteredPoses.value.length > 0) {
    flyToImage(filteredPoses.value[0]);
  } else {
    alert('场景中没有找到符合该描述的视角哦~');
  }
};

const maybeApplyInitialTarget = (forceFallback = false) => {
  if (!pendingInitialTarget || didApplyInitialTarget) return;

  const resolvedPose = findPoseByInitialTarget(pendingInitialTarget, cameraPoses.value);
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

const loadPoses = async () => {
  posesFetchSettled = false;
  cameraPoses.value = [];

  try {
    const response = await fetch(currentPosesPath.value);
    const data = await response.json();

    posesFetchSettled = true;

    if (data.frames) {
      sceneMetadata.value = {
        w: data.w,
        h: data.h,
        fl_x: data.fl_x,
        fl_y: data.fl_y,
      };

      cameraPoses.value = data.frames.map((frame) => ({
        id: frame.id,
        matrix: frame.matrix,
        image_url: resolveImageUrl(frame.image_url, currentPosesPath.value),
        tag: frame.tag,
        fl_x: frame.fl_x || data.fl_x,
        fl_y: frame.fl_y || data.fl_y,
        w: frame.w || data.w,
        h: frame.h || data.h,
      }));
    } else {
      cameraPoses.value = Array.isArray(data) ? data : [];
    }

    const focal = Number(sceneMetadata.value.fl_y || 0);
    if (focal > 0) {
      manualFocalPx.value = Number(focal.toFixed(1));
      applyFocalLengthPx(focal);
    } else {
      manualFocalPx.value = DEFAULT_FOCAL_PX;
      applyFocalLengthPx(DEFAULT_FOCAL_PX);
    }

    // ── Scene topology classification ──
    if (cameraPoses.value.length >= 4) {
      const profile = classifyInteractionProfile(cameraPoses.value, sceneRadius);
      interactionProfile.value = profile.profile;
      profileConfidence.value = profile.confidence;

      if (profile.focusPoint) {
        focusPointArr.value = profile.focusPoint;
        if (cameraRig) {
          cameraRig.pivot.fromArray(profile.focusPoint);
          if (profile.defaultRadius) {
            cameraRig.distance = profile.defaultRadius;
            cameraRig.targetDistance = profile.defaultRadius;
          }
          cameraRig.sceneRadius = sceneRadius;
          cameraRig.initFromCamera();

          if (profile.profile === 'object_orbit') {
            cameraRig.mode = 'inspect';
          }
        }
      }

      // Build pose graph for guided navigation
      poseGraph = new PoseGraph(sceneRadius);
      poseGraph.buildFromPoses(cameraPoses.value);

      notifyFlutter({
        status: 'info',
        msg: `Scene profile: ${profile.profile} (conf: ${(profile.confidence * 100).toFixed(0)}%)`,
      });
    }

    maybeApplyInitialTarget(true);
  } catch (error) {
    posesFetchSettled = true;
    console.error('[SparkViewer] 加载位姿失败:', error);
    manualFocalPx.value = DEFAULT_FOCAL_PX;
    applyFocalLengthPx(DEFAULT_FOCAL_PX);
    maybeApplyInitialTarget(true);
  }
};

const disposeViewer = () => {
  if (renderer) {
    renderer.setAnimationLoop(null);
  }

  if (gestureHandler) {
    gestureHandler.dispose();
    gestureHandler = null;
  }

  if (cameraRig) {
    cameraRig.dispose();
    cameraRig = null;
  }

  stopMemoryPath();
  poseGraph = null;

  if (splatMesh) {
    splatMesh.removeFromParent();
    splatMesh.dispose();
    splatMesh = null;
  }

  highlightEffect = null;
  clipEffect = null;

  if (spark) {
    spark.removeFromParent();
    spark = null;
  }

  if (renderer) {
    renderer.dispose();
    if (renderer.domElement?.parentNode) {
      renderer.domElement.parentNode.removeChild(renderer.domElement);
    }
    renderer = null;
  }

  scene = null;
  camera = null;
};

const setupResizeHandler = () => {
  if (resizeHandler) {
    window.removeEventListener('resize', resizeHandler);
  }

  resizeHandler = () => {
    if (!containerRef.value || !camera || !renderer) return;
    const width = containerRef.value.clientWidth || window.innerWidth;
    const height = containerRef.value.clientHeight || window.innerHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    renderOnce();
  };

  window.addEventListener('resize', resizeHandler);
};

/**
 * Resolve the best model URL using Spark 2.0 format priority:
 * .rad (LoD streaming) > .spz (compressed) > .ksplat > .ply (original)
 */
const resolveBestModelUrl = async (originalUrl) => {
  const stripExt = (url) => url.replace(/\.(ply|splat|ksplat|spz|rad|sog)$/i, '');
  const candidates = ['.rad', '.spz', '.ksplat'];

  for (const ext of candidates) {
    const candidateUrl = stripExt(originalUrl) + ext;
    try {
      const resp = await fetch(candidateUrl, { method: 'HEAD' });
      if (resp.ok) {
        console.log(`[SparkViewer] Using optimized format: ${ext}`);
        return candidateUrl;
      }
    } catch {
      // File doesn't exist or not reachable — skip
    }
  }

  return originalUrl;
};

const initViewer = async (plyUrl, posesUrl, initialTarget) => {
  if (isLoading.value) return;
  isLoading.value = true;
  loadError.value = '';
  pendingInitialTarget = null;
  didApplyInitialTarget = false;

  if (plyUrl) currentModelUrl.value = plyUrl;
  if (posesUrl) currentPosesPath.value = posesUrl;

  try {
    disposeViewer();

    scene = new THREE.Scene();

    const width = containerRef.value?.clientWidth || window.innerWidth;
    const height = containerRef.value?.clientHeight || window.innerHeight;

    camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 2000);
    scene.add(camera);

    renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: true,
      powerPreference: 'high-performance',
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.setSize(width, height, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    containerRef.value.innerHTML = '';
    containerRef.value.appendChild(renderer.domElement);

    // ===== Orbit 指针事件：检测手动拖拽以临时中断/恢复 orbit 旋转 =====
    // 按下时标记拖拽状态，暂停 orbit
    renderer.domElement.addEventListener('pointerdown', () => {
      isOrbitDragging = true;
    });
    // 松手后从当前相机位置重新同步轨道参数，实现无缝恢复
    renderer.domElement.addEventListener('pointerup', () => {
      isOrbitDragging = false;
      if (orbitEnabled.value && !orbitPaused.value) {
        syncOrbitFromCamera();
      }
    });
    // pointerleave 作为安全兜底，防止拖拽状态卡住
    renderer.domElement.addEventListener('pointerleave', () => {
      if (isOrbitDragging) {
        isOrbitDragging = false;
        if (orbitEnabled.value && !orbitPaused.value) {
          syncOrbitFromCamera();
        }
      }
    });

    spark = new SparkRenderer({
      renderer,
      maxStdDev: Math.sqrt(7),
      preUpdate: false,
      view: {
        sortRadial: true,
      },
    });
    scene.add(spark);

    // ── BrainDance camera rig (replaces SparkControls) ──
    cameraRig = new BrainDanceCameraRig({
      camera,
      sceneRadius: DEFAULT_SCENE_RADIUS,
    });

    gestureHandler = new GestureHandler(renderer.domElement, {
      onAction: handleGesture,
    });

    splatMesh = new SplatMesh({
      url: await resolveBestModelUrl(currentModelUrl.value),
      editable: true,
    });
    scene.add(splatMesh);

    clock.start();

    renderer.setAnimationLoop(() => {
      if (!renderer || !scene || !camera) return;

      const dt = clock.getDelta();
      if (cameraRig) {
        cameraRig.update(dt);

        // Adaptive quality: adjust LoD scale and DPR based on FPS
        if (spark && 'lodSplatScale' in spark) {
          spark.lodSplatScale = cameraRig.getRecommendedLodScale(currentFps.value);
        }
        const baseDpr = window.devicePixelRatio || 1;
        renderer.setPixelRatio(cameraRig.getRecommendedDpr(baseDpr));
      }
      const now = performance.now();
      const orbitDt = Math.min((now - orbitLastFrameTime) / 1000, 0.1);
      orbitLastFrameTime = now;

      // FPS 统计
      fpsFrames += 1;
      if (now - fpsTimestamp >= 1000) {
        currentFps.value = fpsFrames;
        fpsFrames = 0;
        fpsTimestamp = now;
      }

      // ===== Orbit 模式：自动旋转相机（非暂停、非手动拖拽时生效） =====
      if (orbitEnabled.value && !orbitPaused.value && !isOrbitDragging && orbitDt > 0) {
        // 根据速度和方向更新旋转角度
        const speedRad = orbitSpeed.value * (Math.PI / 180);
        orbitAngle += speedRad * orbitDt * orbitDirection.value;
        // 计算圆周位置并应用到相机
        const r = orbitRadius.value > 0 ? orbitRadius.value : autoOrbitRadius;
        const x = sceneCenter.x + r * Math.cos(orbitAngle);
        const z = sceneCenter.z + r * Math.sin(orbitAngle);
        camera.position.set(x, orbitY, z);
        camera.lookAt(sceneCenter);
        camera.updateProjectionMatrix();
      }

      // 始终更新 controls（处理滚轮缩放、手动拖拽等输入）
      controls?.update(camera);

      // ===== Orbit 模式下重新锁定相机位置，防止 controls 覆盖 orbit 位置 =====
      if (orbitEnabled.value && !orbitPaused.value && !isOrbitDragging) {
        const r = orbitRadius.value > 0 ? orbitRadius.value : autoOrbitRadius;
        const x = sceneCenter.x + r * Math.cos(orbitAngle);
        const z = sceneCenter.z + r * Math.sin(orbitAngle);
        camera.position.set(x, orbitY, z);
        camera.lookAt(sceneCenter);
      }

      renderer.render(scene, camera);
    });

    setupResizeHandler();
    frameScene();
    manualFocalPx.value = DEFAULT_FOCAL_PX;

    await splatMesh.initialized;

    const bbox = splatMesh.getBoundingBox(true);
    const size = bbox.getSize(new THREE.Vector3());
    sceneCenter = bbox.getCenter(new THREE.Vector3());
    sceneRadius = Math.max(size.length() * 0.32, DEFAULT_SCENE_RADIUS);

    // 如果 orbit 已开启，重新同步轨道参数到新计算的模型中心
    if (orbitEnabled.value) {
      syncOrbitFromCamera();
    }

    frameScene();
    highlightEffect = createSphereHighlightEffect(sceneRadius, highlightEnabled.value);
    splatMesh.add(highlightEffect.edit);
    clipEffect = createClipPlaneEffect(sceneRadius, clipEnabled.value);
    splatMesh.add(clipEffect.edit);
    highlightStatus.value = '局部高亮已挂载';
    notifyFlutter({ status: 'success', msg: 'Spark 模型加载完成' });

    if (initialTarget && (initialTarget.matrix || initialTarget.imageId)) {
      pendingInitialTarget = {
        matrix: initialTarget.matrix || null,
        imageId: initialTarget.imageId || null,
      };
    }

    await loadPoses();
    maybeApplyInitialTarget(true);
    highlightStatus.value = 'Spark 渲染器已接管场景';
  } catch (error) {
    console.error('[SparkViewer] init error:', error);
    loadError.value = (error && (error.message || String(error))) || 'Spark 模型加载失败';
  } finally {
    isLoading.value = false;
  }
};

const toggleHighlight = () => {
  highlightEnabled.value = !highlightEnabled.value;
  syncHighlight(highlightEffect?.sdf?.position || sceneCenter, highlightEffect?.sdf?.radius || null);
  highlightStatus.value = highlightEnabled.value ? '局部高亮已开启' : '局部高亮已关闭';
  renderOnce();
};

const toggleClip = () => {
  clipEnabled.value = !clipEnabled.value;
  syncClipPlane();
  highlightStatus.value = clipEnabled.value ? '剖切预览已开启' : '剖切预览已关闭';
  renderOnce();
};

const onClipOffsetChange = (value) => {
  clipOffset.value = value;
  syncClipPlane();
  if (clipEnabled.value) {
    highlightStatus.value = `剖切位置: ${(value * sceneRadius).toFixed(2)}`;
  }
  renderOnce();
};

// ── Gesture → camera rig bridge ──

let gestureDebugCount = 0;

const handleGesture = (action) => {
  if (!cameraRig) return;

  if (gestureDebugCount < 3 && (action.type === 'look' || action.type === 'pinch' || action.type === 'pan')) {
    gestureDebugCount += 1;
    const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
    notifyFlutter({
      status: 'info',
      msg: `Spark gesture ${action.type}: mode=${cameraRig.mode}, roll=${THREE.MathUtils.radToDeg(euler.z).toFixed(1)}deg`,
    });
  }

  switch (action.type) {
    case 'look':
      cameraRig.onLookDrag(action.dx, action.dy);
      break;
    case 'pinch':
      cameraRig.onPinch(action.scaleDelta);
      break;
    case 'pan':
      cameraRig.onPan(action.dx, action.dy);
      break;
    case 'doubletap':
      frameScene();
      break;
    case 'longpress':
      // Could enter inspect mode at long-press point
      break;
    case 'swipe_forward':
      navigatePoseGraph('forward');
      break;
    case 'swipe_backward':
      navigatePoseGraph('backward');
      break;
  }
};

const navigatePoseGraph = (direction) => {
  if (!poseGraph || poseGraph.size === 0) return;

  const currentIdx = currentPoseIndex.value >= 0
    ? currentPoseIndex.value
    : poseGraph.findNearestNode(cameraRig.position);

  const nextIdx = poseGraph.getNextAlongPath(currentIdx, direction);
  if (nextIdx == null) return;

  const node = poseGraph.getNode(nextIdx);
  if (!node) return;

  currentPoseIndex.value = nextIdx;

  const matrix = new THREE.Matrix4().fromArray(node.matrix);
  const pos = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const scl = new THREE.Vector3();
  matrix.decompose(pos, quat, scl);

  cameraRig.flyToPose(pos, quat);

  activeImage.value = node.imageUrl;
  activeTag.value = node.tag;

  syncHighlight(
    deriveHighlightPointFromPose(node.matrix, sceneRadius),
    Math.max(sceneRadius * 0.12, 0.08),
  );
  highlightStatus.value = node.tag ? `高亮镜头: ${node.tag}` : '高亮当前视角区域';
};

// ── Quality mode switching ──

const setQualityMode = (mode) => {
  qualityMode.value = mode;
  if (!cameraRig) return;

  switch (mode) {
    case 'smooth':
      cameraRig.lookDamping = 22;
      cameraRig.moveDamping = 18;
      break;
    case 'standard':
      cameraRig.lookDamping = 18;
      cameraRig.moveDamping = 14;
      break;
    case 'hd':
      cameraRig.lookDamping = 14;
      cameraRig.moveDamping = 10;
      break;
  }
};

// ── Interaction profile switching ──

const setInteractionMode = (mode) => {
  if (!cameraRig) return;
  cameraRig.setMode(mode);
  highlightStatus.value = `交互模式: ${mode === 'recall' ? '回忆' : mode === 'inspect' ? '观察' : '自由'}`;
};

// ── Memory path auto-camera (recall path cinematography) ──

let memoryPathTimer = null;
let isPlayingMemoryPath = false;
const memoryPathSpeed = ref(1.0); // 0.5x – 3x
let memoryPathPoses = [];
let memoryPathIndex = 0;

const startMemoryPath = (poses) => {
  if (!poses || poses.length < 2 || !cameraRig) return;

  stopMemoryPath();
  memoryPathPoses = poses;
  memoryPathIndex = 0;
  isPlayingMemoryPath = true;
  highlightStatus.value = '回忆路径播放中...';

  flyToNextMemoryPose();
};

const flyToNextMemoryPose = () => {
  if (!isPlayingMemoryPath || memoryPathIndex >= memoryPathPoses.length) {
    stopMemoryPath();
    return;
  }

  const poseData = memoryPathPoses[memoryPathIndex];
  const normalizedMatrix = normalizeMatrixArray(poseData.matrix);
  if (!normalizedMatrix) {
    memoryPathIndex += 1;
    flyToNextMemoryPose();
    return;
  }

  const m = new THREE.Matrix4().fromArray(normalizedMatrix);
  const pos = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const scl = new THREE.Vector3();
  m.decompose(pos, quat, scl);

  cameraRig.flyToPose(pos, quat, () => {
    activeImage.value = poseData.image_url || '';
    activeTag.value = poseData.tag || '';
    syncHighlight(
      deriveHighlightPointFromPose(normalizedMatrix, sceneRadius),
      Math.max(sceneRadius * 0.12, 0.08),
    );

    memoryPathIndex += 1;
    if (memoryPathIndex < memoryPathPoses.length) {
      // Delay between poses inversely proportional to speed
      const delay = Math.max(400, 1800 / memoryPathSpeed.value);
      memoryPathTimer = setTimeout(flyToNextMemoryPose, delay);
    } else {
      stopMemoryPath();
    }
  });
};

const pauseMemoryPath = () => {
  if (memoryPathTimer !== null) {
    clearTimeout(memoryPathTimer);
    memoryPathTimer = null;
  }
  isPlayingMemoryPath = false;
  highlightStatus.value = '回忆路径已暂停';
};

const resumeMemoryPath = () => {
  if (memoryPathIndex >= memoryPathPoses.length) return;
  isPlayingMemoryPath = true;
  highlightStatus.value = '回忆路径播放中...';
  flyToNextMemoryPose();
};

const stopMemoryPath = () => {
  if (memoryPathTimer !== null) {
    clearTimeout(memoryPathTimer);
    memoryPathTimer = null;
  }
  isPlayingMemoryPath = false;
  highlightStatus.value = '回忆路径已停止';
};

onMounted(() => {
  notifyFlutter({ status: 'ready' });

  window.loadModelFromFlutter = (input) => {
    console.log('[Flutter->SparkViewer] 收到加载请求:', input);
    if (typeof input === 'string') {
      initViewer(input, null, null);
      return;
    }

    if (typeof input === 'object' && input !== null) {
      initViewer(input.ply || null, input.poses || null, {
        matrix: input.matrix || null,
        imageId: input.imageId || null,
      });
      return;
    }

    initViewer(null, null, null);
  };

  // 注册供 Flutter 调用的 TimePeeling 模型列表设置函数
  window.setModelListForTimePeeling = (list, currentId) => {
    console.log('[Flutter->SparkViewer] 收到 TimePeeling 模型列表:', list, '当前模型:', currentId);
    // Spark 2.0 当前版本暂不支持 TimePeeling 切换，但需要提供空实现避免 Flutter 端报错
    // 后续可扩展为多模型切换逻辑
  };

  const initialInput = parseInitialInputFromUrl();
  if (window.BrainDanceChannel) {
    return;
  }

  if (initialInput && !hasInitializedFromExternalInput) {
    hasInitializedFromExternalInput = true;
    initViewer(initialInput.ply, initialInput.poses, {
      matrix: initialInput.matrix || null,
      imageId: initialInput.imageId || null,
    });
    return;
  }

  initViewer(null, null, null);
});

onBeforeUnmount(() => {
  if (resizeHandler) {
    window.removeEventListener('resize', resizeHandler);
    resizeHandler = null;
  }

  if (window.loadModelFromFlutter) {
    delete window.loadModelFromFlutter;
  }

  if (window.setModelListForTimePeeling) {
    delete window.setModelListForTimePeeling;
  }

  disposeViewer();
});
</script>

<template>
  <div class="app-shell">
    <div ref="containerRef" class="viewer-layer"></div>
    <div class="ambient-mask"></div>

    <TopHud
      :current-fps="currentFps"
      :highlight-enabled="highlightEnabled"
      :search-query="searchQuery"
      :show-focal-settings="showFocalSettings"
      @update:search-query="searchQuery = $event"
      @search="searchAndFly"
      @toggle-focal="toggleFocalSettings"
      @toggle-highlight="toggleHighlight"
    />

    <div v-if="isLoading" class="overlay">
      <div class="status-card">
        <div class="status-dot"></div>
        <div class="status-title">Spark 渲染器正在接管场景</div>
        <div class="status-copy">模型、位姿和局部特效模块正在初始化。</div>
      </div>
    </div>

    <div v-if="loadError" class="overlay">
      <div class="status-card status-card--error">
        <div class="eyebrow">Load Failed</div>
        <div class="status-title">Spark 备选查看器未能正常打开</div>
        <div class="status-copy">{{ loadError }}</div>
        <button class="panel-btn panel-btn--solid" @click="initViewer(currentModelUrl, currentPosesPath, null)">
          重新载入
        </button>
      </div>
    </div>

    <FocalPanel
      v-if="showFocalSettings"
      :focal-max="focalMax"
      :focal-min="focalMin"
      :manual-focal-px="manualFocalPx"
      :current-view-fov="currentViewFov"
      :current-view-focal-px="currentViewFocalPx"
      @update:manual-focal-px="manualFocalPx = $event"
      @input-focal="onManualFocalChange"
      @change-focal="onManualFocalChange"
      @reset-focal="resetFocalToCapture"
    />

    <StatusRibbon
      :clip-enabled="clipEnabled"
      :clip-offset="clipOffset"
      :current-model-url="currentModelUrl"
      :current-poses-path="currentPosesPath"
      :highlight-status="highlightStatus"
      @toggle-clip="toggleClip"
      @update:clip-offset="onClipOffsetChange"
    />

    <CameraTrack
      :active-image="activeImage"
      :filtered-poses="filteredPoses"
      :search-query="searchQuery"
      @select-pose="flyToImage"
    />

    <!-- Quality HUD -->
    <div class="quality-hud">
      <div class="quality-row">
        <button
          v-for="q in [
            { key: 'smooth', label: '流畅' },
            { key: 'standard', label: '标准' },
            { key: 'hd', label: '高清' },
          ]"
          :key="q.key"
          class="quality-btn"
          :class="{ 'quality-btn--active': qualityMode === q.key }"
          @click="setQualityMode(q.key)"
        >
          {{ q.label }}
        </button>
      </div>
      <div class="quality-row" style="margin-top: 4px">
        <button
          v-for="m in [
            { key: 'recall', label: '回忆' },
            { key: 'inspect', label: '观察' },
            { key: 'freeWalk', label: '自由' },
          ]"
          :key="m.key"
          class="quality-btn quality-btn--mode"
          :class="{ 'quality-btn--active': cameraRig?.mode === m.key }"
          @click="setInteractionMode(m.key)"
        >
          {{ m.label }}
        </button>
      </div>
      <div class="quality-row" style="margin-top: 4px">
        <button
          class="quality-btn quality-btn--path"
          @click="startMemoryPath(filteredPoses)"
        >
          ▶ 路径
        </button>
        <button
          class="quality-btn quality-btn--path"
          @click="isPlayingMemoryPath ? pauseMemoryPath() : resumeMemoryPath()"
        >
          {{ isPlayingMemoryPath ? '⏸ 暂停' : '▶ 继续' }}
        </button>
        <button
          class="quality-btn quality-btn--path"
          @click="stopMemoryPath()"
        >
          ⏹ 停止
        </button>
      </div>
    </div>

    <ReferenceCard
      :active-image="activeImage"
      :active-tag="activeTag"
      :scene-metadata="sceneMetadata"
      @close="activeImage = ''; activeTag = ''"
    />

    <!-- ===== Orbit 控制面板：相机绕模型中心自动旋转 ===== -->
    <div
      v-if="!isLoading && !loadError"
      class="orbit-panel panel-card"
      @mousedown.stop
      @touchstart.stop
      @touchmove.stop
      @touchend.stop
      @touchcancel.stop
    >
      <div class="eyebrow">Orbit Control</div>
      <div class="panel-title">轨道旋转</div>
      <div class="orbit-btn-row">
        <button class="panel-btn panel-btn--solid" @click="orbitEnabled ? stopOrbit() : startOrbit()">
          {{ orbitEnabled ? '关闭轨道' : '开启轨道' }}
        </button>
        <button v-if="orbitEnabled" class="panel-btn panel-btn--ghost" @click="toggleOrbitPause()">
          {{ orbitPaused ? '恢复' : '暂停' }}
        </button>
      </div>
      <template v-if="orbitEnabled">
        <!-- 旋转速度控制 -->
        <div class="focal-row" style="margin-top: 10px;">
          <span>速度</span>
          <span>{{ orbitSpeed }}度/秒</span>
        </div>
        <input
          type="range"
          :min="1"
          :max="120"
          :value="orbitSpeed"
          step="1"
          @input="orbitSpeed = Number($event.target.value)"
        />
        <!-- 旋转半径控制 -->
        <div class="focal-row" style="margin-top: 6px;">
          <span>半径</span>
          <span>{{ orbitRadius > 0 ? orbitRadius.toFixed(2) : '自动' }}</span>
        </div>
        <input
          type="range"
          :min="0"
          :max="20"
          :value="orbitRadius"
          step="0.1"
          @input="orbitRadius = Number($event.target.value)"
        />
        <!-- 旋转方向切换 -->
        <div style="margin-top: 8px;">
          <button class="panel-btn panel-btn--ghost orbit-dir-btn" @click="toggleOrbitDirection()">
            {{ orbitDirection === 1 ? '逆时针' : '顺时针' }}
          </button>
        </div>
      </template>
    </div>
  </div>
</template>

<style>
.app-shell {
  position: relative;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  color: #1e1e20;
  background:
    radial-gradient(circle at top left, rgba(237, 225, 198, 0.2), transparent 28%),
    radial-gradient(circle at top right, rgba(121, 138, 142, 0.18), transparent 30%),
    linear-gradient(180deg, #f6f1e8 0%, #ddd8cf 100%);
  font-family: 'HarmonyOS Sans SC', 'Microsoft YaHei', 'PingFang SC', sans-serif;
}

.viewer-layer {
  position: absolute;
  inset: 0;
  touch-action: none;
  user-select: none;
  -webkit-user-select: none;
  overscroll-behavior: none;
}

.viewer-layer canvas {
  touch-action: none;
  user-select: none;
  -webkit-user-select: none;
}

.ambient-mask {
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(180deg, rgba(22, 25, 28, 0.1), transparent 18%, transparent 82%, rgba(22, 25, 28, 0.16)),
    radial-gradient(circle at center, transparent 58%, rgba(22, 25, 28, 0.12) 100%);
}

.hud {
  position: absolute;
  top: 18px;
  left: 18px;
  right: 18px;
  z-index: 50;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.toolbar {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 10px;
  flex-wrap: wrap;
}

.panel-card {
  background: rgba(250, 248, 243, 0.86);
  border: 1px solid rgba(97, 109, 118, 0.16);
  border-radius: 22px;
  box-shadow: 0 14px 28px rgba(0, 0, 0, 0.08);
  backdrop-filter: blur(16px);
}

.search-panel {
  display: flex;
  align-items: center;
  gap: 8px;
  width: min(580px, 100%);
  padding: 8px;
}

.search-input {
  flex: 1 1 auto;
  min-width: 0;
  padding: 11px 12px;
  border-radius: 12px;
  border: 1px solid rgba(97, 109, 118, 0.16);
  background: rgba(255, 255, 255, 0.7);
  outline: none;
  color: #1e1e20;
  font-size: 13px;
}

.search-input:focus {
  border-color: rgba(97, 109, 118, 0.42);
  box-shadow: 0 0 0 4px rgba(97, 109, 118, 0.08);
}

.panel-btn {
  appearance: none;
  border-radius: 14px;
  border: 1px solid rgba(97, 109, 118, 0.18);
  padding: 10px 14px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 700;
  transition: transform 180ms ease-out, box-shadow 180ms ease-out, background-color 180ms ease-out;
}

.panel-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 18px rgba(0, 0, 0, 0.07);
}

.panel-btn--solid {
  background: #c86b3c;
  border-color: #c86b3c;
  color: #fbf8f3;
}

.panel-btn--solid:hover {
  background: #b85d31;
  border-color: #b85d31;
}

.panel-btn--ghost {
  background: rgba(250, 248, 243, 0.86);
  color: #1e1e20;
}

.fps-chip {
  border-radius: 12px;
  padding: 8px 10px;
  background: rgba(250, 248, 243, 0.86);
  border: 1px solid rgba(97, 109, 118, 0.16);
  font-family: monospace;
  font-size: 12px;
}

.overlay {
  position: absolute;
  inset: 0;
  z-index: 100;
  display: flex;
  justify-content: center;
  align-items: center;
  background: rgba(30, 30, 32, 0.2);
  backdrop-filter: blur(8px);
}

.status-card {
  min-width: min(84vw, 340px);
  padding: 24px 20px;
  border-radius: 24px;
  background: rgba(250, 248, 243, 0.93);
  border: 1px solid rgba(97, 109, 118, 0.18);
  box-shadow: 0 20px 34px rgba(0, 0, 0, 0.1);
  text-align: center;
}

.status-card--error .status-title {
  color: #8d453e;
}

.status-dot {
  width: 12px;
  height: 12px;
  margin: 0 auto 14px;
  border-radius: 999px;
  background: #c86b3c;
  box-shadow: 0 0 0 10px rgba(200, 107, 60, 0.12);
  animation: pulse 1.8s ease-in-out infinite;
}

.status-title {
  font-size: 20px;
  font-weight: 700;
}

.status-copy {
  margin-top: 8px;
  line-height: 1.6;
  font-size: 13px;
  color: rgba(30, 30, 32, 0.68);
  word-break: break-word;
}

.status-ribbon {
  position: absolute;
  top: 86px;
  right: 18px;
  z-index: 60;
  width: min(26vw, 320px);
  min-width: 220px;
  padding: 14px 16px;
}

.status-line {
  margin-top: 4px;
  font-size: 14px;
  font-weight: 700;
}

.status-subline {
  margin-top: 6px;
  font-size: 11px;
  line-height: 1.5;
  color: rgba(30, 30, 32, 0.62);
  word-break: break-all;
}

.clip-controls {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.clip-toggle {
  width: 100%;
}

.clip-slider {
  width: 100%;
  accent-color: #c86b3c;
}

.focal-panel {
  position: absolute;
  top: 128px;
  right: 18px;
  z-index: 70;
  width: 240px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.panel-title {
  font-size: 15px;
  font-weight: 700;
  color: #1e1e20;
}

.eyebrow {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #66757f;
}

.focal-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 12px;
  color: rgba(30, 30, 32, 0.72);
}

.focal-number {
  width: 100px;
  border-radius: 10px;
  border: 1px solid rgba(97, 109, 118, 0.16);
  padding: 8px 10px;
  background: rgba(255, 255, 255, 0.86);
}

.camera-track-dock {
  position: absolute;
  left: 18px;
  bottom: 18px;
  z-index: 55;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
}

.camera-track-toggle {
  padding: 9px 12px;
  border-radius: 999px;
  box-shadow: 0 10px 18px rgba(0, 0, 0, 0.07);
}

.camera-track {
  position: absolute;
  left: 0;
  bottom: 48px;
  display: flex;
  gap: 12px;
  align-items: flex-start;
  width: min(540px, calc(100vw - 36px));
  overflow-x: auto;
  padding: 12px 14px;
}

.track-copy {
  min-width: 110px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.track-text {
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.5;
  color: rgba(30, 30, 32, 0.68);
}

.camera-item {
  position: relative;
  width: 84px;
  height: 60px;
  flex-shrink: 0;
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(97, 109, 118, 0.12);
  background: rgba(255, 255, 255, 0.74);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 200ms ease, box-shadow 200ms ease;
  color: #333;
}

.camera-item:hover,
.camera-item.active {
  transform: translateY(-2px);
  box-shadow: 0 10px 18px rgba(97, 109, 118, 0.12);
}

.camera-thumb {
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0.9;
}

.camera-tag {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 8px 7px 6px;
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  background: linear-gradient(180deg, transparent, rgba(30, 30, 32, 0.74));
}

.reference-card {
  position: absolute;
  top: 260px;
  right: 18px;
  z-index: 60;
  width: min(22vw, 152px);
  min-width: 118px;
  padding: 8px;
  cursor: pointer;
}

.reference-image {
  width: 100%;
  border-radius: 10px;
  border: 1px solid rgba(97, 109, 118, 0.12);
  margin: 6px 0;
}

.reference-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 4px;
}

.meta-chip {
  padding: 3px 6px;
  border-radius: 999px;
  background: rgba(228, 232, 237, 0.78);
  font-size: 9px;
  color: rgba(30, 30, 32, 0.75);
}

.meta-chip--accent {
  color: #c86b3c;
}

.reference-hint {
  font-size: 9px;
  color: rgba(30, 30, 32, 0.48);
}

input[type='range'] {
  accent-color: #c86b3c;
}

button {
  font-family: inherit;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.16);
    opacity: 0.72;
  }
}

@media (max-width: 768px) {
  .hud {
    top: 12px;
    left: 12px;
    right: 12px;
    gap: 8px;
  }

  .toolbar {
    justify-content: space-between;
    gap: 8px;
  }

  .search-panel {
    width: 100%;
    padding: 6px;
  }

  .search-input {
    font-size: 12px;
    padding: 9px 10px;
  }

  .panel-btn {
    padding: 9px 10px;
    font-size: 12px;
  }

  .status-ribbon {
    top: 112px;
    right: 12px;
    width: min(68vw, 280px);
  }

  .focal-panel {
    top: 212px;
    right: 12px;
  }

  .reference-card {
    top: 412px;
    right: 12px;
    width: 112px;
    min-width: 112px;
  }

  .camera-track-dock {
    left: 12px;
    bottom: 12px;
  }

  .camera-track {
    width: min(360px, calc(100vw - 24px));
    bottom: 44px;
    padding: 12px;
  }

  .track-copy {
    min-width: 96px;
  }

  .camera-item {
    width: 78px;
    height: 56px;
  }

  .orbit-panel {
    top: 510px;
    right: 12px;
    width: 180px;
    padding: 10px;
  }
}

/* ===== Orbit 控制面板样式 ===== */
.orbit-panel {
  position: absolute;
  top: 370px;
  right: 18px;
  z-index: 60;
  width: 210px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.orbit-btn-row {
  display: flex;
  gap: 8px;
  margin-top: 8px;
}

.orbit-dir-btn {
  width: 100%;
}

.quality-hud {
  position: absolute;
  bottom: 18px;
  right: 18px;
  z-index: 65;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 6px;
  border-radius: 16px;
  background: rgba(250, 248, 243, 0.86);
  border: 1px solid rgba(97, 109, 118, 0.16);
  box-shadow: 0 10px 18px rgba(0, 0, 0, 0.07);
  backdrop-filter: blur(16px);
}

.quality-row {
  display: flex;
  gap: 3px;
}

.quality-btn {
  appearance: none;
  border: 1px solid rgba(97, 109, 118, 0.16);
  border-radius: 10px;
  padding: 6px 10px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  background: transparent;
  color: rgba(30, 30, 32, 0.62);
  transition: background-color 160ms, color 160ms;
  font-family: inherit;
}

.quality-btn:hover {
  background: rgba(200, 107, 60, 0.08);
}

.quality-btn--active {
  background: rgba(200, 107, 60, 0.14);
  color: #c86b3c;
  border-color: rgba(200, 107, 60, 0.28);
}

.quality-btn--mode {
  font-size: 10px;
  padding: 5px 8px;
}

.quality-btn--path {
  font-size: 10px;
  padding: 5px 7px;
}
</style>


