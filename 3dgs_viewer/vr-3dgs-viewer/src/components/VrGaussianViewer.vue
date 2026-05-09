<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import * as THREE from 'three'
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d'
import { deriveVrConfigUrl, getInitialPayload, normalizePayload } from '../engine/payload'
import { getPreviewMode, switchPreviewMode, type PreviewMode } from '../engine/previewMode'
import { loadVrConfig } from '../engine/vrConfig'
import { getVrModelCandidates } from '../engine/modelUrl'
import type { BrainDanceViewerPayload, BrainDanceVrConfig, RuntimeGaussianViewer } from '../types/viewer'

const containerRef = ref<HTMLDivElement | null>(null)
const status = ref('等待初始化')
const errorMessage = ref('')
const fps = ref(0)
const isVrPresenting = ref(false)
const previewMode = ref<PreviewMode>(getPreviewMode())
const activePayload = ref<BrainDanceViewerPayload | null>(null)
const activeModelUrl = ref('')
const activeConfig = ref<BrainDanceVrConfig | null>(null)
const cameraPosition = ref('0, 0, 0')
const stereoIpd = ref(0.064)

let viewer: GaussianSplats3D.Viewer | null = null
let frameCount = 0
let lastFpsTime = performance.now()
let scaleStep = 0.1
let statsRafId = 0
let stereoRafId = 0
let scaleOverride: number | null = null
let rotationOverride: number | null = null

const sceneLabel = computed(() => activePayload.value?.sceneId || activePayload.value?.imageId || 'BrainDance VR Preview')
const scaleLabel = computed(() => activeConfig.value?.worldScale.toFixed(2) ?? '1.00')
const positionLabel = computed(() => activeConfig.value?.worldPosition.map((item) => item.toFixed(2)).join(', ') ?? '0, 0, 0')
const rotationLabel = computed(() => activeConfig.value?.worldRotationY.toFixed(2) ?? '0.00')

function makeSceneRotationY(rotationY: number): [number, number, number, number] {
  const quaternion = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), rotationY)
  return [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
}

function disposeViewer() {
  if (!viewer) return

  try {
    viewer.stop()
    viewer.dispose()
  } catch (error) {
    console.warn('[BrainDance VR] viewer 释放时出现非致命错误:', error)
  }
  viewer = null
}

function updateFps() {
  frameCount += 1
  const now = performance.now()
  if (now - lastFpsTime < 1000) return

  fps.value = frameCount
  frameCount = 0
  lastFpsTime = now
}

function startFpsLoop() {
  updateFps()
  updateCameraDebug()
  statsRafId = window.requestAnimationFrame(startFpsLoop)
}

function stopStereoLoop() {
  if (!stereoRafId) return
  window.cancelAnimationFrame(stereoRafId)
  stereoRafId = 0
}

function installXrSessionListeners() {
  const xr = viewer?.renderer?.xr
  if (!xr?.addEventListener) return

  xr.addEventListener('sessionstart', () => {
    isVrPresenting.value = true
  })
  xr.addEventListener('sessionend', () => {
    isVrPresenting.value = false
  })
}

function getRuntimeViewer(): RuntimeGaussianViewer | null {
  return viewer as (GaussianSplats3D.Viewer & RuntimeGaussianViewer) | null
}

function renderSceneWithCamera(runtime: RuntimeGaussianViewer, camera: THREE.PerspectiveCamera) {
  if (!runtime.renderer || !runtime.splatMesh) return

  const savedAutoClear = runtime.renderer.autoClear
  if (runtime.threeScene?.children.some((child) => child.visible)) {
    runtime.renderer.render(runtime.threeScene, camera)
    runtime.renderer.autoClear = false
  }

  runtime.renderer.render(runtime.splatMesh, camera)
  runtime.renderer.autoClear = false

  const focusOpacity = runtime.sceneHelper?.getFocusMarkerOpacity?.() ?? 0
  if (focusOpacity > 0 && runtime.sceneHelper?.focusMarker) {
    runtime.renderer.render(runtime.sceneHelper.focusMarker, camera)
  }
  if (runtime.showControlPlane && runtime.sceneHelper?.controlPlane) {
    runtime.renderer.render(runtime.sceneHelper.controlPlane, camera)
  }

  runtime.renderer.autoClear = savedAutoClear
}

function startStereoPreviewLoop() {
  const runtime = getRuntimeViewer()
  if (!runtime?.renderer || !runtime.camera) return

  const baseCamera = runtime.camera
  const leftCamera = baseCamera.clone()
  const rightCamera = baseCamera.clone()

  const renderStereoFrame = () => {
    const currentRuntime = getRuntimeViewer()
    if (!currentRuntime?.renderer || !currentRuntime.camera) return

    currentRuntime.controls?.update()
    currentRuntime.update?.()

    const renderer = currentRuntime.renderer
    const camera = currentRuntime.camera
    const canvas = renderer.domElement
    const width = canvas.width
    const height = canvas.height
    const halfWidth = Math.floor(width / 2)

    camera.updateMatrixWorld()
    leftCamera.copy(camera)
    rightCamera.copy(camera)

    const eyeOffset = new THREE.Vector3(stereoIpd.value / 2, 0, 0)
    const leftOffset = eyeOffset.clone().multiplyScalar(-1).applyQuaternion(camera.quaternion)
    const rightOffset = eyeOffset.clone().applyQuaternion(camera.quaternion)

    leftCamera.position.add(leftOffset)
    rightCamera.position.add(rightOffset)
    leftCamera.aspect = Math.max(0.1, halfWidth / height)
    rightCamera.aspect = Math.max(0.1, halfWidth / height)
    leftCamera.updateProjectionMatrix()
    rightCamera.updateProjectionMatrix()
    leftCamera.updateMatrixWorld()
    rightCamera.updateMatrixWorld()

    renderer.setScissorTest(true)
    renderer.setViewport(0, 0, halfWidth, height)
    renderer.setScissor(0, 0, halfWidth, height)
    renderSceneWithCamera(currentRuntime, leftCamera)

    renderer.setViewport(halfWidth, 0, width - halfWidth, height)
    renderer.setScissor(halfWidth, 0, width - halfWidth, height)
    renderSceneWithCamera(currentRuntime, rightCamera)
    renderer.setScissorTest(false)

    stereoRafId = window.requestAnimationFrame(renderStereoFrame)
  }

  stopStereoLoop()
  renderStereoFrame()
}

function updateCameraDebug() {
  const camera = getRuntimeViewer()?.camera
  if (!camera) return
  cameraPosition.value = [camera.position.x, camera.position.y, camera.position.z]
    .map((item) => item.toFixed(2))
    .join(', ')
}

async function addSplatSceneWithFallback(payload: BrainDanceViewerPayload, config: BrainDanceVrConfig): Promise<string> {
  const candidates = config.preferCompressedModel ? getVrModelCandidates(payload.ply) : [payload.ply]
  let lastError: unknown = null

  for (const candidate of candidates) {
    try {
      status.value = `加载 3DGS 模型：${candidate}`
      await viewer?.addSplatScene(candidate, {
        showLoadingUI: true,
        progressiveLoad: true,
        splatAlphaRemovalThreshold: 5,
        position: config.worldPosition,
        rotation: makeSceneRotationY(config.worldRotationY),
        scale: [config.worldScale, config.worldScale, config.worldScale],
      })
      return candidate
    } catch (error) {
      lastError = error
      console.warn('[BrainDance VR] 模型加载失败，尝试下一个候选:', candidate, error)
    }
  }

  throw lastError || new Error('没有可用的 3DGS 模型候选')
}

async function loadScene(input: unknown) {
  if (!containerRef.value) return

  const payload = normalizePayload(input)
  errorMessage.value = ''
  activePayload.value = payload
  status.value = '初始化 WebXR Viewer'

  disposeViewer()
  containerRef.value.innerHTML = ''

  stopStereoLoop()

  const config = await loadVrConfig(deriveVrConfigUrl(payload))
  if (scaleOverride != null) {
    config.worldScale = scaleOverride
  }
  if (rotationOverride != null) {
    config.worldRotationY = rotationOverride
  }
  activeConfig.value = config
  scaleStep = Math.max(0.02, config.worldScale * 0.1)

  const mode = previewMode.value
  viewer = new GaussianSplats3D.Viewer({
    rootElement: containerRef.value,
    cameraUp: [0, 1, 0],
    initialCameraPosition: [0, config.userHeight, config.startDistance],
    initialCameraLookAt: [0, config.userHeight, 0],
    sharedMemoryForWorkers: typeof crossOriginIsolated !== 'undefined' ? crossOriginIsolated : false,
    gpuAcceleratedSort: false,
    integerBasedSort: false,
    halfPrecisionCovariancesOnGPU: true,
    antialiased: false,
    ignoreDevicePixelRatio: true,
    dynamicScene: false,
    webXRMode: mode === 'webxr' ? GaussianSplats3D.WebXRMode.VR : GaussianSplats3D.WebXRMode.None,
    sphericalHarmonicsDegree: 0,
    selfDrivenMode: mode !== 'stereo',
    useBuiltInControls: mode !== 'webxr',
  })

  if (mode === 'webxr') {
    installXrSessionListeners()
  }
  activeModelUrl.value = await addSplatSceneWithFallback(payload, config)
  if (mode !== 'stereo') {
    viewer.start()
  }
  if (mode === 'stereo') {
    startStereoPreviewLoop()
  }
  status.value = mode === 'webxr' ? '模型已加载，点击 Enter VR 进入 SteamVR' : '模型已加载，可在桌面调试模型尺度和视角'
}

function resetView() {
  scaleOverride = null
  const payload = activePayload.value || getInitialPayload()
  void loadScene(payload)
}

function adjustScale(delta: number) {
  if (!activeConfig.value) return
  const nextScale = Math.max(0.05, activeConfig.value.worldScale + delta)
  scaleOverride = nextScale
  activeConfig.value = {
    ...activeConfig.value,
    worldScale: nextScale,
  }

  // mkkellogg Viewer 第一版没有稳定暴露 scene transform 更新 API，重载可保持 VR 相机链路正确。
  const payload = activePayload.value || getInitialPayload()
  void loadScene({
    ...payload,
    ply: activeModelUrl.value || payload.ply,
  })
}

function adjustRotation(delta: number) {
  if (!activeConfig.value) return
  rotationOverride = activeConfig.value.worldRotationY + delta
  activeConfig.value = {
    ...activeConfig.value,
    worldRotationY: rotationOverride,
  }

  const payload = activePayload.value || getInitialPayload()
  void loadScene({
    ...payload,
    ply: activeModelUrl.value || payload.ply,
  })
}

function moveDebugCamera(localDelta: THREE.Vector3) {
  const runtime = getRuntimeViewer()
  if (!runtime?.camera) return
  localDelta.applyQuaternion(runtime.camera.quaternion)
  runtime.camera.position.add(localDelta)
  runtime.forceRenderNextFrame?.()
  updateCameraDebug()
}

function selectMode(mode: PreviewMode) {
  if (mode === previewMode.value) return
  switchPreviewMode(mode)
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === 'r' || event.key === 'R') resetView()
  if (event.key === '[') adjustScale(-scaleStep)
  if (event.key === ']') adjustScale(scaleStep)
  if (event.key === 'q' || event.key === 'Q') adjustRotation(-0.1)
  if (event.key === 'e' || event.key === 'E') adjustRotation(0.1)
  if (event.key === 'w' || event.key === 'W') moveDebugCamera(new THREE.Vector3(0, 0, -0.12))
  if (event.key === 's' || event.key === 'S') moveDebugCamera(new THREE.Vector3(0, 0, 0.12))
  if (event.key === 'a' || event.key === 'A') moveDebugCamera(new THREE.Vector3(-0.12, 0, 0))
  if (event.key === 'd' || event.key === 'D') moveDebugCamera(new THREE.Vector3(0.12, 0, 0))
  if (event.key === '1') selectMode('desktop')
  if (event.key === '2') selectMode('stereo')
  if (event.key === '3') selectMode('webxr')
}

async function bootstrap(input?: unknown) {
  try {
    await loadScene(input ?? getInitialPayload())
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    errorMessage.value = message
    status.value = 'VR Viewer 初始化失败'
    console.error('[BrainDance VR] 初始化失败:', error)
  }
}

onMounted(() => {
  window.loadModelFromFlutter = (input: unknown) => {
    void bootstrap(input)
  }

  window.addEventListener('keydown', onKeydown)
  startFpsLoop()
  void bootstrap()
})

onBeforeUnmount(() => {
  window.cancelAnimationFrame(statsRafId)
  stopStereoLoop()
  window.removeEventListener('keydown', onKeydown)
  delete window.loadModelFromFlutter
  disposeViewer()
})
</script>

<template>
  <main class="vr-page">
    <div ref="containerRef" class="vr-canvas" />

    <section class="desktop-panel" aria-label="BrainDance VR 状态">
      <header class="panel-header">
        <p class="eyebrow">BrainDance</p>
        <h1>{{ sceneLabel }}</h1>
      </header>

      <div class="status-block">
        <p>{{ status }}</p>
        <p v-if="errorMessage" class="error-text">{{ errorMessage }}</p>
      </div>

      <dl class="metrics">
        <div>
          <dt>Mode</dt>
          <dd>{{ previewMode }}</dd>
        </div>
        <div>
          <dt>FPS</dt>
          <dd>{{ fps }}</dd>
        </div>
        <div>
          <dt>Scale</dt>
          <dd>{{ scaleLabel }}</dd>
        </div>
        <div>
          <dt>XR</dt>
          <dd>{{ isVrPresenting ? 'Presenting' : 'Ready' }}</dd>
        </div>
      </dl>

      <dl class="debug-list">
        <div>
          <dt>Model</dt>
          <dd>{{ activeModelUrl || '-' }}</dd>
        </div>
        <div>
          <dt>Poses</dt>
          <dd>{{ activePayload?.poses || '-' }}</dd>
        </div>
        <div>
          <dt>Position</dt>
          <dd>{{ positionLabel }}</dd>
        </div>
        <div>
          <dt>RotationY</dt>
          <dd>{{ rotationLabel }}</dd>
        </div>
        <div>
          <dt>Camera</dt>
          <dd>{{ cameraPosition }}</dd>
        </div>
      </dl>

      <div class="mode-row">
        <button type="button" :class="{ active: previewMode === 'desktop' }" @click="selectMode('desktop')">Desktop</button>
        <button type="button" :class="{ active: previewMode === 'stereo' }" @click="selectMode('stereo')">Stereo</button>
        <button type="button" :class="{ active: previewMode === 'webxr' }" @click="selectMode('webxr')">WebXR</button>
      </div>

      <div class="button-row">
        <button type="button" @click="adjustScale(-scaleStep)">缩小</button>
        <button type="button" @click="resetView">重置</button>
        <button type="button" @click="adjustScale(scaleStep)">放大</button>
      </div>

      <p class="hint">1/2/3 切换 Desktop/Stereo/WebXR，WASD 调试相机，[ / ] 缩放，Q/E 旋转，R 重置。</p>
    </section>
  </main>
</template>
