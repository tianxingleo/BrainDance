<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import * as THREE from 'three'
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d'
import { deriveVrConfigUrl, getInitialPayload, normalizePayload } from '../engine/payload'
import { loadVrConfig } from '../engine/vrConfig'
import { getVrModelCandidates } from '../engine/modelUrl'
import type { BrainDanceViewerPayload, BrainDanceVrConfig } from '../types/viewer'

const containerRef = ref<HTMLDivElement | null>(null)
const status = ref('等待初始化')
const errorMessage = ref('')
const fps = ref(0)
const isVrPresenting = ref(false)
const activePayload = ref<BrainDanceViewerPayload | null>(null)
const activeModelUrl = ref('')
const activeConfig = ref<BrainDanceVrConfig | null>(null)

let viewer: GaussianSplats3D.Viewer | null = null
let frameCount = 0
let lastFpsTime = performance.now()
let scaleStep = 0.1
let rafId = 0
let scaleOverride: number | null = null

const sceneLabel = computed(() => activePayload.value?.sceneId || activePayload.value?.imageId || 'BrainDance VR Preview')
const scaleLabel = computed(() => activeConfig.value?.worldScale.toFixed(2) ?? '1.00')

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
  rafId = window.requestAnimationFrame(startFpsLoop)
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

  const config = await loadVrConfig(deriveVrConfigUrl(payload))
  if (scaleOverride != null) {
    config.worldScale = scaleOverride
  }
  activeConfig.value = config
  scaleStep = Math.max(0.02, config.worldScale * 0.1)

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
    webXRMode: GaussianSplats3D.WebXRMode.VR,
    sphericalHarmonicsDegree: 0,
    selfDrivenMode: true,
  })

  installXrSessionListeners()
  activeModelUrl.value = await addSplatSceneWithFallback(payload, config)
  viewer.start()
  status.value = '模型已加载，点击 Enter VR 进入 SteamVR'
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

function onKeydown(event: KeyboardEvent) {
  if (event.key === 'r' || event.key === 'R') resetView()
  if (event.key === '[') adjustScale(-scaleStep)
  if (event.key === ']') adjustScale(scaleStep)
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
  window.cancelAnimationFrame(rafId)
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
          <dt>FPS</dt>
          <dd>{{ fps }}</dd>
        </div>
        <div>
          <dt>Scale</dt>
          <dd>{{ scaleLabel }}</dd>
        </div>
        <div>
          <dt>VR</dt>
          <dd>{{ isVrPresenting ? 'Presenting' : 'Ready' }}</dd>
        </div>
      </dl>

      <div class="button-row">
        <button type="button" @click="adjustScale(-scaleStep)">缩小</button>
        <button type="button" @click="resetView">重置</button>
        <button type="button" @click="adjustScale(scaleStep)">放大</button>
      </div>

      <p class="hint">PC Chrome / Edge 打开本页，启动 SteamVR 后点击 Enter VR。键盘 [ / ] 缩放，R 重置。</p>
      <p v-if="activeModelUrl" class="model-url">{{ activeModelUrl }}</p>
    </section>
  </main>
</template>
