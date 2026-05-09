<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import ArControlPanel from './ArControlPanel.vue'
import { applyArTransform, cloneArTransform } from '../ar/arTransform'
import { parseArParams } from '../ar/parseArParams'
import type { ArTransform } from '../types/ar'

type ThreeModule = typeof import('three')
type SparkModule = typeof import('@sparkjsdev/spark')
type WebGLRendererInstance = InstanceType<ThreeModule['WebGLRenderer']>
type SceneInstance = InstanceType<ThreeModule['Scene']>
type CameraInstance = InstanceType<ThreeModule['Camera']>
type GroupInstance = InstanceType<ThreeModule['Group']>
type SparkRendererInstance = InstanceType<SparkModule['SparkRenderer']>
type SplatMeshInstance = InstanceType<SparkModule['SplatMesh']>

type MindARInstance = {
  renderer: WebGLRendererInstance
  scene: SceneInstance
  camera: CameraInstance
  addAnchor: (index: number) => {
    group: GroupInstance
    onTargetFound?: () => void
    onTargetLost?: () => void
  }
  start: () => Promise<void>
  stop?: () => void
}

type MediaDeviceLike = {
  deviceId: string
  label: string
  kind: string
  index: number
}

const containerRef = ref<HTMLDivElement | null>(null)
const status = ref('准备启动 AR...')

const params = parseArParams()
const initialTransform: ArTransform = {
  scale: params.scale,
  rotation: [...params.rotation],
  offset: [...params.offset],
}
const transform = ref<ArTransform>(cloneArTransform(initialTransform))

let mindarThree: MindARInstance | null = null
let renderer: WebGLRendererInstance | null = null
let scene: SceneInstance | null = null
let camera: CameraInstance | null = null
let spark: SparkRendererInstance | null = null
let splatMesh: SplatMeshInstance | null = null
let arRoot: GroupInstance | null = null

const preferredMainBackCameraPattern = /back|rear|environment|wide|main|0|后|后置|主|广角/i
const rejectedAuxCameraPattern = /tele|zoom|macro|depth|ultra|front|长焦|微距|景深|前置|超广角/i

const postBridgeMessage = (payload: Record<string, unknown>) => {
  window.BrainDanceChannel?.postMessage?.(JSON.stringify(payload))
}

const serializeTrackValue = (value: unknown): unknown => {
  if (value == null) return value
  if (typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') return value
  if (Array.isArray(value)) return value.slice(0, 8)
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>
    return Object.fromEntries(Object.entries(record).slice(0, 12))
  }
  return String(value)
}

const serializeTrackMap = (value: MediaTrackCapabilities | MediaTrackSettings | MediaTrackConstraints) => (
  Object.fromEntries(
    Object.entries(value).map(([key, entry]) => [key, serializeTrackValue(entry)]),
  )
)

const cameraLabelIndex = (label: string) => {
  const match = label.match(/camera\s+(\d+)/i)
  return match ? Number(match[1]) : null
}

const buildMainBackConstraints = (constraints: MediaStreamConstraints, deviceId?: string) => {
  const video = typeof constraints.video === 'object' && constraints.video !== null
    ? { ...constraints.video }
    : {}
  const advanced = Array.isArray(video.advanced) ? [...video.advanced] : []
  const deviceConstraint = deviceId ? { deviceId: { exact: deviceId } } : {}

  return {
    ...constraints,
    audio: false,
    video: {
      ...video,
      ...deviceConstraint,
      facingMode: { ideal: 'environment' },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      aspectRatio: { ideal: 1.7777777778 },
      resizeMode: 'none',
      zoom: { ideal: 1 },
      advanced: [
        { zoom: 1, focusMode: 'continuous' },
        ...advanced,
      ],
    },
  } as MediaStreamConstraints
}

const getVideoInputDevices = async () => {
  try {
    const devices = await navigator.mediaDevices?.enumerateDevices?.()
    return devices
      ?.filter((device) => device.kind === 'videoinput')
      .map((device, index) => ({
        index,
        label: device.label,
        deviceId: device.deviceId,
        groupId: device.groupId,
      })) ?? []
  } catch (_) {
    return []
  }
}

const tuneCameraTrack = async (stream: MediaStream) => {
  const [track] = stream.getVideoTracks()
  if (!track) return
  const capabilities = typeof track.getCapabilities === 'function'
    ? track.getCapabilities()
    : {}
  const settingsBefore = typeof track.getSettings === 'function'
    ? track.getSettings()
    : {}

  const advanced: MediaTrackConstraintSet[] = []
  const zoomCapability = (capabilities as MediaTrackCapabilities & { zoom?: { min?: number, max?: number } }).zoom
  if (zoomCapability) {
    advanced.push({ zoom: Math.max(1, zoomCapability.min ?? 1) } as MediaTrackConstraintSet)
  }
  const focusModes = (capabilities as MediaTrackCapabilities & { focusMode?: string[] }).focusMode
  if (Array.isArray(focusModes) && focusModes.includes('continuous')) {
    advanced.push({ focusMode: 'continuous' } as MediaTrackConstraintSet)
  }

  if (advanced.length > 0) {
    try {
      await track.applyConstraints({ advanced })
    } catch (error) {
      postBridgeMessage({
        status: 'info',
        msg: 'Marker AR track tuning skipped',
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }

  const settingsAfter = typeof track.getSettings === 'function'
    ? track.getSettings()
    : {}
  postBridgeMessage({
    status: 'info',
    msg: 'Marker AR camera track',
    label: track.label,
    capabilities: serializeTrackMap(capabilities),
    settingsBefore: serializeTrackMap(settingsBefore),
    settingsAfter: serializeTrackMap(settingsAfter),
  })

  try {
    const devices = await getVideoInputDevices()
    postBridgeMessage({
      status: 'info',
      msg: 'Marker AR camera devices after permission',
      cameras: devices.map((device) => ({
        index: device.index,
        label: device.label,
        idPrefix: device.deviceId.slice(0, 8),
        groupPrefix: device.groupId.slice(0, 8),
      })),
    })
  } catch (_) {}
}

const stopCameraStream = (stream: MediaStream) => {
  stream.getTracks().forEach((track) => track.stop())
}

const shouldRetryCameraTrack = (stream: MediaStream) => {
  const [track] = stream.getVideoTracks()
  const index = cameraLabelIndex(track?.label || '')
  return index !== null && index >= 3
}

const openPreferredCameraStream = async (
  constraints: MediaStreamConstraints,
  getUserMedia: (constraints: MediaStreamConstraints) => Promise<MediaStream>,
) => {
  const firstConstraints = buildMainBackConstraints(constraints)
  postBridgeMessage({
    status: 'info',
    msg: 'Marker AR getUserMedia constraints',
    constraints: firstConstraints,
  })
  const firstStream = await getUserMedia(firstConstraints)
  if (!shouldRetryCameraTrack(firstStream)) return firstStream

  const devices = await getVideoInputDevices()
  const candidates = devices
    .filter((device) => {
      const index = cameraLabelIndex(device.label)
      return index !== null && index < 3
    })
    .sort((a, b) => (cameraLabelIndex(a.label) ?? 99) - (cameraLabelIndex(b.label) ?? 99))

  for (const candidate of candidates) {
    try {
      stopCameraStream(firstStream)
      const retryConstraints = buildMainBackConstraints(constraints, candidate.deviceId)
      postBridgeMessage({
        status: 'info',
        msg: 'Marker AR retry camera device',
        index: candidate.index,
        label: candidate.label,
        idPrefix: candidate.deviceId.slice(0, 8),
      })
      return await getUserMedia(retryConstraints)
    } catch (error) {
      postBridgeMessage({
        status: 'info',
        msg: 'Marker AR retry camera failed',
        label: candidate.label,
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }

  return await getUserMedia(firstConstraints)
}

const installGetUserMediaDiagnostics = () => {
  const mediaDevices = navigator.mediaDevices
  if (!mediaDevices?.getUserMedia) return
  const originalGetUserMedia = mediaDevices.getUserMedia.bind(mediaDevices)
  mediaDevices.getUserMedia = async (constraints) => {
    try {
      const stream = await openPreferredCameraStream(
        constraints || { video: true },
        originalGetUserMedia,
      )
      await tuneCameraTrack(stream)
      return stream
    } catch (error) {
      const domError = error as DOMException
      postBridgeMessage({
        status: 'error',
        msg: 'getUserMedia failed',
        name: domError?.name,
        message: domError?.message,
        constraint: (domError as DOMException & { constraint?: string })?.constraint,
        constraints,
      })
      throw error
    }
  }
}

const pickMainBackCamera = async () => {
  if (params.camera !== 'main-back') return null
  const mediaDevices = navigator.mediaDevices
  if (!mediaDevices?.enumerateDevices) return null

  try {
    const devices = await mediaDevices.enumerateDevices()
    const videoInputs = devices
      .filter((device) => device.kind === 'videoinput')
      .map((device, index) => ({
        deviceId: device.deviceId,
        label: device.label || '',
        kind: device.kind,
        index,
      })) as MediaDeviceLike[]

    // 部分 Android WebView 不暴露摄像头 label，且 environment 默认可能映射到长焦。
    // 因此优先用 label 命中主摄/广角；若 label 不可用，则按常见 Camera2 顺序选择第一个 videoinput。
    const preferredByIndex = videoInputs[params.cameraIndex]
    const preferredByLabel = videoInputs.find((device) => (
      preferredMainBackCameraPattern.test(device.label) &&
      !rejectedAuxCameraPattern.test(device.label)
    ))
    const preferred = preferredByIndex ?? preferredByLabel ?? videoInputs[0] ?? null

    postBridgeMessage({
      status: 'info',
      msg: 'Marker AR camera candidates',
      selected: preferred?.label || preferred?.deviceId || null,
      selectedIndex: preferred?.index ?? null,
      selectedBy: preferredByIndex ? 'url-index' : preferredByLabel ? 'label' : 'first-videoinput',
      cameras: videoInputs.map((device) => ({
        index: device.index,
        label: device.label,
        idPrefix: device.deviceId.slice(0, 8),
      })),
    })

    return preferred?.deviceId || null
  } catch (error) {
    postBridgeMessage({
      status: 'info',
      msg: 'Marker AR camera enumeration skipped',
      error: error instanceof Error ? error.message : String(error),
    })
    return null
  }
}

const switchCamera = () => {
  const url = new URL(window.location.href)
  const current = Number(url.searchParams.get('cameraIndex') || params.cameraIndex || 0)
  const next = Number.isFinite(current) ? current + 1 : 1
  url.searchParams.set('camera', 'main-back')
  url.searchParams.set('cameraIndex', String(next))
  window.location.href = url.toString()
}

const getVideoDiagnostics = () => {
  const video = containerRef.value?.querySelector('video')
  if (!video) return null
  return {
    readyState: video.readyState,
    videoWidth: video.videoWidth,
    videoHeight: video.videoHeight,
    clientWidth: video.clientWidth,
    clientHeight: video.clientHeight,
    paused: video.paused,
    muted: video.muted,
    zIndex: video.style.zIndex,
  }
}

const normalizeMindArLayers = () => {
  const container = containerRef.value
  if (!container) return

  // MindAR 默认把摄像头 video 放到 z-index: -2；在 Android WebView 中会被黑色容器背景盖住。
  // 这里强制把视频放到 WebGL canvas 下方但仍位于容器背景上方，避免 AR 已启动但画面黑屏。
  const videos = container.querySelectorAll('video')
  videos.forEach((video) => {
    video.style.zIndex = '0'
    video.style.objectFit = 'cover'
    video.style.background = 'transparent'
    video.style.pointerEvents = 'none'
  })

  if (renderer?.domElement) {
    renderer.domElement.style.zIndex = '1'
    renderer.domElement.style.pointerEvents = 'none'
  }

  const canvases = container.querySelectorAll('canvas')
  canvases.forEach((canvas) => {
    canvas.style.zIndex = canvas === renderer?.domElement ? '1' : '2'
    canvas.style.pointerEvents = 'none'
  })

  Array.from(container.children).forEach((child) => {
    if (child instanceof HTMLElement && child.tagName !== 'VIDEO' && child.tagName !== 'CANVAS') {
      child.style.zIndex = child.style.zIndex || '2'
      child.style.pointerEvents = 'none'
    }
  })
}

watch(
  transform,
  (value) => {
    if (!arRoot) return
    applyArTransform(arRoot, value)
  },
  { deep: true },
)

const resetTransform = () => {
  transform.value = cloneArTransform(initialTransform)
}

const disposeViewer = async () => {
  renderer?.setAnimationLoop(null)

  const activeMindAr = mindarThree
  mindarThree = null
  if (activeMindAr?.stop) {
    try {
      await activeMindAr.stop()
    } catch (error) {
      console.warn('[MarkerARViewer] stop mindar failed:', error)
    }
  }

  if (splatMesh) {
    splatMesh.removeFromParent()
    splatMesh.dispose()
    splatMesh = null
  }

  if (spark) {
    spark.removeFromParent()
    spark = null
  }

  if (renderer) {
    renderer.dispose()
    if (renderer.domElement?.parentNode) {
      renderer.domElement.parentNode.removeChild(renderer.domElement)
    }
    renderer = null
  }

  arRoot = null
  scene = null
  camera = null
}

onMounted(async () => {
  if (!containerRef.value) return

  try {
    postBridgeMessage({
      status: 'info',
      msg: 'Marker AR params resolved',
      href: window.location.href,
      model: params.modelUrl,
      target: params.targetUrl,
    })

    status.value = '正在检查 Marker 目标文件...'
    const targetResponse = await fetch(params.targetUrl, { method: 'HEAD' })
    postBridgeMessage({
      status: 'info',
      msg: 'Marker AR target HEAD result',
      target: params.targetUrl,
      ok: targetResponse.ok,
      httpStatus: targetResponse.status,
    })
    if (!targetResponse.ok) {
      throw new Error(`Marker 目标文件不存在或不可访问：${params.targetUrl}`)
    }

    status.value = 'AR 引擎加载中...'
    postBridgeMessage({ status: 'info', msg: 'Marker AR loading runtime modules' })
    installGetUserMediaDiagnostics()
    const [threeModule, sparkModule, mindarModule] = await Promise.all([
      import('three'),
      import('@sparkjsdev/spark'),
      import('mind-ar/dist/mindar-image-three.prod.js'),
    ])
    const THREE = threeModule
    const { SparkRenderer, SplatMesh } = sparkModule
    postBridgeMessage({ status: 'info', msg: 'Marker AR runtime modules loaded' })

    const MindARThree = mindarModule.MindARThree as new (options: {
      container: HTMLElement
      imageTargetSrc: string
      maxTrack: number
      uiLoading: boolean
      uiScanning: boolean
      uiError: boolean
      environmentDeviceId?: string | null
      filterMinCF?: number
      filterBeta?: number
      warmupTolerance?: number
      missTolerance?: number
    }) => MindARInstance

    const environmentDeviceId = await pickMainBackCamera()

    mindarThree = new MindARThree({
      container: containerRef.value,
      imageTargetSrc: params.targetUrl,
      maxTrack: 1,
      uiLoading: false,
      uiScanning: false,
      uiError: false,
      environmentDeviceId,
      filterMinCF: params.filterMinCF,
      filterBeta: params.filterBeta,
      warmupTolerance: params.warmupTolerance,
      missTolerance: params.missTolerance,
    })

    renderer = mindarThree.renderer
    scene = mindarThree.scene
    camera = mindarThree.camera

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, params.pixelRatio))
    renderer.outputColorSpace = THREE.SRGBColorSpace

    status.value = '准备 AR 与模型...'
    spark = new SparkRenderer({
      renderer,
      maxStdDev: Math.sqrt(7),
      preUpdate: false,
    })
    scene.add(spark)

    const anchor = mindarThree.addAnchor(0)
    arRoot = new THREE.Group()
    anchor.group.add(arRoot)
    applyArTransform(arRoot, transform.value)

    const startPromise = mindarThree.start()

    splatMesh = new SplatMesh({
      url: params.modelUrl,
      editable: false,
    })
    await Promise.all([startPromise, splatMesh.initialized])
    arRoot.add(splatMesh)

    anchor.onTargetFound = () => {
      status.value = '已识别纸板'
    }

    anchor.onTargetLost = () => {
      status.value = '跟踪丢失，请重新对准纸板'
    }

    status.value = '请允许摄像头权限，并将纸板放入画面'
    postBridgeMessage({ status: 'info', msg: 'Marker AR starting camera' })
    normalizeMindArLayers()
    status.value = '请将纸板放入画面'
    postBridgeMessage({
      status: 'ready',
      msg: 'Marker AR started',
      video: getVideoDiagnostics(),
      arTuning: {
        scale: params.scale,
        pixelRatio: params.pixelRatio,
        filterMinCF: params.filterMinCF,
        filterBeta: params.filterBeta,
        warmupTolerance: params.warmupTolerance,
        missTolerance: params.missTolerance,
        camera: environmentDeviceId ? 'main-back-device' : 'environment-fallback',
        cameraIndex: params.cameraIndex,
      },
    })

    renderer.setAnimationLoop(() => {
      if (!renderer || !scene || !camera) return
      normalizeMindArLayers()
      renderer.render(scene, camera)
    })
  } catch (error) {
    console.error('[MarkerARViewer] init error:', error)
    postBridgeMessage({
      status: 'error',
      msg: error instanceof Error ? error.message : String(error),
      href: window.location.href,
      model: params.modelUrl,
      target: params.targetUrl,
    })
    status.value = error instanceof Error
      ? error.message
      : 'AR 启动失败，请检查摄像头权限和 Marker 目标文件'
  }
})
onBeforeUnmount(() => {
  void disposeViewer()
})
</script>

<template>
  <div class="ar-page">
    <div ref="containerRef" class="ar-container"></div>
    <div class="ar-header">BrainDance Marker AR</div>
    <button type="button" class="ar-camera-button" @click="switchCamera">切换镜头</button>
    <ArControlPanel v-model="transform" @reset="resetTransform" />
    <div class="ar-status">{{ status }}</div>
  </div>
</template>

<style scoped>
.ar-page,
.ar-container {
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background: #000;
}

.ar-header {
  position: fixed;
  top: 20px;
  left: 50%;
  z-index: 20;
  transform: translateX(-50%);
  padding: 10px 14px;
  border-radius: 999px;
  color: #fff;
  background: rgba(0, 0, 0, 0.52);
  backdrop-filter: blur(8px);
  font-size: 14px;
}

.ar-status {
  position: fixed;
  left: 50%;
  bottom: 24px;
  transform: translateX(-50%);
  z-index: 20;
  padding: 10px 14px;
  border-radius: 999px;
  color: white;
  background: rgba(0, 0, 0, 0.55);
  font-size: 14px;
  width: min(92vw, 560px);
  text-align: center;
}

.ar-camera-button {
  position: fixed;
  top: 72px;
  right: 16px;
  z-index: 22;
  border: 0;
  border-radius: 999px;
  padding: 10px 14px;
  color: #fff;
  background: rgba(12, 18, 30, 0.76);
  backdrop-filter: blur(8px);
  font-size: 14px;
}
</style>
