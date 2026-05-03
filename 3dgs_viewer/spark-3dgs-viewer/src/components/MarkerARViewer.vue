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

const postBridgeMessage = (payload: Record<string, unknown>) => {
  window.BrainDanceChannel?.postMessage?.(JSON.stringify(payload))
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

const disposeViewer = () => {
  renderer?.setAnimationLoop(null)

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

  mindarThree?.stop?.()
  mindarThree = null
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
    }) => MindARInstance

    mindarThree = new MindARThree({
      container: containerRef.value,
      imageTargetSrc: params.targetUrl,
      maxTrack: 1,
      uiLoading: false,
      uiScanning: false,
      uiError: false,
    })

    renderer = mindarThree.renderer
    scene = mindarThree.scene
    camera = mindarThree.camera

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5))
    renderer.outputColorSpace = THREE.SRGBColorSpace

    status.value = '模型加载中...'
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

    splatMesh = new SplatMesh({
      url: params.modelUrl,
      editable: true,
    })

    await splatMesh.initialized
    arRoot.add(splatMesh)

    anchor.onTargetFound = () => {
      status.value = '已识别纸板'
    }

    anchor.onTargetLost = () => {
      status.value = '跟踪丢失，请重新对准纸板'
    }

    status.value = '请允许摄像头权限，并将纸板放入画面'
    await mindarThree.start()

    renderer.setAnimationLoop(() => {
      if (!renderer || !scene || !camera) return
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
  disposeViewer()
})
</script>

<template>
  <div class="ar-page">
    <div ref="containerRef" class="ar-container"></div>
    <div class="ar-header">BrainDance Marker AR</div>
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
</style>
