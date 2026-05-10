<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import * as THREE from 'three'
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d'
import { deriveVrConfigUrl, getInitialPayload, normalizePayload } from '../engine/payload'
import { getPreviewMode, switchPreviewMode, type PreviewMode } from '../engine/previewMode'
import { loadVrConfig } from '../engine/vrConfig'
import { getVrModelCandidates } from '../engine/modelUrl'
import {
  decomposeMatrix,
  normalizeMatrixForViewer,
} from '../engine/bridge'
import type {
  BrainDanceAuthSession,
  BrainDanceRecallMarker,
  BrainDanceRecallModel,
  BrainDanceRecallSearchResult,
  BrainDanceViewerPayload,
  BrainDanceVrConfig,
  RuntimeGaussianViewer,
} from '../types/viewer'

type ControllerHand = 'left' | 'right'
type LoadPhase =
  | 'idle'
  | 'config'
  | 'model'
  | 'ready'
  | 'error'

type GrabState =
  | {
    mode: 'one-hand'
    hand: ControllerHand
    startControllerPosition: THREE.Vector3
    startControllerQuaternion: THREE.Quaternion
    startObjectPosition: THREE.Vector3
    startObjectQuaternion: THREE.Quaternion
  }
  | {
    mode: 'two-hand'
    startDistance: number
    startMidpoint: THREE.Vector3
    startObjectPosition: THREE.Vector3
    startObjectScale: THREE.Vector3
  }

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
const sceneRotation = ref('0.00')
const userScale = ref(1)
const loadPhase = ref<LoadPhase>('idle')
const loadProgress = ref(0)
const loadText = ref('等待加载')
const authSession = ref<BrainDanceAuthSession | null>(null)
const modelList = ref<BrainDanceRecallModel[]>([])
const markers = ref<BrainDanceRecallMarker[]>([])
const searchResults = ref<BrainDanceRecallSearchResult[]>([])
const activeModelId = ref('')
const activeSearchQuery = ref('')
const selectedMarkerId = ref('')
const selectedSearchResultId = ref('')
const isMenuOpen = ref(true)

let viewer: GaussianSplats3D.Viewer | null = null
let frameCount = 0
let lastFpsTime = performance.now()
let statsRafId = 0
let stereoRafId = 0
let controllerRafId = 0
let activeModelIndex = -1
let scaleOverride: number | null = null
let rotationOverride: number | null = null
let currentControllerFrame = 0
let xrSession: XRSession | null = null
let leftController: THREE.Group | null = null
let rightController: THREE.Group | null = null
let controllerRay: THREE.Line | null = null
let controllerTip: THREE.Mesh | null = null
let hudCanvas: HTMLCanvasElement | null = null
let hudTexture: THREE.CanvasTexture | null = null
let hudMesh: THREE.Mesh | null = null
let hudContext: CanvasRenderingContext2D | null = null
let lastHudDrawTime = 0
let sceneRoot: THREE.Group | null = null
let xrRig: THREE.Group | null = null
let worldRoot: THREE.Group | null = null
let introGlint: THREE.Points | null = null
let grabState: GrabState | null = null
const buttonLatch = new Map<string, boolean>()

const sceneLabel = computed(() => activePayload.value?.sceneId || activePayload.value?.imageId || 'BrainDance VR Viewer')
const modelLabel = computed(() => {
  if (activeModelIndex >= 0 && modelList.value[activeModelIndex]) {
    return modelList.value[activeModelIndex]?.name || modelList.value[activeModelIndex]?.displayName || modelList.value[activeModelIndex]?.id
  }
  return activePayload.value?.sceneId || activePayload.value?.imageId || '当前场景'
})
const scaleLabel = computed(() => activeConfig.value?.worldScale.toFixed(2) ?? '1.00')
const positionLabel = computed(() => activeConfig.value?.worldPosition.map((item) => item.toFixed(2)).join(', ') ?? '0, 0, 0')
const rotationLabel = computed(() => activeConfig.value?.worldRotationY.toFixed(2) ?? '0.00')
const filteredModels = computed(() => {
  const query = activeSearchQuery.value.trim().toLowerCase()
  if (!query) return modelList.value
  return modelList.value.filter((item) => {
    const haystack = [
      item.id,
      item.name,
      item.displayName,
      item.description,
      ...(item.tags || []),
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase()
    return haystack.includes(query)
  })
})

function makeSceneRotationY(rotationY: number): [number, number, number, number] {
  const quaternion = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), rotationY)
  return [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
}

function setLoadState(phase: LoadPhase, text: string, progress = loadProgress.value) {
  loadPhase.value = phase
  loadText.value = text
  loadProgress.value = THREE.MathUtils.clamp(progress, 0, 1)
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
  sceneRoot = null
  xrRig = null
  worldRoot = null
  leftController = null
  rightController = null
  controllerRay = null
  controllerTip = null
  grabState = null
  buttonLatch.clear()
}

function resetRuntimeState() {
  activeModelIndex = -1
  activeModelUrl.value = ''
  errorMessage.value = ''
  loadProgress.value = 0
  loadText.value = '等待加载'
  loadPhase.value = 'idle'
  selectedMarkerId.value = ''
  selectedSearchResultId.value = ''
}

function updateFps() {
  frameCount += 1
  const now = performance.now()
  if (now - lastFpsTime < 1000) return
  fps.value = frameCount
  frameCount = 0
  lastFpsTime = now
}

function updateCameraDebug() {
  const camera = getRuntimeViewer()?.camera
  if (!camera) return
  cameraPosition.value = [camera.position.x, camera.position.y, camera.position.z]
    .map((item) => item.toFixed(2))
    .join(', ')
  const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ')
  sceneRotation.value = THREE.MathUtils.radToDeg(euler.y).toFixed(2)
}

function startStatsLoop() {
  const tick = () => {
    updateFps()
    updateCameraDebug()
    statsRafId = window.requestAnimationFrame(tick)
  }
  tick()
}

function stopStereoLoop() {
  if (!stereoRafId) return
  window.cancelAnimationFrame(stereoRafId)
  stereoRafId = 0
}

function stopControllerLoop() {
  if (!controllerRafId) return
  window.cancelAnimationFrame(controllerRafId)
  controllerRafId = 0
}

function getRuntimeViewer(): RuntimeGaussianViewer | null {
  return viewer as (GaussianSplats3D.Viewer & RuntimeGaussianViewer) | null
}

function ensureSceneRoots() {
  const runtime = getRuntimeViewer()
  if (!runtime?.threeScene) return
  if (sceneRoot) return

  sceneRoot = new THREE.Group()
  xrRig = new THREE.Group()
  worldRoot = new THREE.Group()
  xrRig.add(worldRoot)
  sceneRoot.add(xrRig)
  runtime.threeScene.add(sceneRoot)
}

function disposeHud() {
  if (getRuntimeViewer()?.threeScene && hudMesh) {
    getRuntimeViewer()!.threeScene!.remove(hudMesh)
  }
  hudMesh?.geometry?.dispose?.()
  ;(hudMesh?.material as THREE.Material | undefined)?.dispose?.()
  hudTexture?.dispose()
  hudCanvas = null
  hudTexture = null
  hudMesh = null
  hudContext = null
}

function createHud() {
  const runtime = getRuntimeViewer()
  if (!runtime?.threeScene || hudMesh) return

  hudCanvas = document.createElement('canvas')
  hudCanvas.width = 1024
  hudCanvas.height = 512
  hudContext = hudCanvas.getContext('2d')
  if (!hudContext) return

  hudTexture = new THREE.CanvasTexture(hudCanvas)
  hudTexture.colorSpace = THREE.SRGBColorSpace
  const material = new THREE.MeshBasicMaterial({
    map: hudTexture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  })
  hudMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(1.35, 0.675),
    material,
  )
  hudMesh.renderOrder = 999
  hudMesh.visible = false
  runtime.threeScene.add(hudMesh)
}

function createIntroGlint() {
  if (!worldRoot || introGlint) return
  const geometry = new THREE.BufferGeometry()
  const points = new Float32Array(600 * 3)
  for (let index = 0; index < 600; index += 1) {
    const i3 = index * 3
    const theta = (index / 600) * Math.PI * 2
    const radius = 0.7 + (index % 17) * 0.02
    points[i3] = Math.cos(theta) * radius
    points[i3 + 1] = Math.sin(theta * 2) * 0.15
    points[i3 + 2] = Math.sin(theta) * radius
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(points, 3))
  const material = new THREE.PointsMaterial({
    color: 0x9ed0c6,
    size: 0.018,
    transparent: true,
    opacity: 0.72,
  })
  introGlint = new THREE.Points(geometry, material)
  worldRoot.add(introGlint)
}

function disposeIntroGlint() {
  if (worldRoot && introGlint) worldRoot.remove(introGlint)
  introGlint?.geometry?.dispose?.()
  ;(introGlint?.material as THREE.Material | undefined)?.dispose?.()
  introGlint = null
}

function drawHud() {
  if (!hudContext || !hudCanvas) return
  const ctx = hudContext
  const { width, height } = hudCanvas
  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = 'rgba(10, 12, 16, 0.82)'
  ctx.fillRect(0, 0, width, height)
  ctx.strokeStyle = 'rgba(230, 235, 240, 0.24)'
  ctx.lineWidth = 3
  ctx.strokeRect(2, 2, width - 4, height - 4)

  ctx.fillStyle = '#f7f8fb'
  ctx.font = '700 48px Inter, sans-serif'
  ctx.fillText('BrainDance VR', 44, 72)
  ctx.font = '28px Inter, sans-serif'
  ctx.fillStyle = 'rgba(247, 248, 251, 0.82)'
  ctx.fillText(`${loadPhase.value.toUpperCase()} · ${loadText.value}`, 44, 118)

  const barWidth = 520
  ctx.fillStyle = 'rgba(247, 248, 251, 0.14)'
  ctx.fillRect(44, 148, barWidth, 18)
  ctx.fillStyle = '#9ed0c6'
  ctx.fillRect(44, 148, barWidth * loadProgress.value, 18)

  ctx.font = '24px Inter, sans-serif'
  ctx.fillStyle = '#f7f8fb'
  ctx.fillText(`FPS ${fps.value || '--'}   |   模型 ${modelLabel.value}`, 44, 224)
  ctx.fillText(`状态 ${status.value}`, 44, 266)
  ctx.fillText(`模式 ${previewMode.value}   |   XR ${isVrPresenting.value ? 'Presenting' : 'Ready'}`, 44, 308)
  ctx.fillText(`场景尺度 ${scaleLabel.value}   |   旋转Y ${rotationLabel.value}`, 44, 350)
  ctx.fillStyle = 'rgba(247, 248, 251, 0.62)'
  ctx.fillText('左摇杆：移动/漫步   右摇杆：转向/升降', 44, 414)
  ctx.fillText('A/X：重置   B/Y：菜单   Trigger：射线选择', 44, 452)

  if (authSession.value?.displayName || authSession.value?.email) {
    ctx.fillStyle = '#f2c38f'
    ctx.fillText(`用户 ${authSession.value.displayName || authSession.value.email}`, 44, 494)
  }

  hudTexture!.needsUpdate = true
}

function placeHudInFrontOfUser() {
  if (!hudMesh || !viewer?.renderer?.xr) return
  const camera = viewer.renderer.xr.getCamera()
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion).normalize()
  hudMesh.position.copy(camera.position).addScaledVector(forward, 1.35).add(new THREE.Vector3(0, -0.18, 0))
  hudMesh.quaternion.copy(camera.quaternion)
  hudMesh.visible = isMenuOpen.value || !isVrPresenting.value
}

function updateHud(nowMs: number) {
  if (!hudMesh) return
  placeHudInFrontOfUser()
  if (nowMs - lastHudDrawTime > 180) {
    lastHudDrawTime = nowMs
    drawHud()
  }
}

function getControllerByHandedness(handedness: XRHandedness) {
  if (handedness === 'left') return leftController
  if (handedness === 'right') return rightController
  return null
}

function ensureControllerRig() {
  const runtime = getRuntimeViewer()
  if (!runtime?.renderer || !runtime.threeScene) return
  if (leftController && rightController) return

  leftController = runtime.renderer.xr.getController(0)
  rightController = runtime.renderer.xr.getController(1)
  runtime.threeScene.add(leftController)
  runtime.threeScene.add(rightController)

  const lineGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, -1.5),
  ])
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0x9ed0c6 })
  controllerRay = new THREE.Line(lineGeometry, lineMaterial)
  controllerRay.visible = false
  rightController.add(controllerRay)

  const tipMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff })
  controllerTip = new THREE.Mesh(
    new THREE.SphereGeometry(0.018, 16, 16),
    tipMaterial,
  )
  controllerTip.position.set(0, 0, -1.5)
  rightController.add(controllerTip)
}

function getWorldRootScale() {
  return getSceneManipulationTarget()?.scale.x || worldRoot?.scale.x || 1
}

function getSceneManipulationTarget() {
  return getRuntimeViewer()?.splatMesh || worldRoot
}

function isButtonPressed(gamepad: Gamepad, indexes: number[]) {
  return indexes.some((index) => Boolean(gamepad.buttons[index]?.pressed))
}

function wasPressedNow(hand: ControllerHand, action: string, pressed: boolean) {
  const key = `${hand}:${action}`
  const previous = buttonLatch.get(key) || false
  buttonLatch.set(key, pressed)
  return pressed && !previous
}

function getControllerWorldPose(controller: THREE.Object3D) {
  return {
    position: controller.getWorldPosition(new THREE.Vector3()),
    quaternion: controller.getWorldQuaternion(new THREE.Quaternion()),
  }
}

function getActiveControllerPoses() {
  if (!leftController || !rightController) return null
  return {
    left: getControllerWorldPose(leftController),
    right: getControllerWorldPose(rightController),
  }
}

function beginOneHandGrab(hand: ControllerHand, controller: THREE.Object3D) {
  const target = getSceneManipulationTarget()
  if (!target) return
  const pose = getControllerWorldPose(controller)
  grabState = {
    mode: 'one-hand',
    hand,
    startControllerPosition: pose.position,
    startControllerQuaternion: pose.quaternion,
    startObjectPosition: target.position.clone(),
    startObjectQuaternion: target.quaternion.clone(),
  }
}

function beginTwoHandGrab() {
  const target = getSceneManipulationTarget()
  if (!target) return
  const poses = getActiveControllerPoses()
  if (!poses) return
  const midpoint = poses.left.position.clone().add(poses.right.position).multiplyScalar(0.5)
  grabState = {
    mode: 'two-hand',
    startDistance: Math.max(0.001, poses.left.position.distanceTo(poses.right.position)),
    startMidpoint: midpoint,
    startObjectPosition: target.position.clone(),
    startObjectScale: target.scale.clone(),
  }
}

function updateGrabState(leftGrip: boolean, rightGrip: boolean) {
  const target = getSceneManipulationTarget()
  if (!target) return
  if (!leftGrip && !rightGrip) {
    grabState = null
    return
  }

  if (leftGrip && rightGrip) {
    if (grabState?.mode !== 'two-hand') beginTwoHandGrab()
    const poses = getActiveControllerPoses()
    if (!poses || grabState?.mode !== 'two-hand') return
    const currentDistance = Math.max(0.001, poses.left.position.distanceTo(poses.right.position))
    const currentMidpoint = poses.left.position.clone().add(poses.right.position).multiplyScalar(0.5)
    const scaleFactor = THREE.MathUtils.clamp(currentDistance / grabState.startDistance, 0.25, 4)
    target.scale.copy(grabState.startObjectScale).multiplyScalar(scaleFactor)
    target.position.copy(grabState.startObjectPosition).add(currentMidpoint.sub(grabState.startMidpoint))
    userScale.value = getWorldRootScale()
    getRuntimeViewer()?.forceRenderNextFrame?.()
    return
  }

  const activeHand: ControllerHand = leftGrip ? 'left' : 'right'
  const controller = activeHand === 'left' ? leftController : rightController
  if (!controller) return
  if (grabState?.mode !== 'one-hand' || grabState.hand !== activeHand) beginOneHandGrab(activeHand, controller)
  if (grabState?.mode !== 'one-hand') return

  const pose = getControllerWorldPose(controller)
  const rotationDelta = pose.quaternion.clone().multiply(grabState.startControllerQuaternion.clone().invert())
  const translationDelta = pose.position.clone().sub(grabState.startControllerPosition)
  // 单手抓取沿用移动端 viewer 的“操作模型而不是改相机”思路，避免破坏 WebXR 头部追踪。
  target.position.copy(grabState.startObjectPosition).add(translationDelta)
  target.quaternion.copy(rotationDelta.multiply(grabState.startObjectQuaternion))
  getRuntimeViewer()?.forceRenderNextFrame?.()
}

function updateControllerState(nowMs: number) {
  const runtime = getRuntimeViewer()
  if (!runtime?.renderer || !xrSession) return
  ensureControllerRig()

  const session = runtime.renderer.xr.getSession() || xrSession
  if (!session) return
  const dt = currentControllerFrame > 0 ? THREE.MathUtils.clamp((nowMs - currentControllerFrame) / 1000, 1 / 120, 0.06) : 1 / 90
  currentControllerFrame = nowMs

  let leftGrip = false
  let rightGrip = false
  for (const source of session.inputSources || []) {
    const gamepad = source.gamepad
    if (!gamepad) continue
    const axes = gamepad.axes || []
    const x = Math.abs(axes[2] ?? axes[0] ?? 0) > 0.18 ? axes[2] ?? axes[0] ?? 0 : 0
    const y = Math.abs(axes[3] ?? axes[1] ?? 0) > 0.18 ? axes[3] ?? axes[1] ?? 0 : 0
    const controller = getControllerByHandedness(source.handedness)
    if (!controller) continue
    const hand = source.handedness === 'left' ? 'left' : source.handedness === 'right' ? 'right' : null
    if (!hand) continue
    const gripPressed = isButtonPressed(gamepad, [1, 2, 3])
    if (hand === 'left') leftGrip = gripPressed
    if (hand === 'right') rightGrip = gripPressed

    if (source.handedness === 'left') {
      moveRig(x, -y, dt)
      if (wasPressedNow(hand, 'reset', isButtonPressed(gamepad, [4, 5]))) {
        resetView()
      }
    } else if (source.handedness === 'right') {
      turnRig(x, dt)
      if (Math.abs(y) > 0) {
        moveRig(0, 0, -y, dt)
      }
      if (wasPressedNow(hand, 'menu', isButtonPressed(gamepad, [0, 1, 4, 5]))) {
        isMenuOpen.value = !isMenuOpen.value
      }
    }
  }

  updateGrabState(leftGrip, rightGrip)
  updateHud(nowMs)
}

function startControllerLoop() {
  const tick = (nowMs: number) => {
    updateControllerState(nowMs)
    controllerRafId = window.requestAnimationFrame(tick)
  }
  controllerRafId = window.requestAnimationFrame(tick)
}

function renderSceneWithCamera(runtime: RuntimeGaussianViewer, camera: THREE.PerspectiveCamera) {
  if (!runtime.renderer || !runtime.splatMesh) return
  runtime.renderer.render(runtime.splatMesh, camera)
  if (worldRoot) runtime.renderer.render(worldRoot, camera)
}

function startStereoPreviewLoop() {
  const runtime = getRuntimeViewer()
  if (!runtime?.renderer || !runtime.camera) return

  const leftCamera = runtime.camera.clone()
  const rightCamera = runtime.camera.clone()

  const renderStereoFrame = () => {
    const currentRuntime = getRuntimeViewer()
    if (!currentRuntime?.renderer || !currentRuntime.camera) return
    const renderer = currentRuntime.renderer
    const camera = currentRuntime.camera
    const canvas = renderer.domElement
    const width = canvas.width
    const height = canvas.height
    const halfWidth = Math.floor(width / 2)

    camera.updateMatrixWorld()
    leftCamera.copy(camera)
    rightCamera.copy(camera)

    const eyeOffset = new THREE.Vector3(0.032, 0, 0)
    const leftOffset = eyeOffset.clone().multiplyScalar(-1).applyQuaternion(camera.quaternion)
    const rightOffset = eyeOffset.clone().applyQuaternion(camera.quaternion)

    leftCamera.position.add(leftOffset)
    rightCamera.position.add(rightOffset)
    leftCamera.aspect = Math.max(0.1, halfWidth / height)
    rightCamera.aspect = Math.max(0.1, halfWidth / height)
    leftCamera.updateProjectionMatrix()
    rightCamera.updateProjectionMatrix()

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

function resolveMatrixPayload(matrix: unknown) {
  const normalized = normalizeMatrixForViewer(matrix)
  if (!normalized || normalized.length !== 16) return null
  const runtime = getRuntimeViewer()
  const rawMatrix = new THREE.Matrix4().fromArray(normalized)
  const finalMatrix = new THREE.Matrix4()
  const splatMesh = runtime?.splatMesh
  if (splatMesh) {
    splatMesh.updateMatrixWorld()
    finalMatrix.copy(splatMesh.matrixWorld).multiply(rawMatrix)
  } else {
    finalMatrix.copy(rawMatrix)
  }
  return decomposeMatrix(finalMatrix.toArray())
}

function selectModel(model: BrainDanceRecallModel) {
  const index = modelList.value.findIndex((item) => item.id === model.id)
  if (index < 0) return
  activeModelIndex = index
  activeModelUrl.value = model.modelUrl || model.ply
  activeModelId.value = model.id
  loadModel(model, { preserveState: true })
}

function selectMarker(marker: BrainDanceRecallMarker) {
  selectedMarkerId.value = marker.id
  if (marker.matrix) {
    flyToMatrix(marker.matrix)
  } else if (marker.position) {
    flyToPosition(marker.position)
  }
}

function selectSearchResult(result: BrainDanceRecallSearchResult) {
  selectedSearchResultId.value = result.id
  if (result.matrix) {
    flyToMatrix(result.matrix)
  } else if (result.markerId) {
    const marker = markers.value.find((item) => item.id === result.markerId)
    if (marker) selectMarker(marker)
  }
}

function flyToMatrix(matrix: unknown) {
  const runtime = getRuntimeViewer()
  const resolved = resolveMatrixPayload(matrix)
  if (!runtime?.camera || !resolved) return
  runtime.camera.position.copy(resolved.position)
  runtime.camera.quaternion.copy(resolved.quaternion)
  runtime.forceRenderNextFrame?.()
}

function flyToPosition(position: [number, number, number]) {
  const runtime = getRuntimeViewer()
  if (!runtime?.camera) return
  runtime.camera.position.set(position[0], position[1], position[2])
  runtime.forceRenderNextFrame?.()
}

async function addSplatSceneWithFallback(payload: BrainDanceViewerPayload, config: BrainDanceVrConfig): Promise<string> {
  const sourceUrl = payload.modelUrl || payload.ply
  const candidates = config.preferCompressedModel ? getVrModelCandidates(sourceUrl) : [sourceUrl]
  let lastError: unknown = null

  for (const candidate of candidates) {
    try {
      setLoadState('model', `加载 3DGS 模型：${candidate}`, 0.2)
      await viewer?.addSplatScene(candidate, {
        showLoadingUI: false,
        progressiveLoad: false,
        optimizeSplatData: true,
        freeIntermediateSplatData: true,
        splatAlphaRemovalThreshold: 5,
        onProgress: (percentComplete: number, percentCompleteLabel?: string, loaderStatus?: number) => {
          const percent = Number(percentComplete)
          if (Number.isFinite(percent)) {
            const normalized = THREE.MathUtils.clamp(percent / 100, 0, 1)
            const progress = loaderStatus === 1
              ? THREE.MathUtils.clamp(0.96 + normalized * 0.04, loadProgress.value, 1)
              : THREE.MathUtils.clamp(0.2 + normalized * 0.76, loadProgress.value, 0.96)
            setLoadState('model', loaderStatus === 1 ? '解析并构建高斯数据' : `下载模型数据 ${percentCompleteLabel || ''}`.trim(), progress)
            drawHud()
          }
        },
        position: config.worldPosition,
        rotation: makeSceneRotationY(config.worldRotationY),
        // 与 my-3dgs-viewer 保持同一处轴系修正：水平面保持不变，只在加载层镜像 Z 轴。
        scale: [config.worldScale, config.worldScale, -config.worldScale],
      })
      return candidate
    } catch (error) {
      lastError = error
      console.warn('[BrainDance VR] 模型加载失败，尝试下一个候选:', candidate, error)
    }
  }

  throw lastError || new Error('没有可用的 3DGS 模型候选')
}

function normalizeModelPayloadList(payload: BrainDanceViewerPayload) {
  const list = payload.modelList || []
  if (list.length > 0) return list
  const fallbackModel: BrainDanceRecallModel = {
    id: payload.sceneId || payload.imageId || 'current',
    name: payload.sceneId || payload.imageId || '当前模型',
    displayName: payload.sceneId || payload.imageId || '当前模型',
    ply: payload.modelUrl || payload.ply,
    modelUrl: payload.modelUrl || payload.ply,
    poses: payload.posesUrl || payload.poses,
    posesUrl: payload.posesUrl || payload.poses,
  }
  return [fallbackModel]
}

async function loadModel(model: BrainDanceRecallModel, options: { preserveState?: boolean } = {}) {
  if (!containerRef.value) return
  const payload = activePayload.value || getInitialPayload()
  const config = activeConfig.value || (await loadVrConfig(deriveVrConfigUrl(payload)))
  const nextPayload = {
    ...payload,
    ply: model.ply || model.modelUrl || payload.ply,
    modelUrl: model.modelUrl || model.ply || payload.modelUrl,
    poses: model.poses || model.posesUrl || payload.poses,
    posesUrl: model.posesUrl || model.poses || payload.posesUrl,
  }

  if (!options.preserveState) {
    resetRuntimeState()
  }

  activePayload.value = nextPayload
  activeModelUrl.value = nextPayload.modelUrl || nextPayload.ply
  setLoadState('config', '读取 VR 配置', 0.06)
  const resolvedConfig = await loadVrConfig(deriveVrConfigUrl(nextPayload))
  if (scaleOverride != null) resolvedConfig.worldScale = scaleOverride
  if (rotationOverride != null) resolvedConfig.worldRotationY = rotationOverride
  activeConfig.value = resolvedConfig
  userScale.value = resolvedConfig.worldScale

  setLoadState('model', '初始化 WebXR Viewer', 0.15)
  disposeHud()
  disposeIntroGlint()
  disposeViewer()
  if (containerRef.value) containerRef.value.innerHTML = ''

  viewer = new GaussianSplats3D.Viewer({
    rootElement: containerRef.value,
    cameraUp: [0, 1, 0],
    initialCameraPosition: [0, resolvedConfig.userHeight, resolvedConfig.startDistance],
    initialCameraLookAt: [0, resolvedConfig.userHeight, 0],
    sharedMemoryForWorkers: typeof crossOriginIsolated !== 'undefined' ? crossOriginIsolated : false,
    gpuAcceleratedSort: false,
    integerBasedSort: false,
    halfPrecisionCovariancesOnGPU: true,
    antialiased: false,
    ignoreDevicePixelRatio: true,
    dynamicScene: true,
    webXRMode: previewMode.value === 'webxr' ? GaussianSplats3D.WebXRMode.VR : GaussianSplats3D.WebXRMode.None,
    sphericalHarmonicsDegree: 0,
    selfDrivenMode: previewMode.value !== 'stereo',
    useBuiltInControls: previewMode.value !== 'webxr',
  })

  if (previewMode.value === 'webxr') {
    installXrSessionListeners()
  }

  activeModelIndex = modelList.value.findIndex((item) => item.id === model.id)
  if (activeModelIndex < 0) activeModelIndex = 0
  ensureSceneRoots()
  createHud()
  createIntroGlint()

  const candidate = await addSplatSceneWithFallback(nextPayload, resolvedConfig)
  activeModelUrl.value = candidate
  setLoadState('ready', '模型已加载，准备进入 VR', 1)
  status.value = '模型已加载，网页端 Recall/VR 壳已就绪'

  if (previewMode.value !== 'stereo') {
    viewer.start()
  }
  if (previewMode.value === 'stereo') {
    startStereoPreviewLoop()
  }
  if (xrSession) startControllerLoop()
  drawHud()
}

async function bootstrap(input?: unknown) {
  try {
    const payload = normalizePayload(input ?? getInitialPayload())
    activePayload.value = payload
    if (payload.previewMode) previewMode.value = payload.previewMode
    authSession.value = payload.authSession || null
    modelList.value = normalizeModelPayloadList(payload)
    markers.value = payload.markers || []
    searchResults.value = payload.searchResults || []
    activeModelId.value = modelList.value[0]?.id || ''
    activeSearchQuery.value = ''
    setLoadState('config', '准备读取模型配置', 0.04)
    const initialModel = modelList.value[0] || normalizeModelPayloadList(payload)[0]
    if (initialModel) {
      await loadModel(initialModel)
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    errorMessage.value = message
    status.value = 'VR Viewer 初始化失败'
    loadPhase.value = 'error'
    loadText.value = '初始化失败'
    console.error('[BrainDance VR] 初始化失败:', error)
  }
}

function resetView() {
  scaleOverride = null
  rotationOverride = null
  const current = modelList.value[activeModelIndex] || modelList.value[0]
  if (!current) return
  void loadModel(current)
}

function adjustScale(delta: number) {
  if (!activeConfig.value) return
  const nextScale = Math.max(0.05, activeConfig.value.worldScale + delta)
  scaleOverride = nextScale
  activeConfig.value = {
    ...activeConfig.value,
    worldScale: nextScale,
  }
  userScale.value = nextScale
  const current = modelList.value[activeModelIndex] || modelList.value[0]
  if (current) void loadModel(current, { preserveState: true })
}

function adjustRotation(delta: number) {
  if (!activeConfig.value) return
  rotationOverride = activeConfig.value.worldRotationY + delta
  activeConfig.value = {
    ...activeConfig.value,
    worldRotationY: rotationOverride,
  }
  const current = modelList.value[activeModelIndex] || modelList.value[0]
  if (current) void loadModel(current, { preserveState: true })
}

function moveRig(strafe: number, forward: number, dt = 1 / 90, vertical = 0) {
  const runtime = getRuntimeViewer()
  if (!runtime?.camera || !xrRig) return
  const direction = new THREE.Vector3()
  runtime.camera.getWorldDirection(direction)
  direction.y = 0
  direction.normalize()
  const right = new THREE.Vector3().crossVectors(direction, new THREE.Vector3(0, 1, 0)).normalize()
  const delta = new THREE.Vector3()
  delta.addScaledVector(direction, -forward * 1.35 * dt)
  delta.addScaledVector(right, strafe * 1.35 * dt)
  delta.y += vertical * 0.85 * dt
  xrRig.position.add(delta)
  runtime.forceRenderNextFrame?.()
}

function turnRig(turn: number, dt = 1 / 90) {
  if (!xrRig) return
  const angle = -turn * 1.65 * dt
  xrRig.rotateY(angle)
}

function selectMode(mode: PreviewMode) {
  if (mode === previewMode.value) return
  switchPreviewMode(mode)
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === 'r' || event.key === 'R') resetView()
  if (event.key === '[') adjustScale(-0.1)
  if (event.key === ']') adjustScale(0.1)
  if (event.key === 'q' || event.key === 'Q') adjustRotation(-0.1)
  if (event.key === 'e' || event.key === 'E') adjustRotation(0.1)
  if (event.key === 'w' || event.key === 'W') moveRig(0, -1, 1 / 60)
  if (event.key === 's' || event.key === 'S') moveRig(0, 1, 1 / 60)
  if (event.key === 'a' || event.key === 'A') moveRig(-1, 0, 1 / 60)
  if (event.key === 'd' || event.key === 'D') moveRig(1, 0, 1 / 60)
  if (event.key === '1') selectMode('desktop')
  if (event.key === '2') selectMode('stereo')
  if (event.key === '3') selectMode('webxr')
  if (event.key === 'm' || event.key === 'M') isMenuOpen.value = !isMenuOpen.value
}

function installXrSessionListeners() {
  const xr = viewer?.renderer?.xr
  if (!xr?.addEventListener) return
  xr.addEventListener('sessionstart', () => {
    isVrPresenting.value = true
    xrSession = xr.getSession() || xrSession
    ensureSceneRoots()
    createHud()
    drawHud()
    startControllerLoop()
    status.value = 'WebXR 会话已启动'
  })
  xr.addEventListener('sessionend', () => {
    isVrPresenting.value = false
    xrSession = null
    stopControllerLoop()
    if (hudMesh) hudMesh.visible = false
    status.value = 'WebXR 会话已结束'
  })
}

async function enterVrSession() {
  const runtime = getRuntimeViewer()
  if (!runtime?.renderer?.xr || !navigator.xr) return
  runtime.renderer.xr.enabled = true
  runtime.renderer.xr.setReferenceSpaceType?.('local-floor')
  const session = await navigator.xr.requestSession('immersive-vr', {
    optionalFeatures: ['local-floor', 'bounded-floor', 'hand-tracking', 'layers'],
  })
  await runtime.renderer.xr.setSession(session)
  xrSession = session
  isVrPresenting.value = true
}

async function exitVrSession() {
  const runtime = getRuntimeViewer()
  const session = runtime?.renderer?.xr.getSession() || xrSession
  if (session) await session.end()
}

function normalizeWindowHooks() {
  window.loadModelFromFlutter = (input: unknown) => {
    void bootstrap(input)
  }
  window.setThemeFromFlutter = () => {
    drawHud()
  }
  window.setModelListForTimePeeling = (list: unknown, currentId?: unknown) => {
    const nextList = normalizePayload({
      ...activePayload.value,
      modelList: list,
    }).modelList || []
    modelList.value = nextList
    if (currentId) {
      const activeId = String(currentId)
      const index = nextList.findIndex((item) => item.id === activeId)
      if (index >= 0) {
        activeModelIndex = index
        activeModelId.value = activeId
      }
    }
    drawHud()
  }
  window.setBrainDanceSession = (session: unknown) => {
    authSession.value = session && typeof session === 'object' ? normalizePayload({
      ...activePayload.value,
      authSession: session,
    }).authSession || null : null
    drawHud()
  }
  window.setRecallSearchResults = (results: unknown) => {
    searchResults.value = normalizePayload({
      ...activePayload.value,
      searchResults: results,
    }).searchResults || []
    drawHud()
  }
  window.setRecallMarkers = (nextMarkers: unknown) => {
    markers.value = normalizePayload({
      ...activePayload.value,
      markers: nextMarkers,
    }).markers || []
    drawHud()
  }
  window.setRecallQuery = (query: string) => {
    activeSearchQuery.value = query || ''
  }
}

onMounted(() => {
  normalizeWindowHooks()
  window.addEventListener('keydown', onKeydown)
  startStatsLoop()
  void bootstrap()
})

onBeforeUnmount(() => {
  window.cancelAnimationFrame(statsRafId)
  stopStereoLoop()
  stopControllerLoop()
  window.removeEventListener('keydown', onKeydown)
  delete window.loadModelFromFlutter
  delete window.setThemeFromFlutter
  delete window.setModelListForTimePeeling
  delete window.setBrainDanceSession
  delete window.setRecallSearchResults
  delete window.setRecallMarkers
  delete window.setRecallQuery
  disposeHud()
  disposeIntroGlint()
  disposeViewer()
})
</script>

<template>
  <main class="vr-page">
    <div ref="containerRef" class="vr-canvas" />

    <section class="desktop-panel" :class="{ collapsed: !isMenuOpen }" aria-label="BrainDance VR 状态">
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
          <dt>Position</dt>
          <dd>{{ positionLabel }}</dd>
        </div>
        <div>
          <dt>RotationY</dt>
          <dd>{{ rotationLabel }}</dd>
        </div>
        <div>
          <dt>User</dt>
          <dd>{{ authSession?.displayName || authSession?.email || '-' }}</dd>
        </div>
      </dl>

      <div class="mode-row">
        <button type="button" :class="{ active: previewMode === 'desktop' }" @click="selectMode('desktop')">Desktop</button>
        <button type="button" :class="{ active: previewMode === 'stereo' }" @click="selectMode('stereo')">Stereo</button>
        <button type="button" :class="{ active: previewMode === 'webxr' }" @click="selectMode('webxr')">WebXR</button>
      </div>

      <div class="button-row">
        <button type="button" @click="adjustScale(-0.1)">缩小</button>
        <button type="button" @click="resetView">重置</button>
        <button type="button" @click="adjustScale(0.1)">放大</button>
      </div>

      <div class="button-row xr-row">
        <button type="button" @click="enterVrSession">进入 VR</button>
        <button type="button" @click="exitVrSession">退出 VR</button>
        <button type="button" @click="isMenuOpen = !isMenuOpen">面板</button>
      </div>

      <div class="search-panel">
        <input v-model="activeSearchQuery" type="text" placeholder="搜索模型、标签、描述" />
        <div class="search-results">
          <button
            v-for="item in filteredModels"
            :key="item.id"
            type="button"
            :class="{ active: activeModelId === item.id }"
            @click="selectModel(item)"
          >
            <strong>{{ item.name || item.displayName || item.id }}</strong>
            <span>{{ item.description || item.ply }}</span>
          </button>
        </div>
      </div>

      <div class="search-panel">
        <p class="panel-label">Search Results</p>
        <div class="search-results">
          <button
            v-for="item in searchResults"
            :key="item.id"
            type="button"
            :class="{ active: selectedSearchResultId === item.id }"
            @click="selectSearchResult(item)"
          >
            <strong>{{ item.label }}</strong>
            <span>{{ item.description || item.markerId || '-' }}</span>
          </button>
        </div>
      </div>

      <div class="search-panel">
        <p class="panel-label">Markers</p>
        <div class="search-results">
          <button
            v-for="item in markers"
            :key="item.id"
            type="button"
            :class="{ active: selectedMarkerId === item.id }"
            @click="selectMarker(item)"
          >
            <strong>{{ item.label }}</strong>
            <span>{{ item.color || 'marker' }}</span>
          </button>
        </div>
      </div>

      <p class="hint">1/2/3 切换 Desktop/Stereo/WebXR，WASD 漫游，M 显隐面板，[ / ] 缩放，Q/E 旋转，R 重置。VR 内左摇杆移动，右摇杆转向/升降，Grip 抓取场景，双 Grip 缩放。</p>
    </section>
  </main>
</template>
