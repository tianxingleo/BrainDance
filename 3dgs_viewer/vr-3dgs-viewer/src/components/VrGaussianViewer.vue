<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import * as THREE from 'three'
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d'
import { deriveVrConfigUrl, getInitialPayload, normalizePayload, resolveRelativeAssetUrl } from '../engine/payload'
import { getPreviewMode, switchPreviewMode, type PreviewMode } from '../engine/previewMode'
import { loadVrConfig } from '../engine/vrConfig'
import { getVrModelCandidates } from '../engine/modelUrl'
import { mergeStandaloneCatalog } from '../engine/catalog'
import {
  decomposeMatrix,
  normalizeMatrixForViewer,
} from '../engine/bridge'
import type {
  BrainDanceAuthSession,
  BrainDanceNavigationPoint,
  BrainDanceRecallMarker,
  BrainDanceRecallModel,
  BrainDanceRecallSearchResult,
  BrainDanceViewerPayload,
  BrainDanceVrConfig,
  RuntimeGaussianViewer,
} from '../types/viewer'

type ControllerHand = 'left' | 'right'
type InteractionMode = 'explore' | 'inspect'
type SceneScaleMode = 'room' | 'diorama'
type TurnMode = 'snap' | 'smooth'
type QualityPreset = 'ultra' | 'high' | 'balanced' | 'performance' | 'potato'
type HudView = 'controls' | 'models' | 'search' | 'markers' | 'nav'
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

type HudAction = {
  id: string
  label: string
  x: number
  y: number
  width: number
  height: number
  kind: 'tab' | 'button' | 'item'
  accent?: string
  disabled?: boolean
  onActivate: () => void
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
const interactionMode = ref<InteractionMode>('explore')
const sceneScaleMode = ref<SceneScaleMode>('room')
const turnMode = ref<TurnMode>('snap')
const hudView = ref<HudView>('controls')
const moveSpeed = ref(1.15)
const snapTurnAngle = ref(30)
const vignetteEnabled = ref(true)
const floorLockEnabled = ref(true)
const qualityPreset = ref<QualityPreset>('balanced')
const clippingEnabled = ref(false)
const clippingDistance = ref(3.2)
const measurementEnabled = ref(false)
const measurementPoints = ref<[number, number, number][]>([])
const selectedNavPointId = ref('')
const navPointIndex = ref(0)
const controllerDebug = ref('等待 SteamVR 手柄输入')
const lastDirectionHint = ref('暂无导航目标')
const modelSummaryText = ref('等待场景摘要')
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
let hudActions: HudAction[] = []
let hudHoveredActionId: string | null = null
let hudPointerHit = false
let hudPointerDistance = 1.5
let hudPointerWorldPoint = new THREE.Vector3()
let hudRaycaster = new THREE.Raycaster()
let hudPointerTargets: THREE.Object3D[] = []
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
let markerGroup: THREE.Group | null = null
let navGroup: THREE.Group | null = null
let clippingPlaneHelper: THREE.Mesh | null = null
let measurementLine: THREE.Line | null = null
let measurementLabelMesh: THREE.Sprite | null = null
let vignetteMesh: THREE.Mesh | null = null
let lastSnapTurnTime = 0
const buttonLatch = new Map<string, boolean>()
const qualityPresets: QualityPreset[] = ['ultra', 'high', 'balanced', 'performance', 'potato']

const sceneLabel = computed(() => activePayload.value?.sceneId || activePayload.value?.imageId || 'BrainDance VR Viewer')
const modelLabel = computed(() => {
  const selected = modelList.value.find((item) => item.id === activeModelId.value) || modelList.value[activeModelIndex]
  if (selected) {
    return selected.name || selected.displayName || selected.id
  }
  return activePayload.value?.sceneId || activePayload.value?.imageId || '当前场景'
})
const scaleLabel = computed(() => activeConfig.value?.worldScale.toFixed(2) ?? '1.00')
const positionLabel = computed(() => activeConfig.value?.worldPosition.map((item) => item.toFixed(2)).join(', ') ?? '0, 0, 0')
const rotationLabel = computed(() => activeConfig.value?.worldRotationY.toFixed(2) ?? '0.00')
const measurementDistanceLabel = computed(() => {
  if (measurementPoints.value.length < 2) return '未完成'
  const a = measurementPoints.value[0]
  const b = measurementPoints.value[1]
  if (!a || !b) return '未完成'
  const distance = new THREE.Vector3(...a).distanceTo(new THREE.Vector3(...b))
  return `${distance.toFixed(2)} m`
})
const activeSearchResult = computed(() => searchResults.value.find((item) => item.id === selectedSearchResultId.value) || null)
const activeMarker = computed(() => markers.value.find((item) => item.id === selectedMarkerId.value) || null)
const activeEvidence = computed(() => {
  const result = activeSearchResult.value
  if (result) return {
    label: result.label,
    description: result.description,
    imageId: result.imageId,
    score: result.score,
    tags: result.tags,
    createdAt: result.createdAt,
  }
  const marker = activeMarker.value
  if (marker) return {
    label: marker.label,
    description: marker.description,
    imageId: marker.imageId,
    score: marker.score,
    tags: marker.tags,
    createdAt: marker.createdAt,
  }
  return null
})
const navigationPoints = computed(() => activeConfig.value?.navigationPoints || [])
const qualityLabel = computed(() => qualityPreset.value)
const authLabel = computed(() => authSession.value?.displayName || authSession.value?.email || '未登录')
const modelCountLabel = computed(() => {
  if (modelList.value.length === 0) return '0/0'
  const currentIndex = Math.max(0, modelList.value.findIndex((item) => item.id === activeModelId.value))
  return `${currentIndex + 1}/${modelList.value.length}`
})
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

function getSceneActionTargets() {
  const runtime = getRuntimeViewer()
  const targets: THREE.Object3D[] = []
  if (runtime?.splatMesh) targets.push(runtime.splatMesh)
  if (worldRoot && !targets.includes(worldRoot)) targets.push(worldRoot)
  return targets
}

function applySceneDelta(delta: THREE.Vector3) {
  for (const target of getSceneActionTargets()) {
    target.position.add(delta)
  }
  getRuntimeViewer()?.forceRenderNextFrame?.()
}

function applySceneRotationY(angle: number) {
  for (const target of getSceneActionTargets()) {
    target.rotateY(angle)
  }
  getRuntimeViewer()?.forceRenderNextFrame?.()
}

function rotateSceneAroundUser(angle: number) {
  const runtime = getRuntimeViewer()
  const camera = runtime?.renderer?.xr?.isPresenting
    ? runtime.renderer.xr.getCamera()
    : runtime?.camera
  if (!camera) {
    applySceneRotationY(angle)
    return
  }

  const pivot = camera.getWorldPosition(new THREE.Vector3())
  for (const target of getSceneActionTargets()) {
    target.position.sub(pivot).applyAxisAngle(new THREE.Vector3(0, 1, 0), angle).add(pivot)
    target.rotateY(angle)
  }
  runtime?.forceRenderNextFrame?.()
}

function focusSceneOnPoint(point: THREE.Vector3, offset = 1.45) {
  const runtime = getRuntimeViewer()
  const camera = runtime?.renderer?.xr?.isPresenting
    ? runtime.renderer.xr.getCamera()
    : runtime?.camera
  if (!camera) return

  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion).normalize()
  const desired = camera.position.clone().addScaledVector(forward, offset)
  const delta = desired.sub(point)
  applySceneDelta(delta)
}

function resolveInitialModel(payload: BrainDanceViewerPayload, models: BrainDanceRecallModel[]) {
  if (models.length === 0) return null
  const candidates = [payload.activeModelId, payload.sceneId, payload.imageId].filter(
    (item): item is string => Boolean(item && item.trim()),
  )
  for (const candidate of candidates) {
    const match = models.find((item) => item.id === candidate || item.sceneId === candidate)
    if (match) return match
  }

  const payloadUrl = payload.modelUrl || payload.ply
  if (payloadUrl) {
    const match = models.find((item) => item.modelUrl === payloadUrl || item.ply === payloadUrl)
    if (match) return match
  }

  return models[0] || null
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
  hudActions = []
  hudHoveredActionId = null
  hudPointerHit = false
  hudPointerWorldPoint = new THREE.Vector3()
  hudPointerTargets = []
  markerGroup = null
  navGroup = null
  clippingPlaneHelper = null
  measurementLine = null
  measurementLabelMesh = null
  vignetteMesh = null
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
  selectedNavPointId.value = ''
  hudHoveredActionId = null
  measurementPoints.value = []
  lastDirectionHint.value = '暂无导航目标'
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
  hudCanvas.height = 640
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
    new THREE.PlaneGeometry(1.6, 1.0),
    material,
  )
  hudMesh.renderOrder = 999
  hudMesh.visible = false
  runtime.threeScene.add(hudMesh)
  hudPointerTargets = [hudMesh]
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

function disposeObject3D(object: THREE.Object3D | null) {
  if (!object) return
  object.traverse((child) => {
    const mesh = child as THREE.Mesh
    mesh.geometry?.dispose?.()
    const material = mesh.material as THREE.Material | THREE.Material[] | undefined
    if (Array.isArray(material)) material.forEach((item) => item.dispose())
    else material?.dispose?.()
  })
  object.parent?.remove(object)
}

function rebuildSpatialHelpers() {
  ensureSceneRoots()
  if (!worldRoot) return
  disposeObject3D(markerGroup)
  disposeObject3D(navGroup)
  markerGroup = new THREE.Group()
  navGroup = new THREE.Group()
  worldRoot.add(markerGroup)
  worldRoot.add(navGroup)
  buildMarkerMeshes()
  buildNavigationMeshes()
  updateClippingPlaneHelper()
  updateMeasurementVisual()
  updateSceneSummary()
}

function createTextSprite(text: string, color = '#f7f8fb') {
  const canvas = document.createElement('canvas')
  canvas.width = 512
  canvas.height = 128
  const ctx = canvas.getContext('2d')!
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = 'rgba(8, 10, 12, 0.78)'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.strokeStyle = 'rgba(247, 248, 251, 0.22)'
  ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4)
  ctx.fillStyle = color
  ctx.font = '700 34px Inter, sans-serif'
  ctx.fillText(text.slice(0, 24), 24, 58)
  ctx.font = '22px Inter, sans-serif'
  ctx.fillStyle = 'rgba(247, 248, 251, 0.72)'
  ctx.fillText('Trigger 选择 / 导航', 24, 96)
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false })
  const sprite = new THREE.Sprite(material)
  sprite.scale.set(0.72, 0.18, 1)
  return sprite
}

function markerPositionFromInput(input: BrainDanceRecallMarker | BrainDanceRecallSearchResult) {
  if (input.position) return new THREE.Vector3(...input.position)
  if (input.matrix) {
    const resolved = resolveMatrixPayload(input.matrix)
    if (resolved) return resolved.position
  }
  return null
}

function buildMarkerMeshes() {
  if (!markerGroup) return
  const resultMarkers = searchResults.value
    .map((result) => ({
      id: `result:${result.id}`,
      label: result.label,
      color: '#f2c38f',
      source: result,
    }))
  const manualMarkers = markers.value.map((marker) => ({
    id: `marker:${marker.id}`,
    label: marker.label,
    color: marker.color || '#9ed0c6',
    source: marker,
  }))

  for (const item of [...manualMarkers, ...resultMarkers]) {
    const position = markerPositionFromInput(item.source)
    if (!position) continue
    const group = new THREE.Group()
    group.name = item.id
    group.position.copy(position)
    const color = new THREE.Color(item.color)
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.075, 24, 16),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.92 }),
    )
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.12, 0.15, 32),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.66, side: THREE.DoubleSide }),
    )
    ring.rotation.x = -Math.PI / 2
    const sprite = createTextSprite(item.label, item.color)
    sprite.position.set(0, 0.22, 0)
    group.add(sphere, ring, sprite)
    markerGroup.add(group)
  }
}

function buildNavigationMeshes() {
  if (!navGroup) return
  const groupRoot = navGroup
  navigationPoints.value.forEach((point) => {
    const group = new THREE.Group()
    group.name = `nav:${point.id}`
    group.position.set(...point.position)
    const material = new THREE.MeshBasicMaterial({ color: 0x87a5ff, transparent: true, opacity: 0.72 })
    const disk = new THREE.Mesh(new THREE.CircleGeometry(0.16, 32), material)
    disk.rotation.x = -Math.PI / 2
    const stem = new THREE.Mesh(
      new THREE.CylinderGeometry(0.012, 0.012, 0.35, 12),
      new THREE.MeshBasicMaterial({ color: 0x87a5ff, transparent: true, opacity: 0.6 }),
    )
    stem.position.y = 0.17
    const sprite = createTextSprite(point.label, '#d8e0ff')
    sprite.position.y = 0.5
    group.add(disk, stem, sprite)
    groupRoot.add(group)
  })
}

function updateSceneSummary() {
  const summary = activeConfig.value?.summary
  const model = modelList.value[activeModelIndex] || modelList.value[0]
  const objects = summary?.objects?.length ? summary.objects : model?.tags
  const searchable = summary?.searchableObjects?.length ? summary.searchableObjects : objects
  modelSummaryText.value = [
    summary?.sceneType ? `场景：${summary.sceneType}` : model?.description || '场景摘要：可从 model_assets / vr_config 注入',
    objects?.length ? `主要对象：${objects.slice(0, 6).join('、')}` : '',
    searchable?.length ? `可检索：${searchable.slice(0, 6).join('、')}` : '',
  ].filter(Boolean).join('；')
}

function setHudView(view: HudView) {
  hudView.value = view
  drawHud()
}

function nextQualityPreset(offset: number) {
  const index = qualityPresets.indexOf(qualityPreset.value)
  const next = qualityPresets[(index + offset + qualityPresets.length) % qualityPresets.length]
  if (next) setQualityPreset(next)
}

function addHudAction(action: HudAction) {
  hudActions.push(action)
}

function drawHudButton(
  ctx: CanvasRenderingContext2D,
  action: HudAction,
  active = false,
) {
  addHudAction(action)
  const hovered = hudHoveredActionId === action.id
  const accent = action.accent || '#9ed0c6'
  ctx.save()
  ctx.globalAlpha = action.disabled ? 0.38 : 1
  ctx.fillStyle = active
    ? `${accent}66`
    : hovered
      ? 'rgba(247, 248, 251, 0.20)'
      : 'rgba(247, 248, 251, 0.08)'
  ctx.strokeStyle = hovered ? accent : active ? `${accent}cc` : 'rgba(247, 248, 251, 0.20)'
  ctx.lineWidth = hovered ? 3 : 2
  ctx.beginPath()
  ctx.roundRect(action.x, action.y, action.width, action.height, 10)
  ctx.fill()
  ctx.stroke()
  ctx.fillStyle = action.disabled ? 'rgba(247, 248, 251, 0.48)' : '#f7f8fb'
  ctx.font = '700 20px Inter, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(action.label.slice(0, 18), action.x + action.width / 2, action.y + action.height / 2)
  ctx.restore()
}

function drawHudListItem(
  ctx: CanvasRenderingContext2D,
  action: HudAction,
  meta: string,
  active = false,
) {
  addHudAction(action)
  const hovered = hudHoveredActionId === action.id
  const accent = action.accent || '#9ed0c6'
  ctx.save()
  ctx.globalAlpha = action.disabled ? 0.38 : 1
  ctx.fillStyle = active
    ? `${accent}44`
    : hovered
      ? 'rgba(247, 248, 251, 0.16)'
      : 'rgba(247, 248, 251, 0.06)'
  ctx.strokeStyle = hovered ? accent : 'rgba(247, 248, 251, 0.16)'
  ctx.lineWidth = hovered ? 3 : 1.5
  ctx.beginPath()
  ctx.roundRect(action.x, action.y, action.width, action.height, 8)
  ctx.fill()
  ctx.stroke()
  ctx.textAlign = 'left'
  ctx.textBaseline = 'alphabetic'
  ctx.fillStyle = '#f7f8fb'
  ctx.font = '700 22px Inter, sans-serif'
  ctx.fillText(action.label.slice(0, 32), action.x + 18, action.y + 28)
  ctx.fillStyle = 'rgba(247, 248, 251, 0.62)'
  ctx.font = '18px Inter, sans-serif'
  ctx.fillText(meta.slice(0, 64), action.x + 18, action.y + 54)
  ctx.restore()
}

function drawHudTabs(ctx: CanvasRenderingContext2D) {
  const tabs: Array<{ view: HudView; label: string }> = [
    { view: 'controls', label: '控制' },
    { view: 'models', label: `模型 ${modelCountLabel.value}` },
    { view: 'search', label: `结果 ${searchResults.value.length}` },
    { view: 'markers', label: `标记 ${markers.value.length}` },
    { view: 'nav', label: `视角 ${navigationPoints.value.length}` },
  ]
  tabs.forEach((tab, index) => {
    drawHudButton(ctx, {
      id: `tab:${tab.view}`,
      label: tab.label,
      x: 44 + index * 184,
      y: 206,
      width: 168,
      height: 42,
      kind: 'tab',
      onActivate: () => setHudView(tab.view),
    }, hudView.value === tab.view)
  })
}

function drawHudControls(ctx: CanvasRenderingContext2D) {
  const buttons: Array<{ id: string; label: string; active?: boolean; accent?: string; onActivate: () => void }> = [
    { id: 'preview:desktop', label: 'Desktop', active: previewMode.value === 'desktop', onActivate: () => selectMode('desktop') },
    { id: 'preview:stereo', label: 'Stereo', active: previewMode.value === 'stereo', onActivate: () => selectMode('stereo') },
    { id: 'preview:webxr', label: 'WebXR', active: previewMode.value === 'webxr', onActivate: () => selectMode('webxr') },
    { id: 'interaction:explore', label: 'Explore', active: interactionMode.value === 'explore', onActivate: () => setInteractionMode('explore') },
    { id: 'interaction:inspect', label: 'Inspect', active: interactionMode.value === 'inspect', onActivate: () => setInteractionMode('inspect') },
    { id: 'scale:room', label: '1:1', active: sceneScaleMode.value === 'room', onActivate: () => setSceneScaleMode('room') },
    { id: 'scale:diorama', label: '沙盘', active: sceneScaleMode.value === 'diorama', onActivate: () => setSceneScaleMode('diorama') },
  ]
  buttons.forEach((button, index) => {
    drawHudButton(ctx, {
      id: button.id,
      label: button.label,
      x: 44 + index * 134,
      y: 282,
      width: 120,
      height: 42,
      kind: 'button',
      accent: button.accent,
      onActivate: button.onActivate,
    }, button.active)
  })

  const rowTwo: Array<{ id: string; label: string; active?: boolean; accent?: string; onActivate: () => void }> = [
    { id: 'model:prev', label: '上一模型', onActivate: () => selectModelByOffset(-1) },
    { id: 'model:next', label: '下一模型', onActivate: () => selectModelByOffset(1) },
    { id: 'view:reset', label: '重置', onActivate: resetView },
    { id: 'vr:enter', label: '进入VR', accent: '#87a5ff', onActivate: () => { void enterVrSession() } },
    { id: 'vr:exit', label: '退出VR', accent: '#f2c38f', onActivate: () => { void exitVrSession() } },
    { id: 'menu:close', label: '收起HUD', onActivate: () => { isMenuOpen.value = false } },
  ]
  rowTwo.forEach((button, index) => {
    drawHudButton(ctx, {
      id: button.id,
      label: button.label,
      x: 44 + index * 154,
      y: 338,
      width: 140,
      height: 42,
      kind: 'button',
      accent: button.accent,
      onActivate: button.onActivate,
    }, button.active)
  })

  const rowThree: Array<{ id: string; label: string; active?: boolean; accent?: string; onActivate: () => void }> = [
    { id: 'quality:prev', label: `质量 -`, onActivate: () => nextQualityPreset(-1) },
    { id: 'quality:next', label: `质量 +`, onActivate: () => nextQualityPreset(1) },
    { id: 'turn:toggle', label: turnMode.value === 'snap' ? 'Snap' : 'Smooth', active: turnMode.value === 'snap', onActivate: () => { turnMode.value = turnMode.value === 'snap' ? 'smooth' : 'snap' } },
    { id: 'floor:toggle', label: '锁地面', active: floorLockEnabled.value, onActivate: () => { floorLockEnabled.value = !floorLockEnabled.value } },
    { id: 'vignette:toggle', label: '黑边', active: vignetteEnabled.value, onActivate: () => { vignetteEnabled.value = !vignetteEnabled.value } },
    { id: 'clip:toggle', label: '剖切', active: clippingEnabled.value, onActivate: () => { clippingEnabled.value = !clippingEnabled.value; updateClippingPlaneHelper() } },
    { id: 'measure:toggle', label: '测量', active: measurementEnabled.value, onActivate: () => { measurementEnabled.value = !measurementEnabled.value } },
    { id: 'measure:clear', label: '清除', onActivate: clearMeasurement },
  ]
  rowThree.forEach((button, index) => {
    drawHudButton(ctx, {
      id: button.id,
      label: button.label,
      x: 44 + index * 116,
      y: 394,
      width: 102,
      height: 42,
      kind: 'button',
      accent: button.accent,
      onActivate: button.onActivate,
    }, button.active)
  })

  ctx.fillStyle = 'rgba(247, 248, 251, 0.70)'
  ctx.font = '22px Inter, sans-serif'
  ctx.textAlign = 'left'
  ctx.fillText(`当前：${modelLabel.value} / ${qualityLabel.value} / ${measurementDistanceLabel.value}`, 44, 492)
  ctx.fillText(`提示：用控制器光标指向按钮，扣 Trigger 执行；Grip 抓取模型，双手 Grip 缩放。`, 44, 532)
}

function drawHudCollection(ctx: CanvasRenderingContext2D) {
  const yStart = 278
  const rowHeight = 68
  const maxRows = 5
  if (hudView.value === 'models') {
    modelList.value.slice(0, maxRows).forEach((model, index) => {
      drawHudListItem(ctx, {
        id: `model:${model.id}`,
        label: model.name || model.displayName || model.id,
        x: 44,
        y: yStart + index * rowHeight,
        width: 936,
        height: 58,
        kind: 'item',
        onActivate: () => selectModel(model),
      }, model.description || model.ply || '-', activeModelId.value === model.id)
    })
    return
  }

  if (hudView.value === 'search') {
    searchResults.value.slice(0, maxRows).forEach((item, index) => {
      drawHudListItem(ctx, {
        id: `search:${item.id}`,
        label: item.label,
        x: 44,
        y: yStart + index * rowHeight,
        width: 936,
        height: 58,
        kind: 'item',
        accent: '#f2c38f',
        onActivate: () => selectSearchResult(item),
      }, `${item.score != null ? `${Math.round(item.score * 100)}% · ` : ''}${item.description || item.markerId || '-'}`, selectedSearchResultId.value === item.id)
    })
    return
  }

  if (hudView.value === 'markers') {
    markers.value.slice(0, maxRows).forEach((item, index) => {
      drawHudListItem(ctx, {
        id: `marker:${item.id}`,
        label: item.label,
        x: 44,
        y: yStart + index * rowHeight,
        width: 936,
        height: 58,
        kind: 'item',
        accent: item.color || '#9ed0c6',
        onActivate: () => selectMarker(item),
      }, item.description || item.imageId || '-', selectedMarkerId.value === item.id)
    })
    return
  }

  navigationPoints.value.slice(0, maxRows).forEach((item, index) => {
    drawHudListItem(ctx, {
      id: `nav:${item.id}`,
      label: item.label,
      x: 44,
      y: yStart + index * rowHeight,
      width: 936,
      height: 58,
      kind: 'item',
      accent: '#87a5ff',
      onActivate: () => selectNavigationPoint(item.id),
    }, item.description || item.kind || item.position.map((value) => value.toFixed(1)).join(', '), selectedNavPointId.value === item.id)
  })
}

function drawHud() {
  if (!hudContext || !hudCanvas) return
  hudActions = []
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
  ctx.fillText(`用户 ${authLabel.value}   |   状态 ${status.value}`, 44, 184)

  drawHudTabs(ctx)
  if (hudView.value === 'controls') {
    drawHudControls(ctx)
  } else {
    drawHudCollection(ctx)
  }

  const hover = hudActions.find((action) => action.id === hudHoveredActionId)
  ctx.fillStyle = hover ? '#f2c38f' : 'rgba(247, 248, 251, 0.58)'
  ctx.font = '20px Inter, sans-serif'
  ctx.textAlign = 'left'
  ctx.fillText(hover ? `Trigger：${hover.label}` : '光标指向按钮后扣 Trigger 选择；B/Y 收起或展开 HUD。', 44, 614)

  if (hudPointerHit && hudPointerDistance > 0) {
    ctx.fillStyle = '#f2c38f'
    ctx.beginPath()
    ctx.arc(1000, 614, 7, 0, Math.PI * 2)
    ctx.fill()
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

function ensureVignette() {
  const runtime = getRuntimeViewer()
  if (!runtime?.threeScene || vignetteMesh) return
  vignetteMesh = new THREE.Mesh(
    new THREE.RingGeometry(0.36, 0.82, 64),
    new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0.34,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  )
  vignetteMesh.renderOrder = 1000
  vignetteMesh.visible = false
  runtime.threeScene.add(vignetteMesh)
}

function updateVignette(visible: boolean) {
  if (!vignetteEnabled.value) visible = false
  ensureVignette()
  if (!vignetteMesh || !viewer?.renderer?.xr) return
  const camera = viewer.renderer.xr.getCamera()
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion).normalize()
  vignetteMesh.position.copy(camera.position).addScaledVector(forward, 0.32)
  vignetteMesh.quaternion.copy(camera.quaternion)
  vignetteMesh.visible = visible && isVrPresenting.value
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

function getControllerHitCandidates() {
  const candidates: THREE.Object3D[] = []
  if (hudMesh) candidates.push(hudMesh)
  for (const item of hudPointerTargets) {
    if (item && !candidates.includes(item)) candidates.push(item)
  }
  return candidates
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

function wasAxisCrossed(hand: ControllerHand, action: string, value: number, threshold = 0.7) {
  return wasPressedNow(hand, action, Math.abs(value) >= threshold)
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

function getHudPointerController() {
  const session = xrSession || getRuntimeViewer()?.renderer?.xr.getSession() || null
  if (!session) return rightController || leftController
  const sources = Array.from(session.inputSources || [])
  const preferred = sources.find((source: XRInputSource) => source.handedness === 'right')
    || sources.find((source: XRInputSource) => source.handedness === 'left')
  if (preferred) {
    return getControllerByHandedness(preferred.handedness) || rightController || leftController
  }
  return rightController || leftController
}

function getHudActionAtUv(uv: THREE.Vector2) {
  if (!hudCanvas) return null
  const x = uv.x * hudCanvas.width
  const y = (1 - uv.y) * hudCanvas.height
  return hudActions.find(
    (action) =>
      x >= action.x &&
      x <= action.x + action.width &&
      y >= action.y &&
      y <= action.y + action.height,
  ) || null
}

function activateHudAction(action: HudAction | null) {
  if (!action || action.disabled) return false
  try {
    action.onActivate()
    drawHud()
    return true
  } catch (error) {
    console.warn('[BrainDance VR] HUD action failed:', action.id, error)
    return false
  }
}

function updateHudPointer(triggerPressed: boolean) {
  if (!hudMesh) {
    hudHoveredActionId = null
    hudPointerHit = false
    return
  }
  const controller = getHudPointerController()
  if (!controller) {
    hudHoveredActionId = null
    hudPointerHit = false
    return
  }

  const origin = controller.getWorldPosition(new THREE.Vector3())
  const direction = controller.getWorldDirection(new THREE.Vector3()).normalize()
  hudRaycaster.set(origin, direction)
  hudRaycaster.far = Math.max(6, hudPointerDistance + 2)
  const intersections = hudRaycaster.intersectObjects(getControllerHitCandidates(), true)
  const hit = intersections.find((entry) => entry.object === hudMesh || entry.object.parent === hudMesh) || intersections[0]
  const previousHoverId = hudHoveredActionId
  hudPointerHit = Boolean(hit && hit.uv)

  if (!hit || !hit.uv) {
    hudHoveredActionId = null
    if (controllerRay) {
      controllerRay.visible = isMenuOpen.value
      controllerRay.scale.z = Math.max(1, hudPointerDistance / 1.5)
    }
    if (controllerTip) {
      controllerTip.visible = isMenuOpen.value
      controllerTip.position.z = -Math.max(0.05, hudPointerDistance)
    }
    if (previousHoverId !== hudHoveredActionId) drawHud()
    if (triggerPressed && !isMenuOpen.value) {
      handlePrimarySelect()
    }
    return
  }

  hudHoveredActionId = getHudActionAtUv(hit.uv)?.id || null
  hudPointerDistance = hit.distance
  hudPointerWorldPoint = hit.point.clone()

  if (controllerRay) {
    controllerRay.visible = true
    controllerRay.scale.z = Math.max(0.01, hit.distance / 1.5)
  }
  if (controllerTip) {
    controllerTip.visible = true
    controllerTip.position.z = -Math.max(0.05, hit.distance)
  }

  if (controllerRay) {
    controllerRay.visible = true
    controllerRay.scale.z = Math.max(0.01, hit.distance / 1.5)
  }
  if (controllerTip) {
    controllerTip.visible = true
    controllerTip.position.z = -Math.max(0.05, hit.distance)
  }

  if (triggerPressed && hudHoveredActionId) {
    const action = hudActions.find((item) => item.id === hudHoveredActionId) || null
    activateHudAction(action)
  }

  if (hudHoveredActionId && controllerTip && controllerRay) {
    const action = hudActions.find((item) => item.id === hudHoveredActionId)
    if (action) {
      const color = action.kind === 'tab' ? 0x87a5ff : action.kind === 'item' ? 0xf2c38f : 0x9ed0c6
      ;(controllerTip.material as THREE.MeshBasicMaterial).color.setHex(color)
      ;(controllerRay.material as THREE.LineBasicMaterial).color.setHex(color)
    }
  }

  if (previousHoverId !== hudHoveredActionId) {
    drawHud()
  }

  if (triggerPressed && !hudHoveredActionId && !isMenuOpen.value) {
    handlePrimarySelect()
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
  if (!hudMesh) createHud()

  const session = runtime.renderer.xr.getSession() || xrSession
  if (!session) return
  const dt = currentControllerFrame > 0 ? THREE.MathUtils.clamp((nowMs - currentControllerFrame) / 1000, 1 / 120, 0.06) : 1 / 90
  currentControllerFrame = nowMs

  let leftGrip = false
  let rightGrip = false
  let moving = false
  let hudTriggerPressed = false
  const debugLines: string[] = []
  for (const source of session.inputSources || []) {
    const gamepad = source.gamepad
    if (!gamepad) continue
    const axes = gamepad.axes || []
    const x = Math.abs(axes[2] ?? axes[0] ?? 0) > 0.18 ? axes[2] ?? axes[0] ?? 0 : 0
    const y = Math.abs(axes[3] ?? axes[1] ?? 0) > 0.18 ? axes[3] ?? axes[1] ?? 0 : 0
    debugLines.push(`${source.handedness || 'unknown'} profiles=${source.profiles?.join(',') || '-'} axes=[${axes.map((axis) => axis.toFixed(2)).join(', ')}] buttons=${gamepad.buttons.map((button, index) => `${index}:${button.pressed ? 'P' : '-'}:${button.value.toFixed(2)}`).join(' ')}`)
    const controller = getControllerByHandedness(source.handedness)
    if (!controller) continue
    const hand = source.handedness === 'left' ? 'left' : source.handedness === 'right' ? 'right' : null
    if (!hand) continue
    const triggerPressed = isButtonPressed(gamepad, [0, 1, 4])
    const gripPressed = isButtonPressed(gamepad, [2, 3, 5])
    const menuPressed = isButtonPressed(gamepad, [3, 4, 5, 6, 7])
    if (hand === 'left') leftGrip = gripPressed
    if (hand === 'right') rightGrip = gripPressed

    if (source.handedness === 'left') {
      if (isMenuOpen.value && Math.abs(x) >= 0.7) {
        if (wasAxisCrossed(hand, 'model-prev', x)) {
          selectModelByOffset(x > 0 ? 1 : -1)
        }
      } else {
        moving = moveRig(x, -y, dt) || moving
      }
      if (wasPressedNow(hand, 'reset', menuPressed)) {
        resetView()
      }
    } else if (source.handedness === 'right') {
      moving = turnRig(x, dt, nowMs) || moving
      if (isMenuOpen.value && Math.abs(y) >= 0.7) {
        if (wasAxisCrossed(hand, 'model-next', y)) {
          selectModelByOffset(y < 0 ? 1 : -1)
        }
      } else if (Math.abs(y) > 0 && !isVrPresenting.value) {
        moving = moveRig(0, 0, dt, -y) || moving
      }
      if (wasPressedNow(hand, 'menu', menuPressed)) {
        isMenuOpen.value = !isMenuOpen.value
      }
      if (wasPressedNow(hand, 'trigger', triggerPressed)) {
        hudTriggerPressed = true
        if (!isMenuOpen.value) handlePrimarySelect()
      }
    }
  }

  if (debugLines.length > 0) {
    controllerDebug.value = debugLines.join('\n')
  }
  updateGrabState(leftGrip, rightGrip)
  updateDirectionHint()
  updateVignette(moving)
  updateHudPointer(hudTriggerPressed)
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

    const eyeOffset = new THREE.Vector3(0.032, 0, 0)
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
  selectedSearchResultId.value = ''
  selectedNavPointId.value = ''
  if (marker.matrix) {
    flyToMatrix(marker.matrix)
  } else if (marker.position) {
    flyToPosition(marker.position)
  }
}

function selectSearchResult(result: BrainDanceRecallSearchResult) {
  selectedSearchResultId.value = result.id
  selectedMarkerId.value = ''
  selectedNavPointId.value = ''
  if (result.matrix) {
    flyToMatrix(result.matrix)
  } else if (result.position) {
    flyToPosition(result.position)
  } else if (result.markerId) {
    const marker = markers.value.find((item) => item.id === result.markerId)
    if (marker) selectMarker(marker)
  }
}

function selectNavigationPoint(pointId: string) {
  const point = navigationPoints.value.find((item) => item.id === pointId)
  if (!point) return
  selectedNavPointId.value = point.id
  selectedMarkerId.value = ''
  selectedSearchResultId.value = ''
  if (point.matrix) flyToMatrix(point.matrix)
  else flyToPosition(point.position)
}

function nextNavigationPoint() {
  if (navigationPoints.value.length === 0) return
  navPointIndex.value = (navPointIndex.value + 1) % navigationPoints.value.length
  const point = navigationPoints.value[navPointIndex.value]
  if (point) selectNavigationPoint(point.id)
}

function handlePrimarySelect() {
  if (measurementEnabled.value) {
    addMeasurementPointFromCamera()
    return
  }
  if (searchResults.value.length > 0) {
    const currentIndex = Math.max(0, searchResults.value.findIndex((item) => item.id === selectedSearchResultId.value))
    const next = searchResults.value[currentIndex] || searchResults.value[0]
    if (next) selectSearchResult(next)
    return
  }
  if (navigationPoints.value.length > 0) nextNavigationPoint()
}

function selectModelByOffset(offset: number) {
  if (modelList.value.length === 0) return
  const currentIndex = Math.max(0, modelList.value.findIndex((item) => item.id === activeModelId.value))
  const nextIndex = (currentIndex + offset + modelList.value.length) % modelList.value.length
  const nextModel = modelList.value[nextIndex]
  if (nextModel) selectModel(nextModel)
}

function flyToMatrix(matrix: unknown) {
  const runtime = getRuntimeViewer()
  const resolved = resolveMatrixPayload(matrix)
  if (!resolved) return
  if (isVrPresenting.value && runtime?.renderer?.xr?.isPresenting) {
    focusSceneOnPoint(resolved.position)
  } else if (runtime?.camera) {
    runtime.camera.position.copy(resolved.position)
    runtime.camera.quaternion.copy(resolved.quaternion)
  }
  runtime?.forceRenderNextFrame?.()
}

function flyToPosition(position: [number, number, number]) {
  const runtime = getRuntimeViewer()
  if (isVrPresenting.value && runtime?.renderer?.xr?.isPresenting) {
    focusSceneOnPoint(new THREE.Vector3(position[0], position[1], position[2]))
  } else if (runtime?.camera) {
    runtime.camera.position.set(position[0], position[1], position[2])
  }
  runtime?.forceRenderNextFrame?.()
}

function addMeasurementPointFromCamera() {
  const runtime = getRuntimeViewer()
  if (!runtime?.camera) return
  const forward = new THREE.Vector3()
  runtime.camera.getWorldDirection(forward)
  const point = runtime.camera.position.clone().addScaledVector(forward, 1.25)
  const next: [number, number, number] = [point.x, point.y, point.z]
  measurementPoints.value = measurementPoints.value.length >= 2 ? [next] : [...measurementPoints.value, next]
  updateMeasurementVisual()
}

function clearMeasurement() {
  measurementPoints.value = []
  updateMeasurementVisual()
}

function updateMeasurementVisual() {
  disposeObject3D(measurementLine)
  disposeObject3D(measurementLabelMesh)
  measurementLine = null
  measurementLabelMesh = null
  if (!worldRoot || measurementPoints.value.length < 2) return
  const points = measurementPoints.value.map((point) => new THREE.Vector3(...point))
  const firstPoint = points[0]
  const secondPoint = points[1]
  if (!firstPoint || !secondPoint) return
  measurementLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color: 0xf2c38f }),
  )
  const mid = firstPoint.clone().add(secondPoint).multiplyScalar(0.5)
  const sprite = createTextSprite(measurementDistanceLabel.value, '#f2c38f')
  sprite.position.copy(mid).add(new THREE.Vector3(0, 0.12, 0))
  measurementLabelMesh = sprite
  worldRoot.add(measurementLine, measurementLabelMesh)
}

function updateClippingPlaneHelper() {
  disposeObject3D(clippingPlaneHelper)
  clippingPlaneHelper = null
  if (!worldRoot || !clippingEnabled.value) return
  clippingPlaneHelper = new THREE.Mesh(
    new THREE.PlaneGeometry(3.2, 2.2),
    new THREE.MeshBasicMaterial({
      color: 0x87a5ff,
      transparent: true,
      opacity: 0.16,
      side: THREE.DoubleSide,
    }),
  )
  clippingPlaneHelper.position.set(0, activeConfig.value?.userHeight || 1.4, -clippingDistance.value)
  worldRoot.add(clippingPlaneHelper)
}

function updateDirectionHint() {
  const runtime = getRuntimeViewer()
  if (!runtime?.camera) return
  const target = getActiveTargetPosition()
  if (!target) {
    lastDirectionHint.value = '暂无导航目标'
    return
  }
  const cameraPos = runtime.camera.getWorldPosition(new THREE.Vector3())
  const cameraForward = new THREE.Vector3()
  runtime.camera.getWorldDirection(cameraForward)
  cameraForward.y = 0
  cameraForward.normalize()
  const toTarget = target.clone().sub(cameraPos)
  const distance = toTarget.length()
  toTarget.y = 0
  toTarget.normalize()
  const cross = new THREE.Vector3().crossVectors(cameraForward, toTarget).y
  const dot = cameraForward.dot(toTarget)
  const direction = dot > 0.85 ? '前方' : cross > 0 ? '左前方' : '右前方'
  lastDirectionHint.value = `${direction} ${distance.toFixed(1)}m`
}

function resolveAssetUrl(assetUrl: string | undefined) {
  const payload = activePayload.value || getInitialPayload()
  const baseUrl = payload.poses || payload.modelUrl || payload.ply || window.location.href
  return resolveRelativeAssetUrl(assetUrl, baseUrl)
}

function getActiveTargetPosition() {
  const result = activeSearchResult.value
  if (result) return markerPositionFromInput(result)
  const marker = activeMarker.value
  if (marker) return markerPositionFromInput(marker)
  const nav = navigationPoints.value.find((point) => point.id === selectedNavPointId.value)
  if (nav) return new THREE.Vector3(...nav.position)
  return null
}

async function addSplatSceneWithFallback(payload: BrainDanceViewerPayload, config: BrainDanceVrConfig): Promise<string> {
  const sourceUrl = resolveAssetUrl(payload.modelUrl || payload.ply)
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

function extractPoseMatrices(input: unknown): number[][] {
  if (!input || typeof input !== 'object') return []
  const root = input as Record<string, unknown>
  const candidates = Array.isArray(root.frames)
    ? root.frames
    : Array.isArray(root.poses)
      ? root.poses
      : Array.isArray(root.cameras)
        ? root.cameras
        : []

  return candidates
    .map((item) => {
      if (!item || typeof item !== 'object') return null
      const entry = item as Record<string, unknown>
      const matrix = entry.transform_matrix || entry.matrix || entry.transform || entry.camera_to_world
      if (Array.isArray(matrix) && matrix.length === 16) return matrix.map(Number).filter(Number.isFinite)
      if (Array.isArray(matrix) && matrix.length >= 3 && Array.isArray(matrix[0])) {
        return (matrix as unknown[][]).flat().map(Number).filter(Number.isFinite)
      }
      return null
    })
    .filter((matrix): matrix is number[] => Boolean(matrix && matrix.length === 16))
}

async function inferNavigationPoints(payload: BrainDanceViewerPayload, config: BrainDanceVrConfig): Promise<BrainDanceNavigationPoint[] | undefined> {
  if (config.navigationPoints?.length || !payload.poses) return config.navigationPoints
  try {
    const resolvedPosesUrl = resolveAssetUrl(payload.poses)
    const response = await fetch(resolvedPosesUrl, { cache: 'no-cache' })
    if (!response.ok) return undefined
    const contentType = response.headers.get('content-type') || ''
    if (!contentType.includes('json')) return undefined
    const matrices = extractPoseMatrices(await response.json())
    if (matrices.length === 0) return undefined
    const sampleIndexes = Array.from(new Set([
      0,
      Math.floor(matrices.length * 0.25),
      Math.floor(matrices.length * 0.5),
      Math.floor(matrices.length * 0.75),
      matrices.length - 1,
    ])).filter((index) => index >= 0 && index < matrices.length)
    const points = sampleIndexes
      .map((matrixIndex, outputIndex) => {
        const matrix = normalizeMatrixForViewer(matrices[matrixIndex])
        if (!matrix) return null
        const pose = decomposeMatrix(matrix)
        const label = outputIndex === 0 ? '入口点' : outputIndex === 2 ? '中心观察点' : `导览点 ${outputIndex + 1}`
        return {
          id: `pose-${matrixIndex}`,
          label,
          position: [pose.position.x, pose.position.y, pose.position.z] as [number, number, number],
          matrix,
          kind: outputIndex === 0 ? 'entry' : outputIndex === 2 ? 'center' : 'tour',
          description: `来自 webgl_poses 第 ${matrixIndex + 1} 帧`,
        } satisfies BrainDanceNavigationPoint
      })
      .filter(Boolean) as BrainDanceNavigationPoint[]
    return points.length > 0 ? points : undefined
  } catch (error) {
    console.warn('[BrainDance VR] 自动导航点生成失败:', error)
    return undefined
  }
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
  const inferredNavigationPoints = await inferNavigationPoints(nextPayload, resolvedConfig)
  if (inferredNavigationPoints?.length) resolvedConfig.navigationPoints = inferredNavigationPoints
  if (scaleOverride != null) resolvedConfig.worldScale = scaleOverride
  if (rotationOverride != null) resolvedConfig.worldRotationY = rotationOverride
  activeConfig.value = resolvedConfig
  userScale.value = resolvedConfig.worldScale

  setLoadState('model', '初始化 WebXR Viewer', 0.15)
  disposeHud()
  disposeIntroGlint()
  disposeObject3D(markerGroup)
  disposeObject3D(navGroup)
  disposeObject3D(clippingPlaneHelper)
  disposeObject3D(measurementLine)
  disposeObject3D(measurementLabelMesh)
  disposeObject3D(vignetteMesh)
  disposeViewer()
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
  rebuildSpatialHelpers()
  applySceneScaleMode()
  applyQualityPreset()

  const candidate = await addSplatSceneWithFallback(nextPayload, resolvedConfig)
  activeModelUrl.value = candidate
  disposeIntroGlint()
  rebuildSpatialHelpers()
  applySceneScaleMode()
  setLoadState('ready', '模型已加载，准备进入 VR', 1)
  status.value = '模型已加载，网页端 Recall/VR 壳已就绪'

  if (previewMode.value !== 'stereo') {
    viewer?.start()
  }
  if (previewMode.value === 'stereo') {
    startStereoPreviewLoop()
  }
  if (xrSession) startControllerLoop()
  drawHud()
}

async function bootstrap(input?: unknown) {
  try {
    const seedPayload = normalizePayload(input ?? getInitialPayload())
    const payload = await mergeStandaloneCatalog(seedPayload)
    activePayload.value = payload
    if (payload.previewMode) previewMode.value = payload.previewMode
    else if (previewMode.value === 'desktop' && (payload.ply || payload.modelUrl || (payload.modelList?.length ?? 0) <= 1)) {
      previewMode.value = 'webxr'
    }
    authSession.value = payload.authSession || null
    modelList.value = normalizeModelPayloadList(payload)
    markers.value = payload.markers || []
    searchResults.value = payload.searchResults || []
    selectedSearchResultId.value = searchResults.value[0]?.id || ''
    selectedMarkerId.value = !selectedSearchResultId.value ? markers.value[0]?.id || '' : ''
    activeModelId.value = payload.activeModelId || modelList.value[0]?.id || ''
    activeSearchQuery.value = ''
    setLoadState('config', '准备读取模型配置', 0.04)
    const initialModel = resolveInitialModel(payload, modelList.value)
    if (initialModel) {
      activeModelId.value = initialModel.id
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

function applySceneScaleMode() {
  const target = getSceneManipulationTarget()
  if (!target) return
  if (sceneScaleMode.value === 'diorama') {
    target.scale.multiplyScalar(0.22)
    target.position.y += 0.8
  }
  userScale.value = getWorldRootScale()
  getRuntimeViewer()?.forceRenderNextFrame?.()
}

function setSceneScaleMode(mode: SceneScaleMode) {
  if (sceneScaleMode.value === mode) return
  sceneScaleMode.value = mode
  const current = modelList.value[activeModelIndex] || modelList.value[0]
  if (current) void loadModel(current, { preserveState: true })
}

function setInteractionMode(mode: InteractionMode) {
  interactionMode.value = mode
  if (mode === 'inspect') {
    setSceneScaleMode('diorama')
  }
}

function applyQualityPreset() {
  const runtime = getRuntimeViewer()
  if (!runtime?.renderer) return
  const ratioMap: Record<QualityPreset, number> = {
    ultra: 1,
    high: 0.88,
    balanced: 0.72,
    performance: 0.58,
    potato: 0.44,
  }
  runtime.renderer.setPixelRatio(ratioMap[qualityPreset.value])
  runtime.forceRenderNextFrame?.()
}

function setQualityPreset(preset: QualityPreset) {
  qualityPreset.value = preset
  applyQualityPreset()
}

function moveRig(strafe: number, forward: number, dt = 1 / 90, vertical = 0) {
  const runtime = getRuntimeViewer()
  if (!runtime?.camera || !xrRig) return false
  if (interactionMode.value === 'inspect') return false
  if (Math.abs(strafe) <= 0.01 && Math.abs(forward) <= 0.01 && Math.abs(vertical) <= 0.01) return false
  const direction = new THREE.Vector3()
  runtime.camera.getWorldDirection(direction)
  direction.y = 0
  direction.normalize()
  const right = new THREE.Vector3().crossVectors(direction, new THREE.Vector3(0, 1, 0)).normalize()
  const delta = new THREE.Vector3()
  delta.addScaledVector(direction, -forward * moveSpeed.value * dt)
  delta.addScaledVector(right, strafe * moveSpeed.value * dt)
  if (!floorLockEnabled.value) delta.y += vertical * 0.85 * dt
  if (isVrPresenting.value) {
    applySceneDelta(delta.clone().multiplyScalar(-1))
  } else {
    xrRig.position.add(delta)
  }
  runtime.forceRenderNextFrame?.()
  return true
}

function turnRig(turn: number, dt = 1 / 90, nowMs = performance.now()) {
  if (!xrRig || Math.abs(turn) <= 0.18) return false
  let angle = -turn * 1.65 * dt
  if (turnMode.value === 'snap') {
    if (nowMs - lastSnapTurnTime < 360) return false
    angle = -Math.sign(turn) * THREE.MathUtils.degToRad(snapTurnAngle.value)
    lastSnapTurnTime = nowMs
  }
  if (isVrPresenting.value) {
    rotateSceneAroundUser(angle)
  } else {
    xrRig.rotateY(angle)
  }
  getRuntimeViewer()?.forceRenderNextFrame?.()
  return true
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
  if (event.key === 'x' || event.key === 'X') setInteractionMode(interactionMode.value === 'explore' ? 'inspect' : 'explore')
  if (event.key === 'z' || event.key === 'Z') setSceneScaleMode(sceneScaleMode.value === 'room' ? 'diorama' : 'room')
  if (event.key === 'n' || event.key === 'N') nextNavigationPoint()
  if (event.key === 'c' || event.key === 'C') {
    clippingEnabled.value = !clippingEnabled.value
    updateClippingPlaneHelper()
  }
  if (event.key === 'v' || event.key === 'V') measurementEnabled.value = !measurementEnabled.value
  if (event.key === 'Enter') handlePrimarySelect()
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
  installXrSessionListeners()
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
  window.loadViewerPayload = (input: unknown) => {
    void bootstrap(input)
  }
  window.loadModelFromFlutter = (input: unknown) => {
    void bootstrap(input)
  }
  window.setViewerTheme = () => {
    drawHud()
  }
  window.setThemeFromFlutter = (theme: string) => {
    if (theme === 'dark' || theme === 'light') drawHud()
  }
  window.setViewerModelList = (list: unknown, currentId?: unknown) => {
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
  window.setModelListForTimePeeling = window.setViewerModelList
  window.setViewerSession = (session: unknown) => {
    authSession.value = session && typeof session === 'object' ? normalizePayload({
      ...activePayload.value,
      authSession: session,
    }).authSession || null : null
    drawHud()
  }
  window.setViewerSearchResults = (results: unknown) => {
    searchResults.value = normalizePayload({
      ...activePayload.value,
      searchResults: results,
    }).searchResults || []
    selectedSearchResultId.value = searchResults.value[0]?.id || ''
    rebuildSpatialHelpers()
    drawHud()
  }
  window.setViewerMarkers = (nextMarkers: unknown) => {
    markers.value = normalizePayload({
      ...activePayload.value,
      markers: nextMarkers,
    }).markers || []
    rebuildSpatialHelpers()
    drawHud()
  }
  window.setViewerQuery = (query: string) => {
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
  delete window.loadViewerPayload
  delete window.setViewerTheme
  delete window.setViewerModelList
  delete window.setViewerSession
  delete window.setViewerSearchResults
  delete window.setViewerMarkers
  delete window.setViewerQuery
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
          <dd>{{ interactionMode }}</dd>
        </div>
        <div>
          <dt>FPS</dt>
          <dd>{{ fps }}</dd>
        </div>
        <div>
          <dt>Scale</dt>
          <dd>{{ sceneScaleMode }}</dd>
        </div>
        <div>
          <dt>Quality</dt>
          <dd>{{ qualityLabel }}</dd>
        </div>
      </dl>

      <dl class="debug-list">
        <div>
          <dt>Model</dt>
          <dd>{{ modelCountLabel }} · {{ activeModelUrl || '-' }}</dd>
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
          <dd>{{ authLabel }}</dd>
        </div>
      </dl>

      <div class="mode-row">
        <button type="button" :class="{ active: previewMode === 'desktop' }" @click="selectMode('desktop')">Desktop</button>
        <button type="button" :class="{ active: previewMode === 'stereo' }" @click="selectMode('stereo')">Stereo</button>
        <button type="button" :class="{ active: previewMode === 'webxr' }" @click="selectMode('webxr')">WebXR</button>
      </div>

      <div class="mode-row">
        <button type="button" :class="{ active: interactionMode === 'explore' }" @click="setInteractionMode('explore')">Explore</button>
        <button type="button" :class="{ active: interactionMode === 'inspect' }" @click="setInteractionMode('inspect')">Inspect</button>
        <button type="button" :class="{ active: sceneScaleMode === 'diorama' }" @click="setSceneScaleMode(sceneScaleMode === 'room' ? 'diorama' : 'room')">沙盘</button>
      </div>

      <div class="button-row">
        <button type="button" @click="adjustScale(-0.1)">缩小</button>
        <button type="button" @click="resetView">重置</button>
        <button type="button" @click="adjustScale(0.1)">放大</button>
      </div>

      <div class="button-row">
        <button type="button" :class="{ active: turnMode === 'snap' }" @click="turnMode = turnMode === 'snap' ? 'smooth' : 'snap'">Snap</button>
        <button type="button" :class="{ active: floorLockEnabled }" @click="floorLockEnabled = !floorLockEnabled">锁地面</button>
        <button type="button" :class="{ active: vignetteEnabled }" @click="vignetteEnabled = !vignetteEnabled">黑边</button>
      </div>

      <div class="control-grid">
        <label>
          <span>移动速度 {{ moveSpeed.toFixed(2) }}</span>
          <input v-model.number="moveSpeed" type="range" min="0.35" max="2.2" step="0.05" />
        </label>
        <label>
          <span>转向角度 {{ snapTurnAngle }}°</span>
          <input v-model.number="snapTurnAngle" type="range" min="15" max="45" step="15" />
        </label>
      </div>

      <div class="quality-row">
        <button v-for="preset in qualityPresets" :key="preset" type="button" :class="{ active: qualityPreset === preset }" @click="setQualityPreset(preset)">
          {{ preset }}
        </button>
      </div>

      <div class="button-row xr-row">
        <button type="button" @click="enterVrSession">进入 VR</button>
        <button type="button" @click="exitVrSession">退出 VR</button>
        <button type="button" @click="isMenuOpen = !isMenuOpen">面板</button>
      </div>

      <div class="tool-panel">
        <div class="button-row">
          <button type="button" :class="{ active: clippingEnabled }" @click="clippingEnabled = !clippingEnabled; updateClippingPlaneHelper()">剖切</button>
          <button type="button" :class="{ active: measurementEnabled }" @click="measurementEnabled = !measurementEnabled">测量</button>
          <button type="button" @click="clearMeasurement">清除</button>
        </div>
        <label>
          <span>剖切距离 {{ clippingDistance.toFixed(1) }}m</span>
          <input v-model.number="clippingDistance" type="range" min="0.8" max="8" step="0.1" @input="updateClippingPlaneHelper" />
        </label>
        <p class="hint">测量距离：{{ measurementDistanceLabel }}；Enter / VR Trigger 记录测量点。</p>
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
            <span>{{ item.score != null ? `${Math.round(item.score * 100)}% · ` : '' }}{{ item.description || item.markerId || '-' }}</span>
          </button>
        </div>
      </div>

      <div v-if="activeEvidence" class="evidence-card">
        <p class="panel-label">Evidence</p>
        <h2>{{ activeEvidence.label }}</h2>
        <p>{{ activeEvidence.description || '暂无证据描述' }}</p>
        <dl>
          <div>
            <dt>Score</dt>
            <dd>{{ activeEvidence.score != null ? `${Math.round(activeEvidence.score * 100)}%` : '-' }}</dd>
          </div>
          <div>
            <dt>Frame</dt>
            <dd>{{ activeEvidence.imageId || '-' }}</dd>
          </div>
          <div>
            <dt>Time</dt>
            <dd>{{ activeEvidence.createdAt || '-' }}</dd>
          </div>
        </dl>
        <p v-if="activeEvidence.tags?.length" class="tag-line">{{ activeEvidence.tags.join('、') }}</p>
        <p class="hint">{{ lastDirectionHint }}</p>
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

      <div class="search-panel">
        <p class="panel-label">Time / Navigation</p>
        <div class="search-results">
          <button
            v-for="item in navigationPoints"
            :key="item.id"
            type="button"
            :class="{ active: selectedNavPointId === item.id }"
            @click="selectNavigationPoint(item.id)"
          >
            <strong>{{ item.label }}</strong>
            <span>{{ item.kind || 'navigation' }} · {{ item.description || item.position.join(', ') }}</span>
          </button>
        </div>
        <button type="button" @click="nextNavigationPoint">下一个导航点</button>
      </div>

      <div class="summary-panel">
        <p class="panel-label">Scene Summary</p>
        <p>{{ modelSummaryText }}</p>
      </div>

      <details class="debug-panel">
        <summary>SteamVR Input Debug</summary>
        <pre>{{ controllerDebug }}</pre>
      </details>

      <p class="hint">1/2/3 切换 Desktop/Stereo/WebXR，WASD 漫游，M 面板，X Explore/Inspect，Z 沙盘/1:1，N 导航点，C 剖切，V 测量，Enter 选择/测量。</p>
    </section>
  </main>
</template>
