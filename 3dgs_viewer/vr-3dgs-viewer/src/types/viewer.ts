export interface BrainDanceViewerPayload {
  ply: string
  modelUrl?: string
  poses?: string
  posesUrl?: string
  matrix?: number[]
  imageId?: string
  sceneId?: string
  modelList?: BrainDanceRecallModel[]
  markers?: BrainDanceRecallMarker[]
  searchResults?: BrainDanceRecallSearchResult[]
  authSession?: BrainDanceAuthSession
  previewMode?: import('../engine/previewMode').PreviewMode
}

export interface BrainDanceRecallModel {
  id: string
  sceneId?: string
  name?: string
  displayName?: string
  ply: string
  modelUrl?: string
  poses?: string
  posesUrl?: string
  previewImage?: string
  previewImg?: string
  description?: string
  tags?: string[]
  createdAt?: string
}

export interface BrainDanceRecallMarker {
  id: string
  label: string
  position?: [number, number, number]
  matrix?: number[]
  color?: string
}

export interface BrainDanceRecallSearchResult {
  id: string
  label: string
  description?: string
  imageId?: string
  matrix?: number[]
  markerId?: string
  score?: number
}

export interface BrainDanceAuthSession {
  userId?: string
  email?: string
  displayName?: string
  accessToken?: string
  refreshToken?: string
  expiresAt?: string
  code?: string
  status?: string
}

export interface BrainDanceVrConfig {
  worldScale: number
  worldPosition: [number, number, number]
  worldRotationY: number
  userHeight: number
  startDistance: number
  near: number
  far: number
  preferCompressedModel: boolean
}

export interface RuntimeGaussianViewer {
  camera?: import('three').PerspectiveCamera
  renderer?: import('three').WebGLRenderer
  splatMesh?: import('three').Object3D
  threeScene?: import('three').Scene
  update?: () => void
  render?: () => void
  start?: () => void
  stop?: () => void
  dispose?: () => void
  controls?: {
    update: () => void
  }
  sceneHelper?: {
    focusMarker?: import('three').Object3D
    controlPlane?: import('three').Object3D
    getFocusMarkerOpacity?: () => number
  }
  webXRActive?: boolean
  showControlPlane?: boolean
  forceRenderNextFrame?: () => void
}

export interface VrLoadResult {
  payload: BrainDanceViewerPayload
  modelUrl: string
}

declare global {
  interface Window {
    loadModelFromFlutter?: (input: unknown) => void
    setModelListForTimePeeling?: (list: unknown, currentId?: unknown) => void
    setBrainDanceSession?: (session: unknown) => void
    setRecallSearchResults?: (results: unknown) => void
    setRecallQuery?: (query: string) => void
    setThemeFromFlutter?: (theme: string) => void
    setRecallMarkers?: (markers: unknown) => void
  }
}
