export interface BrainDanceViewerPayload {
  ply: string
  poses?: string
  matrix?: number[]
  imageId?: string
  sceneId?: string
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
  update?: () => void
  render?: () => void
  forceRenderNextFrame?: () => void
}

export interface VrLoadResult {
  payload: BrainDanceViewerPayload
  modelUrl: string
}

declare global {
  interface Window {
    loadModelFromFlutter?: (input: unknown) => void
  }
}
