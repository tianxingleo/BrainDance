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

export interface VrLoadResult {
  payload: BrainDanceViewerPayload
  modelUrl: string
}

declare global {
  interface Window {
    loadModelFromFlutter?: (input: unknown) => void
  }
}
