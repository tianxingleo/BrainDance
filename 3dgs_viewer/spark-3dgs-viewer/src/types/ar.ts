export type ArMode = 'viewer' | 'marker-ar'

export type ArTransform = {
  scale: number
  rotation: [number, number, number]
  offset: [number, number, number]
}

export type ArViewerParams = ArTransform & {
  mode: ArMode
  modelUrl: string
  targetUrl: string
  camera: string
  cameraIndex: number
  pixelRatio: number
  filterMinCF: number
  filterBeta: number
  warmupTolerance: number
  missTolerance: number
}

export type ArStatus =
  | '准备启动 AR...'
  | '模型加载中...'
  | '请允许摄像头权限，并将纸板放入画面'
  | '已识别纸板'
  | '跟踪丢失，请重新对准纸板'
  | 'AR 启动失败，请检查浏览器摄像头权限'

