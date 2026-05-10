/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BD_DEFAULT_MODEL_URL?: string
  readonly VITE_BD_DEFAULT_POSES_URL?: string
  readonly VITE_BD_DEFAULT_VR_CONFIG_URL?: string
  readonly VITE_BD_DEFAULT_PREVIEW_MODE?: 'desktop' | 'stereo' | 'webxr'
  readonly VITE_BD_MODEL_CATALOG_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<object, object, unknown>
  export default component
}

declare module '@mkkellogg/gaussian-splats-3d' {
  export const WebXRMode: {
    None: number
    VR: number
    AR: number
  }

  export class Viewer {
    constructor(options: Record<string, unknown>)
    camera?: import('three').PerspectiveCamera
    renderer?: import('three').WebGLRenderer
    splatMesh?: import('three').Object3D
    threeScene?: import('three').Scene
    controls?: { update: () => void }
    sceneHelper?: {
      focusMarker?: import('three').Object3D
      controlPlane?: import('three').Object3D
      getFocusMarkerOpacity?: () => number
    }
    webXRActive?: boolean
    showControlPlane?: boolean
    addSplatScene(path: string, options?: Record<string, unknown>): Promise<unknown>
    start(): void
    stop(): void
    dispose(): void
    update(): void
    render(): void
    forceRenderNextFrame(): void
  }
}
