/// <reference types="vite/client" />

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
    renderer?: {
      xr?: {
        enabled: boolean
        isPresenting?: boolean
        addEventListener?: (type: string, listener: EventListenerOrEventListenerObject) => void
        removeEventListener?: (type: string, listener: EventListenerOrEventListenerObject) => void
      }
    }
    addSplatScene(path: string, options?: Record<string, unknown>): Promise<unknown>
    start(): void
    stop(): void
    dispose(): void
  }
}
