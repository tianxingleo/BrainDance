/// <reference types="vite/client" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<object, object, unknown>
  export default component
}

declare module 'mind-ar/dist/mindar-image-three.prod.js' {
  export const MindARThree: unknown
}

interface Window {
  setModelListForTimePeeling?: (...args: unknown[]) => void
  loadModelFromFlutter?: (...args: unknown[]) => void
}
