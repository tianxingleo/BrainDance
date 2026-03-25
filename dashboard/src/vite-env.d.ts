/// <reference types="vite/client" />

declare module '@iconify/vue' {
  import type { DefineComponent } from 'vue'

  export const Icon: DefineComponent<{
    icon: string
  }>
}
