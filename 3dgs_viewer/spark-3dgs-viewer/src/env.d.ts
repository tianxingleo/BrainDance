export {}

declare global {
  interface Window {
    BrainDanceChannel?: {
      postMessage?: (message: string) => void
    }
  }
}
