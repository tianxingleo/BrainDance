import type { BrainDanceAuthSession } from '../types/viewer'
import type { PreviewMode } from './previewMode'

export type ViewerClientState = {
  authSession?: BrainDanceAuthSession | null
  activeModelId?: string
  activeSearchQuery?: string
  previewMode?: PreviewMode
  hudView?: 'controls' | 'auth' | 'models' | 'search' | 'markers' | 'nav'
  interactionMode?: 'explore' | 'inspect'
  sceneScaleMode?: 'room' | 'diorama'
  turnMode?: 'snap' | 'smooth'
  qualityPreset?: 'ultra' | 'high' | 'balanced' | 'performance' | 'potato'
  clippingEnabled?: boolean
  clippingDistance?: number
  measurementEnabled?: boolean
  isMenuOpen?: boolean
}

const storageKey = 'braindance.vr.viewer.client-state'

export function loadViewerClientState(): ViewerClientState {
  if (typeof window === 'undefined') return {}
  try {
    const raw = window.localStorage.getItem(storageKey)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as ViewerClientState
    return typeof parsed === 'object' && parsed ? parsed : {}
  } catch {
    return {}
  }
}

export function saveViewerClientState(state: ViewerClientState) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(storageKey, JSON.stringify(state))
  } catch {
    // 本地存储失败时保持静默，VR 客户端不能因为持久化失败而中断主流程。
  }
}
