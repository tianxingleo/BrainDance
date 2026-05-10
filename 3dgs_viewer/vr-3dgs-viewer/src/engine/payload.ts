import type {
  BrainDanceViewerPayload,
} from '../types/viewer'
import {
  normalizeAuthSession,
  normalizeMarkers,
  normalizeModelList,
  normalizeSearchResults,
} from './bridge'
import type { PreviewMode } from './previewMode'

const defaultModelUrl = normalizeEnvString(
  import.meta.env.VITE_BD_DEFAULT_MODEL_URL,
  './models/point_cloud.splat',
)
const defaultPosesUrl = normalizeEnvString(
  import.meta.env.VITE_BD_DEFAULT_POSES_URL,
  './models/webgl_poses.json',
)
const defaultVrConfigUrl = normalizeEnvString(
  import.meta.env.VITE_BD_DEFAULT_VR_CONFIG_URL,
  './models/vr_config.json',
)
const defaultPreviewMode = normalizePreviewMode(import.meta.env.VITE_BD_DEFAULT_PREVIEW_MODE)

const fallbackPayload: BrainDanceViewerPayload = {
  ply: defaultModelUrl,
  modelUrl: defaultModelUrl,
  poses: defaultPosesUrl,
  posesUrl: defaultPosesUrl,
  previewMode: defaultPreviewMode,
}

export function parsePayloadFromUrl(): BrainDanceViewerPayload | null {
  const params = new URLSearchParams(window.location.search)
  const raw = params.get('payload')
  if (!raw) return null

  try {
    return normalizePayload(JSON.parse(decodeURIComponent(raw)))
  } catch (error) {
    console.error('[BrainDance VR] payload 解析失败:', error)
    return null
  }
}

export function normalizePayload(input: unknown): BrainDanceViewerPayload {
  if (typeof input === 'string') {
    return {
      ...fallbackPayload,
      ply: input,
    }
  }

  if (!input || typeof input !== 'object') {
    return { ...fallbackPayload }
  }

  const value = input as Record<string, unknown>
  const ply = value.ply || value.modelUrl || value.url
  const poses = value.poses || value.posesUrl
  const modelList = normalizeModelList(value.modelList || value.timePeelingModels)
  const markers = normalizeMarkers(value.markers || value.recallMarkers)
  const searchResults = normalizeSearchResults(value.searchResults || value.recallSearchResults)
  const authSession = normalizeAuthSession(value.authSession || value.session || value.userSession)

  return {
    ply: typeof ply === 'string' && ply.trim() ? ply : fallbackPayload.ply,
    modelUrl: typeof ply === 'string' && ply.trim() ? String(ply) : undefined,
    poses: poses ? String(poses) : fallbackPayload.poses,
    posesUrl: poses ? String(poses) : undefined,
    matrix: normalizeNumericArray(value.matrix),
    imageId: value.imageId ? String(value.imageId) : undefined,
    sceneId: value.sceneId ? String(value.sceneId) : undefined,
    activeModelId: value.activeModelId ? String(value.activeModelId) : undefined,
    modelList: modelList.length > 0 ? modelList : undefined,
    markers: markers.length > 0 ? markers : undefined,
    searchResults: searchResults.length > 0 ? searchResults : undefined,
    authSession: authSession || undefined,
    previewMode: normalizePreviewMode(value.previewMode),
  }
}

export function getInitialPayload(): BrainDanceViewerPayload {
  return parsePayloadFromUrl() || { ...fallbackPayload }
}

export function deriveVrConfigUrl(payload: BrainDanceViewerPayload): string {
  if (!payload.poses) return defaultVrConfigUrl
  const nextUrl = payload.poses.replace(/webgl_poses(?:_with_tags)?\.json(?:\?.*)?$/i, 'vr_config.json')
  return nextUrl === payload.poses ? defaultVrConfigUrl : nextUrl
}

export function resolveRelativeAssetUrl(assetUrl: string | undefined, baseUrl: string): string {
  const value = typeof assetUrl === 'string' ? assetUrl.trim() : ''
  if (!value) return value
  try {
    return new URL(value, baseUrl).toString()
  } catch {
    return value
  }
}

function normalizeEnvString(value: unknown, fallback: string): string {
  return typeof value === 'string' && value.trim() ? value.trim() : fallback
}

function normalizeNumericArray(value: unknown): number[] | undefined {
  if (!Array.isArray(value) || value.length === 0) return undefined
  const numbers = value.map(Number).filter(Number.isFinite)
  return numbers.length > 0 ? numbers : undefined
}

function normalizePreviewMode(value: unknown): PreviewMode | undefined {
  if (value === 'desktop' || value === 'stereo' || value === 'webxr') return value
  return undefined
}
