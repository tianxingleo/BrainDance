import type { BrainDanceViewerPayload } from '../types/viewer'

const fallbackPayload: BrainDanceViewerPayload = {
  ply: './models/scene_auto_sync_raw.ply',
  poses: './models/webgl_poses.json',
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

  return {
    ply: typeof ply === 'string' && ply.trim() ? ply : fallbackPayload.ply,
    poses: value.poses ? String(value.poses) : fallbackPayload.poses,
    matrix: Array.isArray(value.matrix) ? value.matrix.map(Number).filter(Number.isFinite) : undefined,
    imageId: value.imageId ? String(value.imageId) : undefined,
    sceneId: value.sceneId ? String(value.sceneId) : undefined,
  }
}

export function getInitialPayload(): BrainDanceViewerPayload {
  return parsePayloadFromUrl() || { ...fallbackPayload }
}

export function deriveVrConfigUrl(payload: BrainDanceViewerPayload): string {
  if (!payload.poses) return './models/vr_config.json'
  const nextUrl = payload.poses.replace(/webgl_poses(?:_with_tags)?\.json(?:\?.*)?$/i, 'vr_config.json')
  return nextUrl === payload.poses ? './models/vr_config.json' : nextUrl
}
