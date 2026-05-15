import * as THREE from 'three'
import type {
  BrainDanceAuthSession,
  BrainDanceRecallMarker,
  BrainDanceRecallModel,
  BrainDanceRecallSearchResult,
} from '../types/viewer'

const sourceToViewer = new THREE.Matrix4().makeScale(1, 1, -1)

export function normalizeNumericArray(value: unknown): number[] | undefined {
  if (!Array.isArray(value) || value.length === 0) return undefined
  const numbers = value.map(Number).filter(Number.isFinite)
  return numbers.length > 0 ? numbers : undefined
}

export function normalizeMatrixForViewer(value: unknown): number[] | undefined {
  const input = normalizeNumericArray(value)
  if (!input || input.length !== 16) return input

  const matrix = new THREE.Matrix4().fromArray(input)
  const converted = sourceToViewer.clone().multiply(matrix).multiply(sourceToViewer)
  return converted.toArray()
}

export function normalizeVector3ForViewer(
  value: unknown,
): [number, number, number] | undefined {
  if (!Array.isArray(value) || value.length < 3) return undefined
  const tuple = value.slice(0, 3).map(Number)
  if (tuple.some((item) => !Number.isFinite(item))) return undefined
  const [x, y, z] = tuple as [number, number, number]
  return [x, y, -z]
}

export function decomposeMatrix(matrixValues: number[]) {
  const matrix = new THREE.Matrix4().fromArray(matrixValues)
  const position = new THREE.Vector3()
  const quaternion = new THREE.Quaternion()
  const scale = new THREE.Vector3()
  matrix.decompose(position, quaternion, scale)
  return { position, quaternion, scale }
}

export function normalizeModelList(value: unknown): BrainDanceRecallModel[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item): BrainDanceRecallModel | null => {
      if (!item || typeof item !== 'object') return null
      const entry = item as Record<string, unknown>
      const id = String(entry.id || entry.modelId || entry.sceneId || '').trim()
      const ply = String(entry.ply || entry.plyUrl || entry.modelUrl || entry.ply_path || '').trim()
      if (!id || !ply) return null
      const displayName = entry.displayName ? String(entry.displayName) : undefined
      const name = entry.name ? String(entry.name) : displayName
      return {
        id,
        sceneId: entry.sceneId ? String(entry.sceneId) : undefined,
        name,
        displayName,
        ply,
        modelUrl: entry.modelUrl ? String(entry.modelUrl) : ply,
        poses: entry.poses ? String(entry.poses) : undefined,
        posesUrl: entry.posesUrl ? String(entry.posesUrl) : undefined,
        previewImage: entry.previewImage ? String(entry.previewImage) : undefined,
        previewImg: entry.previewImg ? String(entry.previewImg) : undefined,
        description: entry.description ? String(entry.description) : undefined,
        tags: Array.isArray(entry.tags) ? entry.tags.map(String).filter(Boolean) : undefined,
        createdAt: entry.createdAt ? String(entry.createdAt) : undefined,
      }
    })
    .filter((item): item is BrainDanceRecallModel => Boolean(item))
}

export function normalizeMarkers(value: unknown): BrainDanceRecallMarker[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item): BrainDanceRecallMarker | null => {
      if (!item || typeof item !== 'object') return null
      const entry = item as Record<string, unknown>
      const id = String(entry.id || entry.markerId || '').trim()
      const label = String(entry.label || entry.name || id).trim()
      const position = normalizeVector3ForViewer(entry.position)
      if (!id && !label) return null
      return {
        id: id || label,
        label: label || id,
        position,
        matrix: normalizeMatrixForViewer(entry.matrix),
        color: entry.color ? String(entry.color) : undefined,
        description: entry.description ? String(entry.description) : undefined,
        imageId: entry.imageId ? String(entry.imageId) : undefined,
        score: Number.isFinite(Number(entry.score)) ? Number(entry.score) : undefined,
        tags: Array.isArray(entry.tags) ? entry.tags.map(String).filter(Boolean) : undefined,
        createdAt: entry.createdAt ? String(entry.createdAt) : undefined,
      }
    })
    .filter((item): item is BrainDanceRecallMarker => Boolean(item))
}

export function normalizeSearchResults(value: unknown): BrainDanceRecallSearchResult[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item): BrainDanceRecallSearchResult | null => {
      if (!item || typeof item !== 'object') return null
      const entry = item as Record<string, unknown>
      const id = String(entry.id || entry.resultId || entry.markerId || '').trim()
      const label = String(entry.label || entry.title || id).trim()
      if (!id && !label) return null
      const scoreValue = entry.score == null ? undefined : Number(entry.score)
      return {
        id: id || label,
        label: label || id,
        description: entry.description ? String(entry.description) : undefined,
        imageId: entry.imageId ? String(entry.imageId) : undefined,
        matrix: normalizeMatrixForViewer(entry.matrix),
        position: normalizeVector3ForViewer(entry.position),
        markerId: entry.markerId ? String(entry.markerId) : undefined,
        score: Number.isFinite(scoreValue) ? scoreValue : undefined,
        tags: Array.isArray(entry.tags) ? entry.tags.map(String).filter(Boolean) : undefined,
        createdAt: entry.createdAt ? String(entry.createdAt) : undefined,
      }
    })
    .filter((item): item is BrainDanceRecallSearchResult => Boolean(item))
}

export function normalizeAuthSession(value: unknown): BrainDanceAuthSession | null {
  if (!value || typeof value !== 'object') return null
  const entry = value as Record<string, unknown>
  const session: BrainDanceAuthSession = {
    userId: entry.userId ? String(entry.userId) : undefined,
    email: entry.email ? String(entry.email) : undefined,
    displayName: entry.displayName ? String(entry.displayName) : undefined,
    expiresAt: entry.expiresAt ? String(entry.expiresAt) : undefined,
    code: entry.code ? String(entry.code) : undefined,
    status: entry.status ? String(entry.status) : undefined,
  }
  return Object.values(session).some(Boolean) ? session : null
}
