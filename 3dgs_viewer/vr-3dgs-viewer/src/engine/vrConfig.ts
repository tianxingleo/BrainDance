import type { BrainDanceVrConfig } from '../types/viewer'

export const defaultVrConfig: BrainDanceVrConfig = {
  worldScale: 1.0,
  worldPosition: [0, 0, -2.2],
  worldRotationY: 0,
  userHeight: 1.6,
  startDistance: 2.2,
  near: 0.01,
  far: 2000,
  preferCompressedModel: true,
  mode: 'auto',
  comfortStart: 'outside',
}

function toNumberTuple(value: unknown, fallback: [number, number, number]): [number, number, number] {
  if (!Array.isArray(value) || value.length < 3) return fallback

  const tuple = value.slice(0, 3).map(Number)
  if (tuple.some((item) => !Number.isFinite(item))) return fallback
  return tuple as [number, number, number]
}

export async function loadVrConfig(url = './models/vr_config.json'): Promise<BrainDanceVrConfig> {
  try {
    const response = await fetch(url, { cache: 'no-cache' })
    if (!response.ok) return { ...defaultVrConfig }

    const input = (await response.json()) as Partial<BrainDanceVrConfig>
    return {
      ...defaultVrConfig,
      ...input,
      worldPosition: toNumberTuple(input.worldPosition, defaultVrConfig.worldPosition),
      worldScale: Number.isFinite(Number(input.worldScale)) ? Number(input.worldScale) : defaultVrConfig.worldScale,
      worldRotationY: Number.isFinite(Number(input.worldRotationY))
        ? Number(input.worldRotationY)
        : defaultVrConfig.worldRotationY,
      userHeight: Number.isFinite(Number(input.userHeight)) ? Number(input.userHeight) : defaultVrConfig.userHeight,
      startDistance: Number.isFinite(Number(input.startDistance))
        ? Number(input.startDistance)
        : defaultVrConfig.startDistance,
      near: Number.isFinite(Number(input.near)) ? Number(input.near) : defaultVrConfig.near,
      far: Number.isFinite(Number(input.far)) ? Number(input.far) : defaultVrConfig.far,
      preferCompressedModel: input.preferCompressedModel ?? defaultVrConfig.preferCompressedModel,
      mode: normalizeMode(input.mode, defaultVrConfig.mode),
      comfortStart: normalizeComfortStart(input.comfortStart, defaultVrConfig.comfortStart),
      navigationPoints: normalizeNavigationPoints(input.navigationPoints),
      summary: normalizeSummary(input.summary),
    }
  } catch {
    return { ...defaultVrConfig }
  }
}

function normalizeMode(value: unknown, fallback?: BrainDanceVrConfig['mode']) {
  return value === 'room' || value === 'object' || value === 'auto' ? value : fallback
}

function normalizeComfortStart(value: unknown, fallback?: BrainDanceVrConfig['comfortStart']) {
  return value === 'outside' || value === 'inside' || value === 'safe-point' ? value : fallback
}

function normalizeNavigationPoints(value: unknown): BrainDanceVrConfig['navigationPoints'] {
  if (!Array.isArray(value)) return undefined
  const points = value
    .map((item, index) => {
      if (!item || typeof item !== 'object') return null
      const entry = item as Record<string, unknown>
      if (!Array.isArray(entry.position) || entry.position.length < 3) return null
      const position = entry.position.slice(0, 3).map(Number)
      if (position.some((number) => !Number.isFinite(number))) return null
      return {
        id: String(entry.id || `nav-${index}`),
        label: String(entry.label || entry.name || `导航点 ${index + 1}`),
        position: position as [number, number, number],
        kind: entry.kind === 'entry' || entry.kind === 'center' || entry.kind === 'best-view' || entry.kind === 'search-hit' || entry.kind === 'tour'
          ? entry.kind
          : undefined,
        matrix: Array.isArray(entry.matrix) ? entry.matrix.map(Number).filter(Number.isFinite) : undefined,
        description: entry.description ? String(entry.description) : undefined,
      }
    })
    .filter(Boolean) as NonNullable<BrainDanceVrConfig['navigationPoints']>
  return points.length > 0 ? points : undefined
}

function normalizeSummary(value: unknown): BrainDanceVrConfig['summary'] {
  if (!value || typeof value !== 'object') return undefined
  const entry = value as Record<string, unknown>
  return {
    sceneType: entry.sceneType ? String(entry.sceneType) : undefined,
    objects: Array.isArray(entry.objects) ? entry.objects.map(String).filter(Boolean) : undefined,
    searchableObjects: Array.isArray(entry.searchableObjects) ? entry.searchableObjects.map(String).filter(Boolean) : undefined,
    recommendedPoints: Array.isArray(entry.recommendedPoints) ? entry.recommendedPoints.map(String).filter(Boolean) : undefined,
  }
}
