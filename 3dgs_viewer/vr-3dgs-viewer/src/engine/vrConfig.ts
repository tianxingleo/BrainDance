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
    }
  } catch {
    return { ...defaultVrConfig }
  }
}
