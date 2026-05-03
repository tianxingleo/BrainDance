import { DEFAULT_MARKER_TARGET, DEFAULT_MODEL_URL } from './markerTargets'
import type { ArViewerParams } from '../types/ar'

const parseNumber = (value: string | null, fallback: number) => {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

const parseText = (value: string | null, fallback: string) => {
  const trimmed = value?.trim()
  return trimmed && trimmed.length > 0 ? trimmed : fallback
}

export function parseArParams(): ArViewerParams {
  const params = new URLSearchParams(window.location.search)

  return {
    mode: (params.get('mode') === 'marker-ar' ? 'marker-ar' : 'viewer'),
    modelUrl: parseText(params.get('model'), DEFAULT_MODEL_URL),
    targetUrl: parseText(params.get('target'), DEFAULT_MARKER_TARGET),
    camera: parseText(params.get('camera'), 'main-back'),
    pixelRatio: parseNumber(params.get('pixelRatio'), 1),
    filterMinCF: parseNumber(params.get('filterMinCF'), 0.0008),
    filterBeta: parseNumber(params.get('filterBeta'), 400),
    warmupTolerance: parseNumber(params.get('warmupTolerance'), 8),
    missTolerance: parseNumber(params.get('missTolerance'), 12),
    scale: parseNumber(params.get('scale'), 0.5),
    rotation: [
      parseNumber(params.get('rx'), -Math.PI / 2),
      parseNumber(params.get('ry'), 0),
      parseNumber(params.get('rz'), 0),
    ],
    offset: [
      parseNumber(params.get('ox'), 0),
      parseNumber(params.get('oy'), 0.04),
      parseNumber(params.get('oz'), 0),
    ],
  }
}

