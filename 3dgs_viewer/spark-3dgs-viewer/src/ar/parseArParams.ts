import { DEFAULT_MARKER_TARGET, DEFAULT_MODEL_URL } from './markerTargets'
import type { ArViewerParams } from '../types/ar'

const parseNumber = (value: string | null, fallback: number) => {
  if (value == null || value.trim().length === 0) return fallback
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

const parsePositiveNumber = (value: string | null, fallback: number) => {
  const parsed = parseNumber(value, fallback)
  return parsed > 0 ? parsed : fallback
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
    cameraIndex: Math.max(0, Math.floor(parseNumber(params.get('cameraIndex'), 0))),
    pixelRatio: parsePositiveNumber(params.get('pixelRatio'), 1),
    filterMinCF: parsePositiveNumber(params.get('filterMinCF'), 0.0008),
    filterBeta: parsePositiveNumber(params.get('filterBeta'), 400),
    warmupTolerance: parsePositiveNumber(params.get('warmupTolerance'), 8),
    missTolerance: parsePositiveNumber(params.get('missTolerance'), 12),
    scale: parsePositiveNumber(params.get('scale'), 0.5),
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

