import { DEFAULT_MARKER_TARGET, DEFAULT_MODEL_URL } from './markerTargets'
import type { ArViewerParams } from '../types/ar'

const parseNumber = (value: string | null, fallback: number) => {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

export function parseArParams(): ArViewerParams {
  const params = new URLSearchParams(window.location.search)

  return {
    mode: (params.get('mode') === 'marker-ar' ? 'marker-ar' : 'viewer'),
    modelUrl: params.get('model') || DEFAULT_MODEL_URL,
    targetUrl: params.get('target') || DEFAULT_MARKER_TARGET,
    scale: parseNumber(params.get('scale'), 0.25),
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

