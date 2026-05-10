export type PreviewMode = 'desktop' | 'stereo' | 'webxr'

export function getPreviewMode(): PreviewMode {
  const params = new URLSearchParams(window.location.search)
  const value = params.get('preview') || getPayloadPreviewMode(params.get('payload'))
  if (value === 'stereo') return 'stereo'
  if (value === 'webxr') return 'webxr'
  return 'desktop'
}

export function switchPreviewMode(mode: PreviewMode) {
  const url = new URL(window.location.href)
  url.searchParams.set('preview', mode)
  window.location.href = url.toString()
}

function getPayloadPreviewMode(rawPayload: string | null): unknown {
  if (!rawPayload) return undefined
  try {
    const payload = JSON.parse(decodeURIComponent(rawPayload)) as Record<string, unknown>
    return payload.previewMode
  } catch {
    return undefined
  }
}
