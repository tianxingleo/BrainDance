export type PreviewMode = 'desktop' | 'stereo' | 'webxr'

export function getPreviewMode(): PreviewMode {
  const value = new URLSearchParams(window.location.search).get('preview')
  if (value === 'stereo') return 'stereo'
  if (value === 'webxr') return 'webxr'
  return 'desktop'
}

export function switchPreviewMode(mode: PreviewMode) {
  const url = new URL(window.location.href)
  url.searchParams.set('preview', mode)
  window.location.href = url.toString()
}
