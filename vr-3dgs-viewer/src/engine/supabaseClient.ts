import { createClient, type SupabaseClient } from '@supabase/supabase-js'

const configuredSupabaseUrl = normalizeSupabaseUrl(
  import.meta.env.VITE_BD_SUPABASE_URL || import.meta.env.VITE_SUPABASE_URL || '',
)
const supabaseUrlFallbacks = parseUrlFallbacks(
  import.meta.env.VITE_BD_SUPABASE_URL_FALLBACKS || import.meta.env.VITE_SUPABASE_URL_FALLBACKS || '',
)
const supabaseUrl = resolveSupabaseUrl([
  supabaseUrlFallbacks,
  configuredSupabaseUrl,
].flat())
const supabaseAnonKey = import.meta.env.VITE_BD_SUPABASE_ANON_KEY?.trim()
  || import.meta.env.VITE_SUPABASE_ANON_KEY?.trim()
  || ''

export const viewerSupabaseClient: SupabaseClient | null =
  supabaseUrl && supabaseAnonKey
    ? createClient(supabaseUrl, supabaseAnonKey, {
        auth: {
          persistSession: true,
          autoRefreshToken: true,
          detectSessionInUrl: false,
        },
      })
    : null

export const viewerSupabaseEnabled = Boolean(viewerSupabaseClient)
export const viewerSupabaseProxyEnabled = false
export const viewerSupabaseTargetUrl = supabaseUrl
export const viewerSupabaseConfiguredUrl = configuredSupabaseUrl
export const viewerSupabaseUrlFallbacks = supabaseUrlFallbacks
export const viewerSupabaseConfigError = ''

function normalizeSupabaseUrl(rawUrl: string) {
  const trimmed = rawUrl.trim()
  if (!trimmed) return ''
  return trimmed.endsWith('/') ? trimmed.slice(0, -1) : trimmed
}

function parseUrlFallbacks(rawValue: string) {
  return rawValue
    .split(',')
    .map(normalizeSupabaseUrl)
    .filter(Boolean)
}

function resolveSupabaseUrl(candidates: string[]) {
  const deduplicated = candidates.filter((candidate, index) => candidate && candidates.indexOf(candidate) === index)
  if (typeof window === 'undefined') return deduplicated[0] || ''
  const pageProtocol = window.location.protocol
  const pageHostname = window.location.hostname
  const pagePort = window.location.port

  const sameHostLocal = deduplicated.find((candidate) => {
    try {
      const url = new URL(candidate)
      return isLikelyLocalSupabaseUrl(url)
        && url.hostname === pageHostname
        && (pageProtocol !== 'https:' || url.protocol === 'https:' || import.meta.env.DEV)
    } catch {
      return false
    }
  })
  if (sameHostLocal) return sameHostLocal

  const loopbackLocal = deduplicated.find((candidate) => {
    try {
      const url = new URL(candidate)
      return isLoopbackHost(url.hostname) && url.port
    } catch {
      return false
    }
  })
  if (loopbackLocal && isLikelyLanHost(pageHostname)) {
    const url = new URL(loopbackLocal)
    url.hostname = pageHostname
    if (pageProtocol === 'https:') url.protocol = 'https:'
    if (pagePort && pagePort !== '5174' && pagePort !== '4174') url.port = pagePort
    return normalizeSupabaseUrl(url.toString())
  }

  return deduplicated[0] || ''
}

function getUrlHost(value: string) {
  try {
    return value ? new URL(value).host : ''
  } catch {
    return ''
  }
}

function isLikelyLocalSupabaseUrl(url: URL) {
  return isLoopbackHost(url.hostname)
    || url.hostname.startsWith('192.168.')
    || url.hostname.startsWith('10.')
    || /^172\.(1[6-9]|2\d|3[01])\./.test(url.hostname)
}

function isLoopbackHost(hostname: string) {
  return hostname === '127.0.0.1' || hostname === 'localhost'
}

function isLikelyLanHost(hostname: string) {
  return hostname.startsWith('192.168.')
    || hostname.startsWith('10.')
    || /^172\.(1[6-9]|2\d|3[01])\./.test(hostname)
}
