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
const configuredProxyTarget = import.meta.env.VITE_BD_SUPABASE_PROXY_TARGET?.trim() || ''
const currentHost = typeof window !== 'undefined' ? window.location.host : ''
const supabaseHost = getUrlHost(supabaseUrl)
const useProxy = typeof window !== 'undefined'
  && window.location.protocol === 'https:'
  && supabaseUrl.startsWith('http://')
  && (configuredProxyTarget || import.meta.env.DEV)
  && currentHost !== supabaseHost
const proxiedSupabaseUrl = useProxy
  ? `${window.location.origin}/supabase-proxy`
  : supabaseUrl

export const viewerSupabaseClient: SupabaseClient | null =
  proxiedSupabaseUrl && supabaseAnonKey
    ? createClient(proxiedSupabaseUrl, supabaseAnonKey, {
        auth: {
          persistSession: true,
          autoRefreshToken: true,
          detectSessionInUrl: false,
        },
      })
    : null

export const viewerSupabaseEnabled = Boolean(viewerSupabaseClient)
export const viewerSupabaseProxyEnabled = useProxy
export const viewerSupabaseTargetUrl = supabaseUrl
export const viewerSupabaseConfiguredUrl = configuredSupabaseUrl
export const viewerSupabaseUrlFallbacks = supabaseUrlFallbacks
export const viewerSupabaseConfigError = currentHost && supabaseHost && currentHost === supabaseHost
  ? `VR Viewer 当前运行在 Supabase 端口 ${currentHost}，请改用 Vite 地址访问，例如 https://<本机IP>:5174。`
  : ''

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
  return sameHostLocal || deduplicated[0] || ''
}

function getUrlHost(value: string) {
  try {
    return value ? new URL(value).host : ''
  } catch {
    return ''
  }
}

function isLikelyLocalSupabaseUrl(url: URL) {
  return url.hostname === '127.0.0.1'
    || url.hostname === 'localhost'
    || url.hostname.startsWith('192.168.')
    || url.hostname.startsWith('10.')
    || /^172\.(1[6-9]|2\d|3[01])\./.test(url.hostname)
}
