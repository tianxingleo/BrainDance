import { createClient, type SupabaseClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_BD_SUPABASE_URL?.trim() || ''
const supabaseAnonKey = import.meta.env.VITE_BD_SUPABASE_ANON_KEY?.trim() || ''
const useProxy = typeof window !== 'undefined'
  && window.location.protocol === 'https:'
  && supabaseUrl.startsWith('http://')
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
