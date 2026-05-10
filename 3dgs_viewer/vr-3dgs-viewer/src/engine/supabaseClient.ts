import { createClient, type SupabaseClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_BD_SUPABASE_URL?.trim() || ''
const supabaseAnonKey = import.meta.env.VITE_BD_SUPABASE_ANON_KEY?.trim() || ''

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
