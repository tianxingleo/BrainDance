import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import basicSsl from '@vitejs/plugin-basic-ssl'

const supabaseProxyTarget = process.env.VITE_BD_SUPABASE_PROXY_TARGET?.trim() || ''

export default defineConfig({
  plugins: [vue(), basicSsl()],
  base: './',
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5174,
    proxy: supabaseProxyTarget
      ? {
          '/supabase-proxy': {
            target: supabaseProxyTarget,
            changeOrigin: true,
            secure: false,
            rewrite: (path) => path.replace(/^\/supabase-proxy/, ''),
          },
        }
      : undefined,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Resource-Policy': 'cross-origin',
    },
  },
})
