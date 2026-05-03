import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import basicSsl from '@vitejs/plugin-basic-ssl'

export default defineConfig({
  plugins: [
    vue(),
    basicSsl()
  ],
  base: './', 
  resolve: {
    alias: [
      { find: '@', replacement: fileURLToPath(new URL('./src', import.meta.url)) },
      { find: /^three$/, replacement: fileURLToPath(new URL('./src/vendor/three-compat.ts', import.meta.url)) },
    ],
  },
  build: {
    rollupOptions: {
      output: {
        entryFileNames: 'assets/[name]-rollfix-[hash].js',
        chunkFileNames: 'assets/[name]-rollfix-[hash].js',
        assetFileNames: 'assets/[name]-rollfix-[hash][extname]',
      },
    },
  },
  server: {
    host: '0.0.0.0',
    // 【✅ 必须添加以下 headers 才能让 3DGS 运行】
    // https 由 basicSsl 插件自动处理，无需在此设置
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    }
  }
})
