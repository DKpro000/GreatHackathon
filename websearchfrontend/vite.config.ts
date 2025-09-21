import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // 前端发 /search，会被转发到 8080
      '/search':  { target: 'http://127.0.0.1:8080', changeOrigin: true },
      '/health':  { target: 'http://127.0.0.1:8080', changeOrigin: true },
      '/debug':   { target: 'http://127.0.0.1:8080', changeOrigin: true },
      '/reload':  { target: 'http://127.0.0.1:8080', changeOrigin: true, secure: false, ws: false },
    }
  }
})
