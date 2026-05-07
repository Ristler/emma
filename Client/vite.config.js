import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react()],
    server: {
      proxy: {
        '/predict': {
          target: env.VITE_CHAT_ENDPOINT || 'http://localhost:5000',
          changeOrigin: true,
          secure: false
        },
        '/models': {
          target: env.VITE_CHAT_ENDPOINT || 'http://localhost:5000',
          changeOrigin: true,
          secure: false
        }
      }
    }
  }
})
