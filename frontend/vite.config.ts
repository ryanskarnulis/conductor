import { defineConfig } from 'vitest/config'
import { loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // DEV_HOST (from the gitignored .env) controls the dev server bind.
  // Defaults to loopback; set DEV_HOST=0.0.0.0 to expose on the LAN.
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react()],
    server: { host: env.DEV_HOST || '127.0.0.1', port: 5174, strictPort: true },
    test: {
      environment: 'jsdom',
      setupFiles: './src/test/setup.ts',
    },
  }
})
