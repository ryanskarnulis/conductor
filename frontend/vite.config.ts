import { defineConfig } from 'vitest/config'
import { loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // DEV_HOST (from the gitignored .env) controls the dev server bind.
  // Defaults to loopback; set DEV_HOST=0.0.0.0 to expose on the LAN.
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [
      react(),
      // Dev-only: Vite's import analysis appends `?import` to onnxruntime's
      // dynamic import of its /vad/ .mjs loader. public/ serving tolerates
      // the query, but strip it anyway so the asset can never fall through
      // to the SPA-fallback HTML. Prod builds keep the import native.
      // (The /vad/ assets are copied into public/ by predev/prebuild —
      // see scripts/copy-vad-assets.mjs.)
      {
        name: 'vad-assets-ignore-query',
        configureServer(server) {
          server.middlewares.use((req, _res, next) => {
            if (req.url?.startsWith('/vad/')) req.url = req.url.replace(/\?.*$/, '')
            next()
          })
        },
      },
    ],
    server: {
      host: env.DEV_HOST || '127.0.0.1',
      port: 5174,
      strictPort: true,
      // The vendored voice modules (src/voice/, chess-canonical) fetch
      // relative /api/voice/* URLs — same-origin by design. nginx proxies
      // /api in the docker deployment; this is the dev equivalent.
      proxy: { '/api': 'http://127.0.0.1:8301' },
    },
    test: {
      environment: 'jsdom',
      setupFiles: './src/test/setup.ts',
    },
  }
})
