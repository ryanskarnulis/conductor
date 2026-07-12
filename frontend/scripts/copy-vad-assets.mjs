// Copy the hands-free-voice runtime assets out of node_modules into
// public/vad/ (gitignored): the Silero VAD models, the audio worklet, and
// the onnxruntime WASM + loader. Local-first — the app must never fetch
// these from a CDN — and vad.ts points baseAssetPath/onnxWASMBasePath at
// /vad/. Runs as predev/prebuild so the dev server, CI, and the Docker
// image all agree (postinstall wouldn't work: the Docker build runs npm ci
// before the source tree — including this script — is copied in). Vite
// serves public/ in dev and copies it into dist/ on build.
//
// (vite-plugin-static-copy would be the obvious tool, but its dev-server
// middleware mis-orders itself under Vite 8/rolldown and everything 404s.)

import { copyFileSync, globSync, mkdirSync } from 'node:fs'
import path from 'node:path'

const here = import.meta.dirname
const root = path.join(here, '..')
const dest = path.join(root, 'public', 'vad')

const patterns = [
  'node_modules/@ricky0123/vad-web/dist/*.onnx',
  'node_modules/@ricky0123/vad-web/dist/vad.worklet.bundle.min.js',
  'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded*',
]

mkdirSync(dest, { recursive: true })
let copied = 0
for (const pattern of patterns) {
  for (const file of globSync(path.join(root, pattern))) {
    copyFileSync(file, path.join(dest, path.basename(file)))
    copied++
  }
}
if (copied === 0) throw new Error('no VAD assets found — is @ricky0123/vad-web installed?')
console.log(`[copy-vad-assets] ${copied} files → public/vad/`)
