// vendored voice module — canonical: chess/frontend/src/wav.ts (agent-standard/voice.md); re-copy, never edit
// The VAD emits raw Float32 samples (16 kHz mono, -1..1); the transcribe
// endpoint wants a real audio container. 16-bit PCM WAV is the simplest
// format every STT backend accepts.

const SAMPLE_RATE = 16000
const HEADER_BYTES = 44

export function encodeWav(samples: Float32Array): Blob {
  const dataBytes = samples.length * 2
  const buffer = new ArrayBuffer(HEADER_BYTES + dataBytes)
  const view = new DataView(buffer)

  writeAscii(view, 0, 'RIFF')
  view.setUint32(4, HEADER_BYTES + dataBytes - 8, true)
  writeAscii(view, 8, 'WAVE')
  writeAscii(view, 12, 'fmt ')
  view.setUint32(16, 16, true) // fmt chunk size
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, SAMPLE_RATE, true)
  view.setUint32(28, SAMPLE_RATE * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  writeAscii(view, 36, 'data')
  view.setUint32(40, dataBytes, true)

  for (let i = 0; i < samples.length; i++) {
    const clipped = Math.max(-1, Math.min(1, samples[i]))
    // -1..1 → signed 16-bit; the negative side has one extra step.
    view.setInt16(HEADER_BYTES + i * 2, Math.round(clipped * (clipped < 0 ? 32768 : 32767)), true)
  }

  return new Blob([buffer], { type: 'audio/wav' })
}

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i))
}
