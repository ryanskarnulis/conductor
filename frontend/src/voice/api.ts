// Conductor-owned adapter for the vendored voice modules (which import
// './api'). Not vendored itself: this is the app-side seam the standard
// allows — restyling and wiring live outside the verbatim copies.
//
// Relative URL on purpose, matching the vendored tts.ts: nginx proxies /api
// in the docker deployment, the Vite dev server proxies it to the backend.

/**
 * Send recorded audio to the backend for transcription. Returns the
 * recognized text, or null when voice is unavailable (no speech service →
 * 503, speech backend down → 502). The caller feeds the text into the same
 * agent pipeline as typed input — voice never gets its own path.
 */
export async function transcribe(
  audio: Blob,
  filename = 'clip.webm',
): Promise<string | null> {
  const form = new FormData()
  // The filename extension tells the speech backend the container format
  // (webm from MediaRecorder push-to-talk, wav from the hands-free VAD).
  form.append('audio', audio, filename)
  const res = await fetch('/api/voice/transcribe', { method: 'POST', body: form })
  if (!res.ok) return null
  const data = (await res.json()) as { text: string }
  return data.text
}
