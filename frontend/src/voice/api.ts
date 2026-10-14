// Conductor-owned adapter for the vendored voice modules (which import
// './api'). Not vendored itself: this is the app-side seam the standard
// allows — restyling and wiring live outside the verbatim copies.
//
// Relative URL on purpose, matching the vendored tts.ts: nginx proxies /api
// in the docker deployment, the Vite dev server proxies it to the backend.

/**
 * Client deadline for one /api/voice/transcribe request, in milliseconds.
 *
 * A request the backend or proxy accepts but never settles is the one failure
 * the catch below cannot see — nothing rejects, so the promise stays pending
 * forever and the mic never leaves 'transcribing', the state that disables the
 * button the user would tap to retry or exit. So the fetch carries its own
 * deadline.
 *
 * The value sits between the two timeouts that bracket it: comfortably ABOVE
 * the backend's own speech timeout (60s read + 10s connect), so a slow but
 * living upstream always wins the race and reports its own 502 rather than
 * being cut off by the client, and far BELOW the ~330s nginx ceiling, so the
 * give-up is always this deadline rather than a socket dying under us. Same
 * value and same shape as the vendored tts.ts SPEAK_DEADLINE_MS — one voice
 * deadline, not two.
 */
const TRANSCRIBE_DEADLINE_MS = 90_000

/** Post the clip and read the transcript out of the response, or null when
 * voice is unavailable (503/502) or the body isn't the shape we expect. */
async function requestTranscript(
  audio: Blob,
  filename: string,
  signal: AbortSignal,
  interactionId: string | undefined,
): Promise<string | null> {
  const form = new FormData()
  // The filename extension tells the speech backend the container format
  // (webm from MediaRecorder push-to-talk, wav from the hands-free VAD).
  form.append('audio', audio, filename)
  const res = await fetch('/api/voice/transcribe', {
    method: 'POST',
    body: form,
    signal,
    ...(interactionId ? { headers: { 'X-Interaction-Id': interactionId } } : {}),
  })
  if (!res.ok) return null
  const data = (await res.json()) as { text?: unknown }
  return typeof data.text === 'string' ? data.text : null
}

/**
 * Send recorded audio to the backend for transcription. Returns the
 * recognized text, or null when voice is unavailable (no speech service →
 * 503, speech backend down → 502) or the deadline expired. The caller feeds
 * the text into the same agent pipeline as typed input — voice never gets its
 * own path.
 *
 * Never rejects: an offline backend, DNS/proxy loss, a malformed response
 * body, or a request that never answers at all is an availability failure, not
 * a programmer error, and the vendored MicButton relies on the null to leave
 * 'working'/'transcribing'. A rejection escaping here strands the mic in a
 * state the user can't tap out of.
 *
 * `interactionId` is whatever the vendored MicButton's optional
 * `beginUtterance` hook returned (chess's latency tracing); it rides along as
 * `X-Interaction-Id`. Conductor passes no hook today, so no header is sent.
 */
export async function transcribe(
  audio: Blob,
  filename = 'clip.webm',
  interactionId?: string,
): Promise<string | null> {
  // Expiry has to convert into the same "unavailable" null every other failure
  // produces. Aborting alone would not be enough: a fetch that ignores the
  // signal — the wedge shape this guards against — leaves the request pending
  // regardless, so the race against the timer is what guarantees settlement
  // and the abort is what makes a real browser actually cancel the request.
  const controller = new AbortController()
  let expire = () => {}
  const expired = new Promise<null>((resolve) => {
    expire = () => resolve(null)
  })
  const deadline = setTimeout(() => {
    controller.abort()
    expire()
  }, TRANSCRIBE_DEADLINE_MS)

  try {
    return await Promise.race([
      requestTranscript(audio, filename, controller.signal, interactionId),
      expired,
    ])
  } catch {
    return null
  } finally {
    // Every exit — transcribed, unavailable, failed, expired — drops the timer.
    clearTimeout(deadline)
  }
}
