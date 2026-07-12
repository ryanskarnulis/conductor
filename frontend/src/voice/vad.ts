// vendored voice module — canonical: chess/frontend/src/vad.ts (agent-standard/voice.md); re-copy, never edit
// Voice activity detection for hands-free conversation mode: Silero VAD
// running in an AudioWorklet + WASM, entirely on-device (the project's
// local-only-voice decision bans cloud speech paths, not local models).
//
// This wrapper is the only place that knows the library; the component deals
// in these three verbs and two callbacks, which also makes it mockable in
// jsdom, where no audio machinery exists.

export interface SpeechEvents {
  /** The user started talking. */
  onSpeechStart: () => void
  /** The user stopped: the full utterance as 16 kHz mono Float32 samples. */
  onSpeechEnd: (audio: Float32Array) => void
  /**
   * The VAD could not start and hands-free is off the table — the reason,
   * for the UI. Phones have no console; a silent fallback to push-to-talk
   * reads as "continuous voice is broken" with nothing to diagnose from.
   */
  onUnavailable?: (reason: string) => void
}

export interface Vad {
  /** Stop feeding audio to the model (half-duplex: agent's turn). */
  pause: () => void
  /** Start listening again (user's turn). */
  resume: () => void
  /** Tear down the worklet and release the microphone. */
  destroy: () => void
}

/**
 * Silence that ends an utterance. Long enough for a mid-command think
 * ("knight takes… e5"), short enough that sending doesn't feel laggy.
 */
export const END_OF_SPEECH_MS = 1000

const RELOADED_KEY = 'vad-stale-chunk-reloaded'

/**
 * True when `e` is a dynamic-import failure and this tab hasn't already
 * reloaded for one. A rebuilt image renames every hashed chunk, so a tab
 * still running the old build 404s importing the lazy VAD chunk; one reload
 * fetches the fresh index.html. The storage flag stops a reload loop when a
 * chunk is genuinely missing.
 */
export function staleChunkReload(
  e: unknown,
  storage: Storage = sessionStorage,
): boolean {
  const message = e instanceof Error ? e.message : String(e)
  const stale =
    /dynamically imported module|module script failed/i.test(message)
  try {
    if (!stale || storage.getItem(RELOADED_KEY)) return false
    storage.setItem(RELOADED_KEY, '1')
    return true
  } catch {
    return false
  }
}

/**
 * Build and start a microphone VAD, or return null when anything fails
 * (no worklet support, model failed to load, mic denied) — the caller
 * falls back to push-to-talk.
 */
export async function createVad(events: SpeechEvents): Promise<Vad | null> {
  try {
    // Lazy-loaded: the VAD + onnxruntime stack is heavy and only needed once
    // the user actually taps the mic.
    const { MicVAD } = await import('@ricky0123/vad-web')
    const vad = await MicVAD.new({
      // Local-first: model, worklet, and ORT WASM are copied to /vad/ at
      // build time (vite-plugin-static-copy) — never fetched from a CDN.
      baseAssetPath: '/vad/',
      onnxWASMBasePath: '/vad/',
      redemptionMs: END_OF_SPEECH_MS,
      onSpeechStart: events.onSpeechStart,
      onSpeechEnd: events.onSpeechEnd,
      ortConfig: (ort) => {
        // iOS Safari dies allocating the threaded runtime's upfront wasm
        // memory ("no available backend found … RangeError: Out of memory",
        // seen live 2026-07-11). One thread is plenty for the ~2MB Silero
        // model; keep the library's error-only log level.
        ort.env.logLevel = 'error'
        ort.env.wasm.numThreads = 1
      },
    })
    try {
      // Healthy again: allow a future stale-build state to reload once more.
      sessionStorage.removeItem(RELOADED_KEY)
    } catch {
      // Storage unavailable (private mode); the guard just stays one-shot.
    }
    return {
      pause: () => void vad.pause(),
      resume: () => void vad.start(),
      destroy: () => void vad.destroy(),
    }
  } catch (e) {
    console.warn('hands-free VAD unavailable:', e)
    if (staleChunkReload(e)) {
      // Stale build: the lazy chunk this page references no longer exists
      // on the server. Reload once to pick up the fresh index.html.
      location.reload()
      return null
    }
    events.onUnavailable?.(e instanceof Error ? e.message : String(e))
    return null
  }
}
