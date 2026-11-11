// vendored voice module — canonical: speech/client/frontend/vad.ts (agent-standard/voice.md); re-copy, never edit
// Voice activity detection for hands-free conversation mode: Silero VAD
// running in an AudioWorklet + WASM, entirely on-device (the project's
// local-only-voice decision bans cloud speech paths, not local models).
//
// This wrapper is the only place that knows the library; the component deals
// in these three verbs and two callbacks, which also makes it mockable in
// jsdom, where no audio machinery exists.

import type { MicVAD as MicVadInstance } from '@ricky0123/vad-web'

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
  /**
   * Speech has lasted at least `sustainedSpeechMs` without a break — fires
   * once per run, so a cough or a click never reaches it. This is what
   * barge-in listens to: the moment the user talks over a spoken reply.
   * Optional; leaving it out adds no per-frame work.
   */
  onSustainedSpeech?: () => void
}

export interface VadOptions {
  /** How long speech must last before `onSustainedSpeech` fires. */
  sustainedSpeechMs?: number
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

/**
 * Minimum speech, in milliseconds, before it counts as barge-in. Short enough
 * that talking over the agent cuts it off within ~200 ms; long enough that a
 * blip, or a trace of the agent's own voice the browser's echo cancellation
 * let through, does not. The VAD works in 96 ms frames, so in practice this is
 * two frames of speech.
 */
export const BARGE_IN_MIN_SPEECH_MS = 150

// The browser's own echo cancellation, noise suppression and gain control are
// the only defence against the agent hearing itself while it talks (barge-in
// keeps the mic open over replies). These are the library's defaults, pinned
// here so they stay on for every grab of the microphone, including the
// re-grab after each pause().
const MIC_CONSTRAINTS: MediaStreamConstraints = {
  audio: {
    channelCount: 1,
    echoCancellation: true,
    autoGainControl: true,
    noiseSuppression: true,
  },
}
const grabMic = () => navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS)

/** Library fallbacks for the speech thresholds, used until the VAD exists. */
const DEFAULT_POSITIVE_THRESHOLD = 0.3
const DEFAULT_NEGATIVE_THRESHOLD = 0.25

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
export async function createVad(
  events: SpeechEvents,
  options: VadOptions = {},
): Promise<Vad | null> {
  // Consecutive speech so far, in ms, and whether this run has already fired.
  let run = 0
  let fired = false
  let vad: MicVadInstance | null = null
  const sustained = events.onSustainedSpeech
  const minMs = options.sustainedSpeechMs ?? BARGE_IN_MIN_SPEECH_MS
  const resetRun = () => {
    run = 0
    fired = false
  }
  try {
    // Lazy-loaded: the VAD + onnxruntime stack is heavy and only needed once
    // the user actually taps the mic.
    const { MicVAD } = await import('@ricky0123/vad-web')
    vad = await MicVAD.new({
      // Local-first: model, worklet, and ORT WASM are copied to /vad/ at
      // build time (vite-plugin-static-copy) — never fetched from a CDN.
      baseAssetPath: '/vad/',
      onnxWASMBasePath: '/vad/',
      redemptionMs: END_OF_SPEECH_MS,
      onSpeechStart: events.onSpeechStart,
      onSpeechEnd: events.onSpeechEnd,
      getStream: grabMic,
      resumeStream: grabMic,
      // Per-frame speech probability, for the minimum-speech rule. The library
      // only reports a segment once the user has stopped for END_OF_SPEECH_MS,
      // far too late to tell a blip from a barge-in. Added only when asked
      // for, so apps that ignore barge-in run the VAD exactly as before.
      ...(sustained && {
        onFrameProcessed: (probs: { isSpeech: number }, frame: Float32Array) => {
          const positive = vad?.options.positiveSpeechThreshold ?? DEFAULT_POSITIVE_THRESHOLD
          const negative = vad?.options.negativeSpeechThreshold ?? DEFAULT_NEGATIVE_THRESHOLD
          if (probs.isSpeech >= positive) {
            run += frame.length / 16 // 16 kHz samples → ms
            if (!fired && run >= minMs) {
              fired = true
              sustained()
            }
          } else if (probs.isSpeech < negative) {
            // A real gap ends the run; a murky frame between the thresholds
            // neither counts nor breaks it, as in the library's own segmenting.
            resetRun()
          }
        },
      }),
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
    const live = vad
    return {
      pause: () => {
        resetRun() // a half-counted run must not carry over a pause
        void live.pause()
      },
      resume: () => {
        resetRun()
        void live.start()
      },
      destroy: () => void live.destroy(),
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
