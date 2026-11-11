// vendored voice module — canonical: speech/client/frontend/tts.ts (agent-standard/voice.md); re-copy, never edit
// Voice out: fetch spoken audio for a piece of agent commentary and play it.
// Best-effort by design — if voice is unavailable (503/502) or autoplay is
// blocked, the commentary is still on screen, so failures are silent.
//
// Mobile autoplay: iOS Safari and Android Chrome refuse play() outside a user
// gesture, and the agent reply lands seconds after the tap that requested it.
// The escape hatch is per-element: an <audio> element that has once played
// inside a gesture may be reused programmatically forever after. So all
// playback goes through ONE shared element, and unlockAudio() — called
// synchronously from the gesture handlers — primes it with a silent clip.

/**
 * Client deadline for one /api/voice/speak request, in milliseconds.
 *
 * A request the backend or proxy accepts but never settles is the one failure
 * the catch below cannot see — nothing rejects, so the promise stays pending
 * forever, currentPlayback never settles and the hands-free loop awaiting
 * audioIdle() never reopens the mic. So the fetch carries its own deadline.
 *
 * The value sits between the two timeouts that bracket it: comfortably ABOVE
 * the backend's own speech timeout (60s read + 10s connect), so a slow but
 * living upstream always wins the race and reports its own 502 rather than
 * being cut off by the client, and far BELOW the ~300s proxy ceiling, so the
 * give-up is always this deadline rather than a socket dying under us.
 */
const SPEAK_DEADLINE_MS = 90_000

// Four samples of 8 kHz 8-bit silence: the shortest well-formed WAV.
const SILENT_WAV =
  'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQQAAACAgICA'

let sharedAudio: HTMLAudioElement | null = null
let unlocked = false
/** Object URL of the clip currently loaded in the shared element. */
let liveUrl: string | null = null
/** How one playback ended. `ended` played to the end; `error` failed while
 * playing; `interrupted` was cut off before it ended — by a newer clip, or by
 * stopPlayback() (mid-clip or while its audio was still being fetched);
 * `blocked` was refused by the browser (autoplay); `no_audio` never got a clip
 * (voice unavailable, a transport failure); `timeout` gave up at the speak
 * deadline. */
export type PlaybackOutcome =
  | 'ended'
  | 'error'
  | 'interrupted'
  | 'blocked'
  | 'no_audio'
  | 'timeout'

/** How much of a reply the listener heard before it was cut off by
 * stopPlayback() (a barge-in, a mute) — what a host sends along with the next
 * utterance so the agent does not assume the rest was heard. Read it once with
 * takeHeard(). A reply that played to the end has no report. */
export interface Heard {
  /** Milliseconds of the reply that played before the cut (0 if none did). */
  playedMs: number
  /** Length of the whole reply audio, or null when the browser can't say. */
  totalMs: number | null
  /** Estimated characters of the reply text heard: text length × played/total,
   * rounded. null only when audio played but its total length is unknown. */
  heardChars: number | null
  /** Length of the reply text, so `heardChars` of `totalChars` was heard. */
  totalChars: number
}

/** Optional observer of one playback, for latency measurement. `id` rides the
 * speak request as `X-Interaction-Id`, so the host's server can join its own
 * record of the synthesis to the client's; the callbacks fire once each, at
 * the moment the clip is loaded and the moment audio actually starts. */
export interface PlaybackObserver {
  id?: string
  onReady?: () => void
  onStart?: () => void
}

/** The most recently requested playback, settled when it finishes (or fails,
 * or is interrupted) — never rejected. */
let currentPlayback: Promise<PlaybackOutcome | void> = Promise.resolve()
/** Settles the promise of the clip currently in the element; called when a
 * new clip interrupts it, because its own onended will never fire. */
let settleInterrupted: (() => void) | null = null
/** Cancels each speak request still fetching its clip; stopPlayback() runs
 * them all, since any of them would otherwise start playing when it lands. */
const pendingStops = new Set<() => void>()
/** Text of the clip currently loaded in the shared element, for heardReport. */
let liveText = ''
/** The report of the reply stopPlayback() last cut, until read or superseded. */
let heardReport: Heard | null = null

function element(): HTMLAudioElement {
  if (!sharedAudio) sharedAudio = new Audio()
  return sharedAudio
}

/**
 * Prime the shared audio element so later programmatic playback is allowed.
 * MUST be called synchronously from a user-gesture handler (click/submit) —
 * that's the whole point. No-op once an unlock has succeeded.
 */
export function unlockAudio(): void {
  if (unlocked) return
  const el = element()
  el.src = SILENT_WAV
  el.play().then(
    () => {
      unlocked = true
    },
    () => {
      // Blocked even inside the gesture (or jsdom); the next gesture retries.
    },
  )
}

/**
 * Resolves when the most recently requested clip has finished playing (or
 * failed, or was interrupted). Resolves immediately when nothing is pending.
 * The hands-free loop waits on this before reopening the mic, so the agent
 * never listens to its own voice.
 */
export function audioIdle(): Promise<void> {
  return currentPlayback.then(() => undefined)
}

/**
 * Fetch spoken audio for `text` and play it. Resolves once playback has
 * *finished* — not merely started — so callers can sequence work after the
 * reply has been heard, with how it finished. Never rejects.
 */
export function playText(text: string, observer?: PlaybackObserver): Promise<PlaybackOutcome> {
  const playback = speak(text, observer)
  currentPlayback = playback
  return playback
}

/**
 * The heard-up-to report of the reply stopPlayback() last cut short, or null
 * when there is none — and then clears it, so each cut is reported once, with
 * the next utterance. Null for a reply that played to the end, failed, or was
 * replaced by a newer one; a report is also dropped when the next reply's clip
 * loads, since the next utterance answers that one. Hosts that ignore it lose
 * nothing.
 */
export function takeHeard(): Heard | null {
  const report = heardReport
  heardReport = null
  return report
}

function heardOf(text: string, playedMs: number, totalMs: number | null): Heard {
  const totalChars = text.length
  let heardChars: number | null = null
  if (playedMs <= 0) heardChars = 0
  else if (totalMs !== null) {
    heardChars = Math.min(totalChars, Math.round((totalChars * playedMs) / totalMs))
  }
  return { playedMs, totalMs, heardChars, totalChars }
}

/**
 * Stop speaking now: cut off the clip that is playing and cancel any clip
 * still being fetched. Every playback this cuts short resolves `interrupted`,
 * so audioIdle() resolves and a hands-free loop awaiting it reopens the mic.
 * The shared element is paused, not replaced, so its mobile-autoplay unlock
 * survives and the next playText() plays normally. No-op when nothing is
 * playing or pending — in particular it leaves an in-progress unlockAudio()
 * alone, since pausing that silent clip would void the unlock.
 *
 * What was cut is recorded for takeHeard(): the live clip with how much of it
 * played, or a clip still being fetched as nothing heard.
 */
export function stopPlayback(): void {
  if (liveUrl !== null) {
    const el = element()
    const playedMs = Math.round((Number.isFinite(el.currentTime) ? el.currentTime : 0) * 1000)
    const totalMs =
      Number.isFinite(el.duration) && el.duration > 0 ? Math.round(el.duration * 1000) : null
    heardReport = heardOf(liveText, playedMs, totalMs)
  }
  // Requests still fetching are newer than the live clip, so they win.
  for (const stop of [...pendingStops]) stop()
  if (liveUrl === null) return
  element().pause()
  // finish() revokes the live URL and clears the settle hook.
  settleInterrupted?.()
}

/**
 * Request the audio for `text` and hand back an object URL for it, or null
 * when voice is unavailable (503/502) or the deadline has already passed.
 */
async function loadClip(
  text: string,
  signal: AbortSignal,
  id: string | undefined,
): Promise<string | null> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (id) headers['X-Interaction-Id'] = id
  const res = await fetch('/api/voice/speak', {
    method: 'POST',
    headers,
    body: JSON.stringify({ text }),
    signal,
  })
  if (!res.ok) return null
  const url = URL.createObjectURL(await res.blob())
  // The deadline may have fired while the body was being read. Nobody is
  // waiting for this clip any more, so release it rather than leak it.
  if (signal.aborted) {
    URL.revokeObjectURL(url)
    return null
  }
  return url
}

async function speak(text: string, observer?: PlaybackObserver): Promise<PlaybackOutcome> {
  // Voice out is best-effort: a transport failure (offline, DNS, aborted
  // connection), an unreadable body or a request that never answers at all
  // must settle exactly like a 503 does, or the hands-free loop awaiting
  // audioIdle() never reopens the mic. So the deadline resolves the race with
  // "no clip" — the same best-effort completion every other failure produces —
  // rather than rejecting, and it settles the wait even if the request itself
  // stays pending forever. stopPlayback() cuts the wait short the same way.
  const controller = new AbortController()
  let release = () => {}
  /** Why the wait was cut short — the deadline or a stop — once it was. */
  let cut: PlaybackOutcome | null = null
  const cutShort = new Promise<null>((resolve) => {
    release = () => resolve(null)
  })
  const cutOff = (outcome: PlaybackOutcome) => {
    if (cut !== null) return
    cut = outcome
    controller.abort()
    release()
  }
  const deadline = setTimeout(() => cutOff('timeout'), SPEAK_DEADLINE_MS)
  const stop = () => {
    if (cut !== null) return
    heardReport = heardOf(text, 0, null)
    cutOff('interrupted')
  }
  pendingStops.add(stop)

  let url: string
  try {
    const clip = await Promise.race([loadClip(text, controller.signal, observer?.id), cutShort])
    if (clip !== null && cut !== null) {
      // The clip landed in the same tick it was given up on; nobody wants it.
      URL.revokeObjectURL(clip)
    }
    if (clip === null || cut !== null) return cut ?? 'no_audio'
    url = clip
  } catch {
    return cut ?? 'no_audio'
  } finally {
    // Every exit — played, unavailable, failed, expired, stopped — drops the
    // timer and the stop hook.
    clearTimeout(deadline)
    pendingStops.delete(stop)
  }
  const el = element()
  // A new clip interrupts whatever was loaded: settle the old clip's promise
  // (its onended will never fire now), release its URL, and guard the stale
  // handlers so they can't revoke the live clip.
  settleInterrupted?.()
  if (liveUrl) URL.revokeObjectURL(liveUrl)
  liveUrl = url
  liveText = text
  // The next utterance answers this reply, not an earlier one that was cut.
  heardReport = null
  el.src = url
  observer?.onReady?.()
  return new Promise<PlaybackOutcome>((resolve) => {
    let settled = false
    const finish = (outcome: PlaybackOutcome) => {
      if (settled) return
      settled = true
      if (liveUrl === url) {
        URL.revokeObjectURL(url)
        liveUrl = null
      }
      if (settleInterrupted === interrupt) settleInterrupted = null
      resolve(outcome)
    }
    const interrupt = () => finish('interrupted')
    settleInterrupted = interrupt
    let started = false
    el.onplaying = () => {
      if (started || liveUrl !== url) return
      started = true
      observer?.onStart?.()
    }
    el.onended = () => finish('ended')
    el.onerror = () => finish('error')
    // Autoplay blocked (no unlocked element) or playback failed — don't
    // leak the URL, and settle so callers never hang.
    el.play().then(undefined, () => finish('blocked'))
  })
}
