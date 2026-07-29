// vendored voice module — canonical: chess/frontend/src/tts.ts (agent-standard/voice.md); re-copy, never edit
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
/** The most recently requested playback, settled when it finishes (or fails,
 * or is interrupted) — never rejected. */
let currentPlayback: Promise<void> = Promise.resolve()
/** Settles the promise of the clip currently in the element; called when a
 * new clip interrupts it, because its own onended will never fire. */
let settleInterrupted: (() => void) | null = null

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
  return currentPlayback
}

/**
 * Fetch spoken audio for `text` and play it. Resolves once playback has
 * *finished* — not merely started — so callers can sequence work after the
 * reply has been heard. Never rejects.
 */
export function playText(text: string): Promise<void> {
  const playback = speak(text)
  currentPlayback = playback
  return playback
}

/**
 * Request the audio for `text` and hand back an object URL for it, or null
 * when voice is unavailable (503/502) or the deadline has already passed.
 */
async function loadClip(text: string, signal: AbortSignal): Promise<string | null> {
  const res = await fetch('/api/voice/speak', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
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

async function speak(text: string): Promise<void> {
  // Voice out is best-effort: a transport failure (offline, DNS, aborted
  // connection), an unreadable body or a request that never answers at all
  // must settle exactly like a 503 does, or the hands-free loop awaiting
  // audioIdle() never reopens the mic. So the deadline resolves the race with
  // "no clip" — the same best-effort completion every other failure produces —
  // rather than rejecting, and it settles the wait even if the request itself
  // stays pending forever.
  const controller = new AbortController()
  let expire = () => {}
  const expired = new Promise<null>((resolve) => {
    expire = () => resolve(null)
  })
  const deadline = setTimeout(() => {
    controller.abort()
    expire()
  }, SPEAK_DEADLINE_MS)

  let url: string
  try {
    const clip = await Promise.race([loadClip(text, controller.signal), expired])
    if (clip === null) return
    url = clip
  } catch {
    return
  } finally {
    // Every exit — played, unavailable, failed, expired — drops the timer.
    clearTimeout(deadline)
  }
  const el = element()
  // A new clip interrupts whatever was loaded: settle the old clip's promise
  // (its onended will never fire now), release its URL, and guard the stale
  // handlers so they can't revoke the live clip.
  settleInterrupted?.()
  if (liveUrl) URL.revokeObjectURL(liveUrl)
  liveUrl = url
  el.src = url
  await new Promise<void>((resolve) => {
    settleInterrupted = resolve
    const finish = () => {
      if (liveUrl === url) {
        URL.revokeObjectURL(url)
        liveUrl = null
      }
      if (settleInterrupted === resolve) settleInterrupted = null
      resolve()
    }
    el.onended = finish
    el.onerror = finish
    // Autoplay blocked (no unlocked element) or playback failed — don't
    // leak the URL, and settle so callers never hang.
    el.play().then(undefined, finish)
  })
}
