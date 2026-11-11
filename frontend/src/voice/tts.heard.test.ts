import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Canonical tests for the heard-up-to report (speech#5). They run inside a
// consumer's vitest setup next to its vendored copy of tts.ts — this repo has
// no JS tooling of its own — and assume jsdom, like the apps' own tts tests.
//
// jsdom plays no audio, so the shared <audio> element is faked, with the two
// fields the report is computed from (currentTime, duration) settable by hand.
// tts.ts keeps module-level state, so every test re-imports a fresh copy.

// Subclass, not a bare object: Response and friends still construct URLs.
class FakeURL extends URL {
  static createObjectURL = vi.fn(() => 'blob:fake')
  static revokeObjectURL = vi.fn()
}

const play = vi.fn()
let audioInstances: FakeAudio[]

class FakeAudio {
  src = ''
  currentTime = 0
  duration = NaN
  onended: (() => void) | null = null
  onerror: (() => void) | null = null
  onplaying: (() => void) | null = null
  play = play
  pause = vi.fn()
  constructor() {
    audioInstances.push(this)
  }
}

async function loadTts() {
  vi.resetModules()
  return import('./tts')
}

const clip = () => vi.fn(async () => new Response(new Blob([new Uint8Array([1])]), { status: 200 }))

beforeEach(() => {
  audioInstances = []
  play.mockReset().mockResolvedValue(undefined)
  vi.stubGlobal('Audio', FakeAudio)
  vi.stubGlobal('URL', FakeURL)
  vi.stubGlobal('fetch', clip())
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('takeHeard', () => {
  it('reports played and total milliseconds and a scaled character estimate', async () => {
    const tts = await loadTts()
    const reply = tts.playText('x'.repeat(100))
    await vi.waitFor(() => expect(play).toHaveBeenCalled())
    audioInstances[0].duration = 4
    audioInstances[0].currentTime = 1
    tts.stopPlayback()
    await expect(reply).resolves.toBe('interrupted')
    expect(audioInstances[0].pause).toHaveBeenCalled()
    expect(tts.takeHeard()).toEqual({
      playedMs: 1000,
      totalMs: 4000,
      heardChars: 25,
      totalChars: 100,
    })
  })

  it('is delivered once: the next call has nothing', async () => {
    const tts = await loadTts()
    const reply = tts.playText('hello there')
    await vi.waitFor(() => expect(play).toHaveBeenCalled())
    audioInstances[0].duration = 2
    audioInstances[0].currentTime = 1
    tts.stopPlayback()
    await reply
    expect(tts.takeHeard()).not.toBeNull()
    expect(tts.takeHeard()).toBeNull()
  })

  it('rounds the estimate to whole characters and never exceeds the text', async () => {
    const tts = await loadTts()
    const reply = tts.playText('abcdefghij') // 10 chars
    await vi.waitFor(() => expect(play).toHaveBeenCalled())
    audioInstances[0].duration = 3
    audioInstances[0].currentTime = 1 // 3.33 chars
    tts.stopPlayback()
    await reply
    expect(tts.takeHeard()?.heardChars).toBe(3)

    const second = tts.playText('abcdefghij')
    await vi.waitFor(() => expect(play).toHaveBeenCalledTimes(2))
    audioInstances[0].duration = 3
    audioInstances[0].currentTime = 3.2 // overshoot: the element can report past the end
    tts.stopPlayback()
    await second
    expect(tts.takeHeard()?.heardChars).toBe(10)
  })

  it('says nothing for a reply that played to the end', async () => {
    const tts = await loadTts()
    const reply = tts.playText('all of it')
    await vi.waitFor(() => expect(play).toHaveBeenCalled())
    audioInstances[0].duration = 2
    audioInstances[0].currentTime = 2
    audioInstances[0].onended?.()
    await expect(reply).resolves.toBe('ended')
    expect(tts.takeHeard()).toBeNull()
  })

  it('says nothing for a reply that failed to play', async () => {
    const tts = await loadTts()
    play.mockRejectedValue(new Error('NotAllowedError'))
    await expect(tts.playText('blocked')).resolves.toBe('blocked')
    expect(tts.takeHeard()).toBeNull()
  })

  it('says nothing when stopPlayback finds nothing playing or pending', async () => {
    const tts = await loadTts()
    tts.stopPlayback()
    expect(tts.takeHeard()).toBeNull()
  })

  it('drops the report of a cut reply once a newer reply starts', async () => {
    // The next utterance answers the newer reply, which the listener may have
    // heard in full; the earlier cut is no longer what they are replying to.
    const tts = await loadTts()
    const first = tts.playText('first reply')
    await vi.waitFor(() => expect(play).toHaveBeenCalled())
    audioInstances[0].duration = 2
    audioInstances[0].currentTime = 1
    tts.stopPlayback()
    await first
    const second = tts.playText('second reply')
    await vi.waitFor(() => expect(play).toHaveBeenCalledTimes(2))
    expect(tts.takeHeard()).toBeNull()
    audioInstances[0].onended?.()
    await second
  })

  it('does not report a reply a newer one replaced (nobody stopped it)', async () => {
    const tts = await loadTts()
    const first = tts.playText('first reply')
    await vi.waitFor(() => expect(play).toHaveBeenCalled())
    const second = tts.playText('second reply')
    await expect(first).resolves.toBe('interrupted')
    await vi.waitFor(() => expect(play).toHaveBeenCalledTimes(2))
    expect(tts.takeHeard()).toBeNull()
    audioInstances[0].onended?.()
    await second
  })

  it('reports nothing heard for a reply cut before its audio arrived', async () => {
    const tts = await loadTts()
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(() => {})))
    const reply = tts.playText('x'.repeat(40))
    tts.stopPlayback()
    await expect(reply).resolves.toBe('interrupted')
    expect(tts.takeHeard()).toEqual({ playedMs: 0, totalMs: null, heardChars: 0, totalChars: 40 })
  })

  it('leaves the character estimate unknown when the clip length is unknown', async () => {
    const tts = await loadTts()
    const reply = tts.playText('x'.repeat(60))
    await vi.waitFor(() => expect(play).toHaveBeenCalled())
    audioInstances[0].duration = Infinity // streamed media can report this
    audioInstances[0].currentTime = 1.5
    tts.stopPlayback()
    await reply
    expect(tts.takeHeard()).toEqual({
      playedMs: 1500,
      totalMs: null,
      heardChars: null,
      totalChars: 60,
    })
  })

  it('still resolves audioIdle and keeps the element reusable after a cut', async () => {
    const tts = await loadTts()
    const reply = tts.playText('x'.repeat(10))
    await vi.waitFor(() => expect(play).toHaveBeenCalled())
    tts.stopPlayback()
    await reply
    await expect(tts.audioIdle()).resolves.toBeUndefined()
    const next = tts.playText('again')
    await vi.waitFor(() => expect(play).toHaveBeenCalledTimes(2))
    expect(audioInstances).toHaveLength(1)
    audioInstances[0].onended?.()
    await expect(next).resolves.toBe('ended')
  })
})
