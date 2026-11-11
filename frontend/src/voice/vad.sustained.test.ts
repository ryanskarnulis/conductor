import { beforeEach, describe, expect, it, vi } from 'vitest'
import { BARGE_IN_MIN_SPEECH_MS, createVad } from './vad'
import { MicVAD } from '@ricky0123/vad-web'

// Canonical tests for the sustained-speech rule behind barge-in (speech#5);
// they run in a consumer's vitest setup next to its vendored vad.ts. The real
// VAD (AudioWorklet + WASM) is replaced, as in the apps' own vad tests, and
// the tests feed the per-frame callback by hand.

vi.mock('@ricky0123/vad-web', () => ({ MicVAD: { new: vi.fn() } }))

const micVadNew = vi.mocked(MicVAD.new)
const fakeVad = {
  start: vi.fn(async () => {}),
  pause: vi.fn(async () => {}),
  destroy: vi.fn(async () => {}),
  options: { positiveSpeechThreshold: 0.3, negativeSpeechThreshold: 0.25 },
}

/** The legacy Silero model the library defaults to: 1536 samples = 96 ms. */
const FRAME = new Float32Array(1536)
const SPEECH = { notSpeech: 0.1, isSpeech: 0.9 }
const SILENCE = { notSpeech: 0.95, isSpeech: 0.05 }
const MURKY = { notSpeech: 0.72, isSpeech: 0.28 } // between the two thresholds

type Probs = typeof SPEECH
type Options = NonNullable<Parameters<typeof MicVAD.new>[0]>

async function setup(sustainedSpeechMs?: number) {
  const onSustainedSpeech = vi.fn()
  const vad = await createVad(
    { onSpeechStart: vi.fn(), onSpeechEnd: vi.fn(), onSustainedSpeech },
    sustainedSpeechMs === undefined ? undefined : { sustainedSpeechMs },
  )
  const options = micVadNew.mock.calls[0][0] as Options
  const feed = (...frames: Probs[]) =>
    frames.forEach((p) => void options.onFrameProcessed!(p, FRAME))
  return { vad: vad!, onSustainedSpeech, options, feed }
}

beforeEach(() => {
  vi.clearAllMocks()
  micVadNew.mockResolvedValue(fakeVad as unknown as MicVAD)
})

describe('sustained speech (the minimum-speech rule for barge-in)', () => {
  it('defaults to a short, named minimum', () => {
    expect(BARGE_IN_MIN_SPEECH_MS).toBe(150)
  })

  it('ignores a blip shorter than the minimum', async () => {
    const { onSustainedSpeech, feed } = await setup()
    feed(SPEECH) // one 96 ms frame: a cough, a click, a bit of echo
    feed(SILENCE)
    expect(onSustainedSpeech).not.toHaveBeenCalled()
  })

  it('fires on the frame that takes continuous speech past the minimum', async () => {
    const { onSustainedSpeech, feed } = await setup()
    feed(SPEECH)
    expect(onSustainedSpeech).not.toHaveBeenCalled() // 96 ms < 150 ms
    feed(SPEECH)
    expect(onSustainedSpeech).toHaveBeenCalledTimes(1) // 192 ms >= 150 ms
  })

  it('fires once per run, not once per frame', async () => {
    const { onSustainedSpeech, feed } = await setup()
    feed(SPEECH, SPEECH, SPEECH, SPEECH, SPEECH)
    expect(onSustainedSpeech).toHaveBeenCalledTimes(1)
  })

  it('does not add up separate blips into speech', async () => {
    const { onSustainedSpeech, feed } = await setup()
    feed(SPEECH, SILENCE, SPEECH, SILENCE, SPEECH)
    expect(onSustainedSpeech).not.toHaveBeenCalled()
  })

  it('treats a frame between the thresholds as neither speech nor a break', async () => {
    // Same hysteresis the library uses to hold a segment open.
    const { onSustainedSpeech, feed } = await setup()
    feed(SPEECH, MURKY)
    expect(onSustainedSpeech).not.toHaveBeenCalled()
    feed(SPEECH)
    expect(onSustainedSpeech).toHaveBeenCalledTimes(1)
  })

  it('fires again for the next run after a break', async () => {
    const { onSustainedSpeech, feed } = await setup()
    feed(SPEECH, SPEECH, SILENCE, SPEECH, SPEECH)
    expect(onSustainedSpeech).toHaveBeenCalledTimes(2)
  })

  it('takes the minimum as an option', async () => {
    const { onSustainedSpeech, feed } = await setup(400)
    feed(SPEECH, SPEECH, SPEECH, SPEECH) // 384 ms
    expect(onSustainedSpeech).not.toHaveBeenCalled()
    feed(SPEECH)
    expect(onSustainedSpeech).toHaveBeenCalledTimes(1)
  })

  it('forgets a half-counted run across pause/resume', async () => {
    const { vad, onSustainedSpeech, feed } = await setup()
    feed(SPEECH)
    vad.pause()
    vad.resume()
    feed(SPEECH)
    expect(onSustainedSpeech).not.toHaveBeenCalled()
  })

  it('adds no frame hook when nobody listens for sustained speech', async () => {
    // Apps that ignore barge-in get exactly the VAD they had before.
    await createVad({ onSpeechStart: vi.fn(), onSpeechEnd: vi.fn() })
    const options = micVadNew.mock.calls[0][0] as Options
    expect('onFrameProcessed' in options).toBe(false)
  })
})

describe('echo cancellation', () => {
  it('asks the browser for it on every microphone grab, including after a pause', async () => {
    // Barge-in listens while the agent talks, so the speaker's own voice must
    // be cancelled by the browser (no custom echo handling).
    const getUserMedia = vi.fn(async () => ({}) as MediaStream)
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia } })
    await createVad({ onSpeechStart: vi.fn(), onSpeechEnd: vi.fn() })
    const options = micVadNew.mock.calls[0][0] as Options
    await options.getStream!()
    await options.resumeStream!({} as MediaStream)
    expect(getUserMedia).toHaveBeenCalledTimes(2)
    for (const [constraints] of getUserMedia.mock.calls as unknown as [MediaStreamConstraints][]) {
      expect(constraints.audio).toMatchObject({ echoCancellation: true })
    }
    vi.unstubAllGlobals()
  })
})
