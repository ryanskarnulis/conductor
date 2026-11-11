import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MicButton } from './MicButton'
import { playText, stopPlayback, unlockAudio } from './tts'
import type { SpeechEvents, Vad } from './vad'

// Canonical tests for barge-in wiring in MicButton (speech#5); they run in a
// consumer's vitest setup next to its vendored copies. tts.ts is the REAL
// module over a fake <audio> element and fetch, so "speech stops playback" and
// "the next utterance carries what was heard" are exercised end to end; only
// the VAD (no AudioWorklet in jsdom), transcription and the host's icons are
// stubbed.

const transcribe = vi.hoisted(() => vi.fn())
const createVad = vi.hoisted(() => vi.fn())
vi.mock('./api', () => ({ transcribe }))
vi.mock('./vad', () => ({ createVad }))
vi.mock('./wav', () => ({ encodeWav: () => new Blob(['wav']) }))
vi.mock('./icons', () => ({ MicIcon: () => null, ListeningIcon: () => null }))
vi.mock('./tts', async () => ({
  ...(await vi.importActual<typeof import('./tts')>('./tts')),
  unlockAudio: vi.fn(), // a pure user-gesture side effect; jsdom cannot play
}))

// Subclass, not a bare object: Response and friends still construct URLs.
class FakeURL extends URL {
  static createObjectURL = vi.fn(() => 'blob:fake')
  static revokeObjectURL = vi.fn()
}

const play = vi.fn()
// tts.ts creates its shared element once per module load, so one instance
// serves every test (reset in beforeEach); audio[0] is always that element.
const audio: FakeAudio[] = []

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
    audio.push(this)
  }
}

const pause = vi.fn()
const resume = vi.fn()
const destroy = vi.fn()
const vad: Vad = { pause, resume, destroy }
let events: SpeechEvents

class FakeMediaRecorder {}

beforeEach(() => {
  if (audio[0]) Object.assign(audio[0], { src: '', currentTime: 0, duration: NaN, onended: null })
  audio[0]?.pause.mockClear()
  play.mockReset().mockResolvedValue(undefined)
  pause.mockReset()
  resume.mockReset()
  destroy.mockReset()
  transcribe.mockReset()
  vi.mocked(unlockAudio).mockReset()
  vi.stubGlobal('Audio', FakeAudio)
  vi.stubGlobal('URL', FakeURL)
  vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob([new Uint8Array([1])]))))
  vi.stubGlobal('MediaRecorder', FakeMediaRecorder)
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: vi.fn() },
  })
  createVad.mockReset().mockImplementation(async (e: SpeechEvents) => {
    events = e
    return vad
  })
})

afterEach(() => {
  // tts.ts state is module-wide: a reply left playing would stall the next
  // test's conversation start, which waits for audio to go idle.
  stopPlayback()
  vi.unstubAllGlobals()
})

type OnTranscript = React.ComponentProps<typeof MicButton>['onTranscript']

/** A host whose first turn speaks a 100-character reply and later turns don't. */
function replyToFirstOnly() {
  const onTranscript = vi.fn<OnTranscript>(async () => {
    if (onTranscript.mock.calls.length === 1) void playText('x'.repeat(100))
  })
  return onTranscript
}

/** Open a hands-free conversation and wait for the mic to be listening. */
async function startConversation(props: Partial<React.ComponentProps<typeof MicButton>>) {
  const onTranscript = props.onTranscript ?? vi.fn()
  render(<MicButton onTranscript={onTranscript} disabled={false} {...props} />)
  fireEvent.click(screen.getByRole('button', { name: /start voice conversation/i }))
  await waitFor(() => expect(createVad).toHaveBeenCalled())
  await act(async () => {})
  return onTranscript
}

/** The user says something; the mic hands it to transcription. */
async function say(text: string) {
  transcribe.mockResolvedValueOnce(text)
  await act(async () => {
    events.onSpeechEnd(new Float32Array(16))
  })
}

describe('barge-in', () => {
  it('stops the reply the moment sustained speech is heard over it', async () => {
    const onTranscript = vi.fn(async () => {
      void playText('x'.repeat(100))
    })
    await startConversation({ onTranscript, bargeIn: true })
    await say('what is on my calendar')
    await waitFor(() => expect(play).toHaveBeenCalled())

    expect(audio[0].pause).not.toHaveBeenCalled()
    // Synchronous: no timer, no round trip between the detection and the cut.
    act(() => events.onSustainedSpeech!())
    expect(audio[0].pause).toHaveBeenCalledTimes(1)
  })

  it('listens over the reply: the mic reopens as the reply starts, not after it', async () => {
    const onTranscript = vi.fn(async () => {
      void playText('hello')
    })
    await startConversation({ onTranscript, bargeIn: true })
    await say('hi')
    await waitFor(() => expect(play).toHaveBeenCalled())
    await waitFor(() => expect(resume).toHaveBeenCalled())
    // …while the reply is still playing.
    expect(audio[0].onended).not.toBeNull()
  })

  it('ignores a blip: speech that never becomes sustained leaves the reply alone', async () => {
    // The VAD only reports sustained speech once it has lasted the minimum
    // (see vad.sustained.test.ts); a bare speech-start is not a barge-in.
    const onTranscript = vi.fn(async () => {
      void playText('x'.repeat(100))
    })
    await startConversation({ onTranscript, bargeIn: true })
    await say('hi')
    await waitFor(() => expect(play).toHaveBeenCalled())
    act(() => events.onSpeechStart())
    expect(audio[0].pause).not.toHaveBeenCalled()
  })

  it('hands the next utterance how much of the interrupted reply was heard', async () => {
    const onTranscript = replyToFirstOnly()
    await startConversation({ onTranscript, bargeIn: true })
    await say('tell me about saturday')
    await waitFor(() => expect(play).toHaveBeenCalled())

    audio[0].duration = 4
    audio[0].currentTime = 1
    act(() => events.onSustainedSpeech!())
    await act(async () => {}) // the cut reply settles; the mic carries on

    await say('no, sunday')
    await waitFor(() => expect(onTranscript).toHaveBeenCalledTimes(2))
    expect(onTranscript.mock.calls[1]).toEqual([
      'no, sunday',
      undefined,
      { playedMs: 1000, totalMs: 4000, heardChars: 25, totalChars: 100 },
    ])
  })

  it('hands over nothing for a reply that played to the end', async () => {
    const onTranscript = replyToFirstOnly()
    await startConversation({ onTranscript, bargeIn: true })
    await say('tell me about saturday')
    await waitFor(() => expect(play).toHaveBeenCalled())
    audio[0].duration = 4
    audio[0].currentTime = 4
    await act(async () => audio[0].onended?.())

    await say('thanks')
    await waitFor(() => expect(onTranscript).toHaveBeenCalledTimes(2))
    // Two arguments, exactly as before barge-in existed.
    expect(onTranscript.mock.calls[1]).toEqual(['thanks', undefined])
  })

  it('keeps the report for a real utterance when the barge-in transcribes to nothing', async () => {
    const onTranscript = replyToFirstOnly()
    await startConversation({ onTranscript, bargeIn: true })
    await say('tell me about saturday')
    await waitFor(() => expect(play).toHaveBeenCalled())
    audio[0].duration = 4
    audio[0].currentTime = 2
    act(() => events.onSustainedSpeech!())
    await act(async () => {})

    await say('   ') // a misfire: nothing to send
    expect(onTranscript).toHaveBeenCalledTimes(1)
    await say('wait, saturday')
    await waitFor(() => expect(onTranscript).toHaveBeenCalledTimes(2))
    expect(onTranscript.mock.calls[1][2]).toMatchObject({ playedMs: 2000, heardChars: 50 })
  })
})

describe('without bargeIn (the default)', () => {
  it('behaves as before: half-duplex, no sustained hook, nothing extra delivered', async () => {
    const onTranscript = replyToFirstOnly()
    await startConversation({ onTranscript })
    expect(createVad.mock.calls[0][0]).not.toHaveProperty('onSustainedSpeech')

    await say('hi')
    await waitFor(() => expect(play).toHaveBeenCalled())
    // The mic stays shut for the whole reply.
    await act(async () => {})
    expect(resume).not.toHaveBeenCalled()
    await act(async () => audio[0].onended?.())
    await waitFor(() => expect(resume).toHaveBeenCalled())

    await say('next')
    await waitFor(() => expect(onTranscript).toHaveBeenCalledTimes(2))
    expect(onTranscript.mock.calls[1]).toEqual(['next', undefined])
  })
})
