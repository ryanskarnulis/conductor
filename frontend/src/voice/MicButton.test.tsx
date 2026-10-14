import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { SpeechEvents, Vad } from './vad'
import { MicButton } from './MicButton'
import { playText } from './tts'

// The vendored MicButton talks to the browser through vad/tts/wav; jsdom has
// none of the audio machinery, so those seams are stubbed and only the STT
// fetch is exercised for real (#24: a failed STT request must not strand
// the button).
const createVad = vi.hoisted(() => vi.fn())
const pause = vi.hoisted(() => vi.fn())
const resume = vi.hoisted(() => vi.fn())
const destroy = vi.hoisted(() => vi.fn())

vi.mock('./vad', () => ({ createVad }))
// tts is real apart from unlockAudio (jsdom has no media playback and the
// unlock is a pure user-gesture side effect): chess#141 is about audioIdle()
// staying resolvable when the TTS fetch fails, which only the real module
// exercises.
vi.mock('./tts', async () => ({
  ...(await vi.importActual<typeof import('./tts')>('./tts')),
  unlockAudio: vi.fn(),
}))
vi.mock('./wav', () => ({ encodeWav: () => new Blob(['wav']) }))

const vad: Vad = { pause, resume, destroy }

class FakeMediaRecorder {
  static instance: FakeMediaRecorder | null = null
  state = 'inactive'
  mimeType = 'audio/webm'
  ondataavailable: ((e: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  constructor() {
    FakeMediaRecorder.instance = this
  }
  start() {
    this.state = 'recording'
  }
  stop() {
    this.state = 'inactive'
    this.ondataavailable?.({ data: new Blob(['clip']) })
    this.onstop?.()
  }
}

function stubMediaDevices() {
  const stream = { getTracks: () => [{ stop: vi.fn() }] } as unknown as MediaStream
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: vi.fn(async () => stream) },
  })
}

beforeEach(() => {
  vi.stubGlobal('MediaRecorder', FakeMediaRecorder)
  stubMediaDevices()
  createVad.mockReset()
  pause.mockReset()
  resume.mockReset()
  destroy.mockReset()
  FakeMediaRecorder.instance = null
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('MicButton STT transport failures', () => {
  it('hands-free: a rejected transcribe fetch ends the session instead of hanging in working', async () => {
    const user = userEvent.setup()
    let events: SpeechEvents | undefined
    createVad.mockImplementation(async (e: SpeechEvents) => {
      events = e
      return vad
    })
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))),
    )
    const onTranscript = vi.fn()

    render(<MicButton onTranscript={onTranscript} disabled={false} />)
    await user.click(screen.getByRole('button'))
    await waitFor(() => expect(screen.getByRole('button')).toHaveClass('mic-listening'))

    events?.onSpeechEnd(new Float32Array(16))

    await screen.findByRole('alert')
    expect(screen.getByText('Voice input is unavailable.')).toBeInTheDocument()
    // Back to an actionable state, with the VAD torn down rather than left paused.
    expect(screen.getByRole('button')).toHaveClass('mic-idle')
    expect(screen.getByRole('button')).not.toBeDisabled()
    expect(destroy).toHaveBeenCalled()
    expect(onTranscript).not.toHaveBeenCalled()
  })

  it('push-to-talk: a rejected transcribe fetch returns the button to idle, not stuck transcribing', async () => {
    const user = userEvent.setup()
    createVad.mockResolvedValue(null) // forces the MediaRecorder fallback
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))),
    )
    const onTranscript = vi.fn()

    render(<MicButton onTranscript={onTranscript} disabled={false} />)
    await user.click(screen.getByRole('button'))
    await waitFor(() => expect(screen.getByRole('button')).toHaveClass('mic-recording'))

    await user.click(screen.getByRole('button')) // stop → onstop → transcribe

    await screen.findByRole('alert')
    expect(screen.getByText('Voice input is unavailable.')).toBeInTheDocument()
    expect(screen.getByRole('button')).toHaveClass('mic-idle')
    expect(screen.getByRole('button')).not.toBeDisabled()
    expect(onTranscript).not.toHaveBeenCalled()
  })
})

describe('MicButton TTS transport failures', () => {
  it('hands-free: a rejected speak fetch still returns the mic to listening', async () => {
    const user = userEvent.setup()
    let events: SpeechEvents | undefined
    createVad.mockImplementation(async (e: SpeechEvents) => {
      events = e
      return vad
    })
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL) => {
        if (String(input).includes('/api/voice/speak')) {
          return Promise.reject(new TypeError('Failed to fetch'))
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ text: 'add milk' }),
        } as Response)
      }),
    )
    // The typed/voice reply path: fire-and-forget playText, then audioIdle().
    const onTranscript = vi.fn(() => {
      void playText('sure, added it')
    })

    render(<MicButton onTranscript={onTranscript} disabled={false} />)
    await user.click(screen.getByRole('button'))
    await waitFor(() => expect(screen.getByRole('button')).toHaveClass('mic-listening'))
    resume.mockClear()

    events?.onSpeechEnd(new Float32Array(16))

    await waitFor(() => expect(onTranscript).toHaveBeenCalledWith('add milk', undefined))
    // Not wedged in 'working' with the VAD paused.
    await waitFor(() => expect(screen.getByRole('button')).toHaveClass('mic-listening'))
    expect(resume).toHaveBeenCalled()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})

describe('MicButton agent-turn failures', () => {
  it('hands-free: an agent turn that rejects ends the session instead of hanging in working', async () => {
    // Conductor's turn path resolves null rather than reject, but MicButton
    // doesn't own onTranscript's promise — chess#232: an escape here used to
    // leave the mic on "…" with the VAD paused and nothing left to reopen it.
    const user = userEvent.setup()
    let events: SpeechEvents | undefined
    createVad.mockImplementation(async (e: SpeechEvents) => {
      events = e
      return vad
    })
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ text: 'add milk' }),
        } as Response),
      ),
    )
    const onTranscript = vi.fn(() => Promise.reject(new Error('turn died')))

    render(<MicButton onTranscript={onTranscript} disabled={false} />)
    await user.click(screen.getByRole('button'))
    await waitFor(() => expect(screen.getByRole('button')).toHaveClass('mic-listening'))

    events?.onSpeechEnd(new Float32Array(16))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Voice conversation stopped — tap to start again.',
    )
    expect(screen.getByRole('button')).toHaveClass('mic-idle')
    expect(screen.getByRole('button')).not.toBeDisabled()
    expect(destroy).toHaveBeenCalled()
  })
})

describe('MicButton icons', () => {
  it('draws the mic as a monochrome inline icon, never a color emoji', () => {
    render(<MicButton onTranscript={vi.fn()} disabled={false} />)
    expect(document.body.textContent).not.toMatch(/\p{Extended_Pictographic}/u)
    const icon = screen
      .getByRole('button', { name: 'Start voice conversation' })
      .querySelector('svg')
    expect(icon).toHaveAttribute('aria-hidden', 'true')
  })
})
