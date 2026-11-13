import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { getConversation, getTimeline, getTurnActivity, postMessage } from '../../api/agent'
import type { AgentMessage, ConversationDetail, MessageExchange } from '../../types/agent'
import { stopPlayback } from '../../voice/tts'
import type { SpeechEvents, Vad } from '../../voice/vad'
import { AgentPage } from './AgentPage'

// Barge-in on the page (#77). The vendored MicButton and tts.ts are the REAL
// modules over a fake <audio> element and fetch, as the voice client's own
// barge-in tests run them; only the VAD (no AudioWorklet in jsdom),
// transcription and the conductor API are stubbed. Real audio (does the
// speaker's own voice trip the mic?) is not something jsdom can judge.

const transcribe = vi.hoisted(() => vi.fn())
const createVad = vi.hoisted(() => vi.fn())
vi.mock('../../voice/api', () => ({ transcribe }))
vi.mock('../../voice/vad', () => ({ createVad }))
vi.mock('../../voice/wav', () => ({ encodeWav: () => new Blob(['wav']) }))
vi.mock('../../voice/tts', async () => ({
  ...(await vi.importActual<typeof import('../../voice/tts')>('../../voice/tts')),
  unlockAudio: vi.fn(),
}))
vi.mock('../../api/sort', () => ({ getWorklist: vi.fn(), openGroup: vi.fn(), fileSongs: vi.fn() }))
vi.mock('../../api/agent', () => ({
  listThreads: vi.fn(() => Promise.resolve([])),
  cancelTurn: vi.fn(),
  getConversation: vi.fn(),
  getTimeline: vi.fn(),
  getTurnActivity: vi.fn(),
  postMessage: vi.fn(),
  turnEventsUrl: (id: number) => `/api/agent/conversations/${id}/events`,
}))

class FakeURL extends URL {
  static createObjectURL = vi.fn(() => 'blob:fake')
  static revokeObjectURL = vi.fn()
}

const play = vi.fn()
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

const vad: Vad = { pause: vi.fn(), resume: vi.fn(), destroy: vi.fn() }
let events: SpeechEvents

class FakeMediaRecorder {}

const REPLY = 'x'.repeat(100)

const message = (over: Partial<AgentMessage>): AgentMessage => ({
  id: 1,
  conversation_id: 1,
  role: 'user',
  content: 'hello',
  tool_calls: null,
  stop_reason: null,
  created_at: new Date().toISOString(),
  ...over,
})

const detail: ConversationDetail = {
  id: 1,
  title: 'Timeline',
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  messages: [],
}

const exchange = (words: string, reply: string): MessageExchange => ({
  user_message: message({ id: 10, content: words, surface: 'voice' }),
  assistant_message: message({ id: 11, role: 'assistant', content: reply, stop_reason: 'completed', surface: 'voice' }),
})

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  if (audio[0]) Object.assign(audio[0], { src: '', currentTime: 0, duration: NaN, onended: null })
  play.mockReset().mockResolvedValue(undefined)
  vi.stubGlobal('Audio', FakeAudio)
  vi.stubGlobal('URL', FakeURL)
  vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob([new Uint8Array([1])]))))
  vi.stubGlobal('MediaRecorder', FakeMediaRecorder)
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: vi.fn() } })
  createVad.mockReset().mockImplementation(async (e: SpeechEvents) => {
    events = e
    return vad
  })
  vi.mocked(getTimeline).mockResolvedValue({ ...detail, quiet_gap_seconds: 10_800, incognito_notice: 'Incognito.' })
  vi.mocked(getConversation).mockResolvedValue(detail)
  vi.mocked(getTurnActivity).mockResolvedValue({ active: false, kind: null, tool: null, iteration: null, elapsed_seconds: null })
  vi.mocked(postMessage).mockImplementation(async (_id, words) => exchange(words, REPLY))
})

afterEach(() => {
  stopPlayback()
  vi.unstubAllGlobals()
})

async function openPage() {
  render(
    <MemoryRouter>
      <AgentPage />
    </MemoryRouter>,
  )
  await waitFor(() => expect(getConversation).toHaveBeenCalled())
}

async function startTalking() {
  fireEvent.click(screen.getByRole('button', { name: /start voice conversation/i }))
  await waitFor(() => expect(createVad).toHaveBeenCalled())
  await act(async () => {})
}

async function say(text: string) {
  transcribe.mockResolvedValueOnce(text)
  await act(async () => {
    events.onSpeechEnd(new Float32Array(16))
  })
}

/** The first reply plays; the person speaks over it a quarter of the way in. */
async function cutTheReplyAQuarterIn() {
  await say('what is on saturday')
  await waitFor(() => expect(play).toHaveBeenCalled())
  audio[0].duration = 4
  audio[0].currentTime = 1
  act(() => events.onSustainedSpeech!())
  await act(async () => {})
}

describe('AgentPage: barge-in (#77)', () => {
  it('is on by default: talking over a reply stops it and the next turn says how much was heard', async () => {
    await openPage()
    expect(screen.getByRole('button', { name: 'Talk over replies' })).toHaveAttribute('aria-pressed', 'true')
    await startTalking()
    expect(createVad.mock.calls[0][0].onSustainedSpeech).toBeTypeOf('function')

    await cutTheReplyAQuarterIn()
    expect(audio[0].pause).toHaveBeenCalled()
    expect(postMessage).toHaveBeenNthCalledWith(1, 1, 'what is on saturday', { surface: 'voice' })

    await say('no, sunday')
    await waitFor(() => expect(postMessage).toHaveBeenCalledTimes(2))
    expect(postMessage).toHaveBeenNthCalledWith(2, 1, 'no, sunday', {
      surface: 'voice',
      heard: { playedMs: 1000, totalMs: 4000, heardChars: 25, totalChars: 100 },
    })
  })

  it('turned off, the reply plays to the end and stays off after a reload', async () => {
    await openPage()
    const toggle = screen.getByRole('button', { name: 'Talk over replies' })
    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-pressed', 'false')
    expect(localStorage.getItem('agent-barge-in')).toBe('off')
    await startTalking()
    expect(createVad.mock.calls[0][0].onSustainedSpeech).toBeUndefined()
  })

  it('never sends the report with an incognito turn', async () => {
    await openPage()
    await startTalking()
    await cutTheReplyAQuarterIn()
    fireEvent.click(screen.getByRole('button', { name: 'Incognito' }))
    await say('no, sunday')
    await waitFor(() => expect(postMessage).toHaveBeenCalledTimes(2))
    expect(postMessage).toHaveBeenNthCalledWith(2, 1, 'no, sunday', { surface: 'voice', incognito: true })
  })
})
