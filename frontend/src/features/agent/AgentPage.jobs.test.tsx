import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cancelJob, getConversation, getTimeline } from '../../api/agent'
import type { AgentMessage, ConversationDetail, JobCancelResult, JobEvent } from '../../types/agent'
import { AgentPage } from './AgentPage'

// Jobs in the UI (#90): live cards from the stream, Cancel from the card, and
// the finished note landing in the thread.

vi.mock('../../api/sort', () => ({ getWorklist: vi.fn(), openGroup: vi.fn(), fileSongs: vi.fn() }))
vi.mock('../../api/agent', () => ({
  cancelJob: vi.fn(),
  cancelTurn: vi.fn(),
  getConversation: vi.fn(),
  getTurnActivity: vi.fn(),
  getTimeline: vi.fn(),
  postMessage: vi.fn(),
  turnEventsUrl: (id: number) => `/api/agent/conversations/${id}/events`,
}))

const mockGet = vi.mocked(getConversation)
const mockCancel = vi.mocked(cancelJob)

const message = (over: Partial<AgentMessage>): AgentMessage => ({
  id: 1,
  conversation_id: 1,
  role: 'user',
  content: 'grab this whole playlist',
  tool_calls: null,
  stop_reason: null,
  created_at: '2026-11-06T18:00:00Z',
  ...over,
})
const queued = message({ id: 2, role: 'assistant', content: 'Queued your playlist: 40 tracks.', stop_reason: 'completed' })
const detail = (...more: AgentMessage[]): ConversationDetail => ({
  id: 1,
  title: 'playlist',
  created_at: '2026-11-06T18:00:00Z',
  updated_at: '2026-11-06T18:00:00Z',
  messages: [message({}), queued, ...more],
})
const job = (over: Partial<JobEvent> = {}): JobEvent => ({ app: 'music', ref: '118', status: 'submitted', done: 0, total: 40, title: 'Late Night Drive', untrusted: true, ended: false, ...over })

/** A stand-in for the browser's EventSource: the test plays the server. */
class FakeEventSource {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSED = 2
  static instances: FakeEventSource[] = []
  readyState = FakeEventSource.CONNECTING
  onerror: ((event: Event) => void) | null = null
  private listeners = new Map<string, ((event: MessageEvent<string>) => void)[]>()
  constructor() {
    FakeEventSource.instances.push(this)
  }
  addEventListener(name: string, listener: (event: MessageEvent<string>) => void) {
    this.listeners.set(name, [...(this.listeners.get(name) ?? []), listener])
  }
  close() {
    this.readyState = FakeEventSource.CLOSED
  }
  emit(name: string, data: unknown, id: number) {
    this.readyState = FakeEventSource.OPEN
    const event = new MessageEvent(name, { data: JSON.stringify(data), lastEventId: String(id) })
    for (const listener of this.listeners.get(name) ?? []) listener(event)
  }
}

const stream = () => FakeEventSource.instances.filter((es) => es.readyState !== FakeEventSource.CLOSED).at(-1)!

beforeEach(() => {
  vi.clearAllMocks()
  FakeEventSource.instances = []
  vi.stubGlobal('EventSource', FakeEventSource)
  vi.mocked(getTimeline).mockResolvedValue({ ...detail(), messages: [], quiet_gap_seconds: 10_800, incognito_notice: 'Incognito.' })
  mockGet.mockResolvedValue(detail())
})
afterEach(() => vi.unstubAllGlobals())

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/c/1']}>
      <Routes>
        <Route path="/c/:conversationId" element={<AgentPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('AgentPage: jobs in the UI (#90)', () => {
  it('finds a running job after a reload and shows queued, then working, from the events', async () => {
    renderPage()
    await screen.findByText('Queued your playlist: 40 tracks.')
    expect(screen.queryByRole('group', { name: /Job:/ })).toBeNull()

    act(() => stream().emit('snapshot', { active: false, jobs: [job()] }, 5))
    const card = screen.getByRole('group', { name: 'Job: Late Night Drive' })
    expect(card).toHaveTextContent('queued · 0 of 40')
    expect(card).not.toHaveTextContent('working')

    act(() => stream().emit('job', job({ status: 'working', done: 12 }), 6))
    expect(screen.getByRole('group', { name: 'Job: Late Night Drive' })).toHaveTextContent('working · 12 of 40')
    // The change is said once, in words, for a screen reader.
    expect(screen.getAllByRole('status').map((el) => el.textContent)).toContain('music: “Late Night Drive”, working, 12 of 40.')
  })

  it('Cancel on the card asks the app to stop, shows asking then asked, and the person’s note joins the thread', async () => {
    renderPage()
    await screen.findByText('Queued your playlist: 40 tracks.')
    act(() => stream().emit('snapshot', { active: false, jobs: [job({ status: 'working', done: 12 })] }, 5))

    const stopped = message({ id: 3, content: 'Asked music to stop “Late Night Drive” from its card: 12 of 40 done when I asked.' })
    let answer: () => void = () => {}
    mockCancel.mockReturnValue(new Promise<JobCancelResult>((resolve) => (answer = () => resolve({} as JobCancelResult))))
    mockGet.mockResolvedValue(detail(stopped))
    fireEvent.click(screen.getByRole('button', { name: 'Stop Late Night Drive' }))
    expect(mockCancel).toHaveBeenCalledWith(1, 'music', '118')
    expect(screen.getByRole('button', { name: 'Stop Late Night Drive' })).toHaveTextContent('Asking…')

    await act(async () => answer())
    expect(await screen.findByText(stopped.content!)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Stop Late Night Drive' })).toHaveTextContent('Asked to stop')
    expect(screen.getByRole('group', { name: 'Job: Late Night Drive' })).toHaveTextContent('stopping · 12 of 40')
  })

  it('when the job ends its note appears in the thread, naming what failed, and the card goes', async () => {
    renderPage()
    await screen.findByText('Queued your playlist: 40 tracks.')
    act(() => stream().emit('snapshot', { active: false, jobs: [job({ status: 'working', done: 20 })] }, 5))

    const note = message({
      id: 4,
      role: 'assistant',
      content: 'Your playlist is in: 38 of 40. “Lost Signal” and “Paper Moons” couldn’t be found.',
      stop_reason: 'job_note',
    })
    mockGet.mockResolvedValue(detail(note))
    act(() => stream().emit('job', job({ status: 'completed', done: 38, ended: true, how: 'ended', note_message_id: 4 }), 6))

    const block = await screen.findByRole('complementary', { name: 'Job note' })
    expect(block).toHaveTextContent('“Lost Signal” and “Paper Moons” couldn’t be found.')
    await waitFor(() => expect(screen.queryByRole('group', { name: 'Job: Late Night Drive' })).toBeNull())
    // The note is a note, not an answer: the exchange above it is still the latest turn.
    expect(within(screen.getByRole('article', { name: 'Latest turn' })).getByText('Queued your playlist: 40 tracks.')).toBeInTheDocument()
  })

  it('several jobs at once are one stack, not several cards', async () => {
    renderPage()
    await screen.findByText('Queued your playlist: 40 tracks.')
    act(() =>
      stream().emit(
        'snapshot',
        { active: false, jobs: [job({ ref: '1', title: 'Road trip' }), job({ ref: '2', title: 'Library sort', total: 1480 }), job({ ref: '3', title: 'Late Night Drive' })] },
        5,
      ),
    )
    const stack = screen.getByRole('region', { name: 'Background jobs' })
    expect(within(stack).getByText('3 jobs in the background')).toBeInTheDocument()
    expect(within(stack).getAllByRole('button', { name: /^Stop / })).toHaveLength(3)
  })
})
