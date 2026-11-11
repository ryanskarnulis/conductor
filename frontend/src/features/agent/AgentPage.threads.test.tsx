import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { closeThread, getConversation, getThread, getTimeline, getTurnActivity, listThreads, postMessage, startThread } from '../../api/agent'
import type { AgentMessage, ConversationDetail, ThreadClosed, ThreadInfo } from '../../types/agent'
import { AgentPage } from './AgentPage'

// Side threads (#139): start one from the page, work in it, close it, and
// its result shows in the timeline.

vi.mock('../../api/sort', () => ({ getWorklist: vi.fn(), openGroup: vi.fn(), fileSongs: vi.fn() }))
vi.mock('../../api/agent', () => ({
  cancelTurn: vi.fn(),
  closeThread: vi.fn(),
  getConversation: vi.fn(),
  getThread: vi.fn(),
  getTimeline: vi.fn(),
  getTurnActivity: vi.fn(),
  listThreads: vi.fn(),
  postMessage: vi.fn(),
  startThread: vi.fn(),
  turnEventsUrl: (id: number) => `/api/agent/conversations/${id}/events`,
}))

const NOTICE = 'Incognito: this turn’s words aren’t kept. Apps still see the request.'
const now = new Date().toISOString()
const RESULT = 'Side thread “Garage shelves”: added “order the shelf brackets”.'

const message = (over: Partial<AgentMessage>): AgentMessage => ({
  id: 1,
  conversation_id: 1,
  role: 'user',
  content: 'hello',
  tool_calls: null,
  stop_reason: null,
  created_at: now,
  ...over,
})

const detail = (id: number, title: string, ...messages: AgentMessage[]): ConversationDetail => ({ id, title, created_at: now, updated_at: now, messages })

const timeline = detail(
  1,
  'Timeline',
  message({ id: 1, content: 'add a task to measure the garage wall' }),
  message({ id: 2, role: 'assistant', content: 'Added “measure the garage wall”.', stop_reason: 'completed' }),
)
const note = message({ id: 9, role: 'assistant', content: RESULT, stop_reason: 'thread_note', thread: { id: 7, title: 'Garage shelves' } })

const garage: ThreadInfo = {
  id: 7,
  title: 'Garage shelves',
  created_at: now,
  updated_at: now,
  closed_at: null,
  result: null,
  carries: [
    { type: 'entity', app: 'tasks', kind: 'task', title: 'measure the garage wall', untrusted: false },
    { type: 'fact', text: 'the shelves are 28 inches deep' },
  ],
}

let threadMessages: AgentMessage[] = []
let timelineMessages: AgentMessage[] = timeline.messages

function renderPage() {
  return render(
    <MemoryRouter>
      <AgentPage />
    </MemoryRouter>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  threadMessages = []
  timelineMessages = timeline.messages
  vi.mocked(getTimeline).mockResolvedValue({ ...timeline, quiet_gap_seconds: 10_800, incognito_notice: NOTICE })
  vi.mocked(getConversation).mockImplementation((id: number) =>
    Promise.resolve(id === 7 ? detail(7, 'Garage shelves', ...threadMessages) : detail(1, 'Timeline', ...timelineMessages)),
  )
  vi.mocked(getTurnActivity).mockResolvedValue({ active: false, kind: null, tool: null, iteration: null, elapsed_seconds: null })
  vi.mocked(listThreads).mockResolvedValue([])
  vi.mocked(startThread).mockResolvedValue(garage)
})

async function startGarage() {
  renderPage()
  await screen.findByText('Added “measure the garage wall”.')
  fireEvent.click(screen.getByRole('button', { name: 'Side thread' }))
  fireEvent.change(screen.getByLabelText('Name (optional)'), { target: { value: 'Garage shelves' } })
  fireEvent.click(screen.getByRole('button', { name: 'Start' }))
  return screen.findByRole('region', { name: 'Side thread: Garage shelves' })
}

describe('AgentPage: side threads (#139)', () => {
  it('starts a thread that shows what it carries and takes the turns', async () => {
    const head = await startGarage()
    expect(startThread).toHaveBeenCalledWith('Garage shelves', false)
    expect(within(head).getByText('measure the garage wall')).toBeInTheDocument()
    expect(within(head).getByText('the shelves are 28 inches deep')).toBeInTheDocument()
    // The thread is on the page: its own (empty) conversation, not the timeline's.
    await waitFor(() => expect(getConversation).toHaveBeenCalledWith(7))
    expect(screen.queryByText('Added “measure the garage wall”.')).not.toBeInTheDocument()
    expect(screen.getByText(/A side thread starts with only what it carries/)).toBeInTheDocument()
    const tabs = screen.getByRole('navigation', { name: 'Timeline and side threads' })
    expect(within(tabs).getByRole('button', { name: 'Garage shelves' })).toHaveAttribute('aria-current', 'page')

    const said = message({ id: 3, conversation_id: 7, content: 'add a task to order the shelf brackets' })
    const answered = message({ id: 4, conversation_id: 7, role: 'assistant', content: 'Added “order the shelf brackets”.', stop_reason: 'completed' })
    vi.mocked(postMessage).mockImplementation(() => {
      threadMessages = [said, answered]
      return Promise.resolve({ user_message: said, assistant_message: answered })
    })
    fireEvent.change(screen.getByLabelText('Message conductor'), { target: { value: 'add a task to order the shelf brackets' } })
    expect(screen.getByLabelText('Message conductor')).toHaveAttribute('placeholder', 'Ask in this side thread…')
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    await waitFor(() => expect(postMessage).toHaveBeenCalledWith(7, 'add a task to order the shelf brackets'))
    expect(await screen.findByText('Added “order the shelf brackets”.')).toBeInTheDocument()

    // Back to the timeline without closing: the thread stays open, in its tab.
    fireEvent.click(within(tabs).getByRole('button', { name: 'Timeline' }))
    expect(await screen.findByText('Added “measure the garage wall”.')).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Side thread: Garage shelves' })).not.toBeInTheDocument()
    expect(within(screen.getByRole('navigation', { name: 'Timeline and side threads' })).getByRole('button', { name: 'Garage shelves' })).toBeInTheDocument()
  })

  it('closes into the timeline, saying closed only once the result is posted', async () => {
    await startGarage()
    let finish: (closed: ThreadClosed) => void = () => {}
    vi.mocked(closeThread).mockReturnValue(new Promise((resolve) => (finish = resolve)))
    fireEvent.click(screen.getByRole('button', { name: 'Done · back to the timeline' }))
    expect(closeThread).toHaveBeenCalledWith(7, undefined)
    // Still the thread, still open, until the server answers.
    expect(screen.getByRole('button', { name: 'Closing…' })).toBeDisabled()
    expect(screen.getByRole('region', { name: 'Side thread: Garage shelves' })).toBeInTheDocument()
    expect(screen.queryByText(RESULT)).not.toBeInTheDocument()

    timelineMessages = [...timeline.messages, note]
    await act(async () => finish({ thread: { ...garage, closed_at: now, result: RESULT }, message: note }))
    const flowed = await screen.findByRole('group', { name: 'Side thread result: Garage shelves' })
    expect(flowed).toHaveTextContent(RESULT)
    expect(screen.queryByRole('region', { name: 'Side thread: Garage shelves' })).not.toBeInTheDocument()
    expect(screen.queryByRole('navigation', { name: 'Timeline and side threads' })).not.toBeInTheDocument()
    // The result answers nothing: the last real turn is still the latest.
    expect(within(screen.getByRole('article', { name: 'Latest turn' })).getByText('Added “measure the garage wall”.')).toBeInTheDocument()
  })

  it('keeps the thread open and says why when the close fails', async () => {
    await startGarage()
    vi.mocked(closeThread).mockRejectedValue(new Error('the model stopped answering'))
    fireEvent.click(screen.getByRole('button', { name: 'Done · back to the timeline' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('The side thread is still open: the model stopped answering')
    expect(screen.getByRole('region', { name: 'Side thread: Garage shelves' })).toBeInTheDocument()
  })

  it('opens a closed thread from its result to read, never to write', async () => {
    timelineMessages = [...timeline.messages, note]
    vi.mocked(getThread).mockResolvedValue({ ...garage, closed_at: now, result: RESULT })
    renderPage()
    const flowed = await screen.findByRole('group', { name: 'Side thread result: Garage shelves' })
    fireEvent.click(within(flowed).getByRole('button', { name: 'Open' }))
    const head = await screen.findByRole('region', { name: 'Side thread: Garage shelves' })
    expect(within(head).getByText(`Closed. What flowed back: ${RESULT}`)).toBeInTheDocument()
    expect(screen.queryByLabelText('Message conductor')).not.toBeInTheDocument()
    fireEvent.click(within(head).getByRole('button', { name: 'Back to the timeline' }))
    expect(await screen.findByLabelText('Message conductor')).toBeInTheDocument()
  })

  it('starts incognito without a name', async () => {
    renderPage()
    await screen.findByText('Added “measure the garage wall”.')
    fireEvent.click(screen.getByRole('button', { name: 'Incognito' }))
    fireEvent.click(screen.getByRole('button', { name: 'Side thread' }))
    expect(screen.queryByLabelText('Name (optional)')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Start' }))
    await waitFor(() => expect(startThread).toHaveBeenCalledWith(null, true))
  })

  it('lists the threads already open', async () => {
    vi.mocked(listThreads).mockResolvedValue([garage, { ...garage, id: 8, title: 'Party', carries: [] }])
    renderPage()
    const tabs = await screen.findByRole('navigation', { name: 'Timeline and side threads' })
    expect(within(tabs).getAllByRole('button').map((b) => b.textContent)).toEqual(['Timeline', 'Garage shelves', 'Party'])
    expect(within(tabs).getByRole('button', { name: 'Timeline' })).toHaveAttribute('aria-current', 'page')
    fireEvent.click(within(tabs).getByRole('button', { name: 'Party' }))
    expect(await screen.findByRole('region', { name: 'Side thread: Party' })).toHaveTextContent('nothing: it started empty')
  })
})
