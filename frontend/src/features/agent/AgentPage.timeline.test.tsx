import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getConversation, getTimeline, getTurnActivity, postMessage } from '../../api/agent'
import type { AgentMessage, ConversationDetail, MessageExchange } from '../../types/agent'
import { searchTimeline } from '../timeline/timelineApi'
import { AgentPage } from './AgentPage'

// The one timeline (#82): day separators, surface tags, incognito, search.

vi.mock('../../api/sort', () => ({ getWorklist: vi.fn(), openGroup: vi.fn(), fileSongs: vi.fn() }))
vi.mock('../timeline/timelineApi', () => ({ searchTimeline: vi.fn() }))
vi.mock('../../api/agent', () => ({
  // Side threads (#139): none open.
  listThreads: vi.fn(() => Promise.resolve([])),
  cancelTurn: vi.fn(),
  getConversation: vi.fn(),
  getTimeline: vi.fn(),
  getTurnActivity: vi.fn(),
  postMessage: vi.fn(),
  turnEventsUrl: (id: number) => `/api/agent/conversations/${id}/events`,
}))

const NOTICE = 'Incognito: this turn’s words aren’t kept. Apps still see the request.'
const mockGet = vi.mocked(getConversation)
const mockPost = vi.mocked(postMessage)

const message = (over: Partial<AgentMessage>): AgentMessage => ({
  id: 1,
  conversation_id: 1,
  role: 'user',
  content: 'hello',
  tool_calls: null,
  stop_reason: null,
  created_at: '2026-11-08T15:00:00Z',
  ...over,
})

const noonYesterday = new Date()
noonYesterday.setDate(noonYesterday.getDate() - 1)
noonYesterday.setHours(12, 0, 0, 0)
const yesterday = noonYesterday.toISOString()
const today = new Date().toISOString()

const detail = (...messages: AgentMessage[]): ConversationDetail => ({
  id: 1,
  title: 'Timeline',
  created_at: yesterday,
  updated_at: today,
  messages,
})

const history = detail(
  message({ id: 1, content: 'grab Odessa by Caribou', created_at: yesterday }),
  message({ id: 2, role: 'assistant', content: 'Saved “Odessa”.', stop_reason: 'completed', created_at: yesterday }),
  message({ id: 3, content: 'what’s due today?', created_at: today, surface: 'voice' }),
  message({ id: 4, role: 'assistant', content: 'Nothing is due.', stop_reason: 'completed', created_at: today, surface: 'voice' }),
)

function renderPage() {
  return render(
    <MemoryRouter>
      <AgentPage />
    </MemoryRouter>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(getTimeline).mockResolvedValue({ ...history, quiet_gap_seconds: 10_800, incognito_notice: NOTICE })
  mockGet.mockResolvedValue(history)
  vi.mocked(getTurnActivity).mockResolvedValue({ active: false, kind: null, tool: null, iteration: null, elapsed_seconds: null })
})

describe('AgentPage: the timeline (#82)', () => {
  it('is one list with a separator for each day, each entry tagged with its surface', async () => {
    renderPage()
    await screen.findByText('Nothing is due.')
    expect(screen.getAllByRole('separator').map((el) => el.textContent)).toEqual(['Yesterday', 'Today'])
    const turns = screen.getAllByRole('article')
    expect(within(turns[0]).getByText('web')).toBeInTheDocument()
    expect(within(turns[1]).getByText('voice')).toBeInTheDocument()
    // No conversation list and nothing to delete: "forget that" is how.
    expect(screen.queryByRole('button', { name: /Delete conversation|New conversation/ })).not.toBeInTheDocument()
  })

  it('sends incognito while the switch is on, says apps still see it, and shows the live words', async () => {
    renderPage()
    await screen.findByText('Nothing is due.')
    fireEvent.click(screen.getByRole('button', { name: 'Composer options' }))
    const toggle = screen.getByRole('menuitemcheckbox', { name: /Incognito/ })
    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByText(NOTICE)).toBeInTheDocument()

    const exchange: MessageExchange = {
      user_message: message({ id: 5, content: 'plan the surprise party', created_at: today, incognito: true }),
      assistant_message: message({ id: 6, role: 'assistant', content: 'Added it.', stop_reason: 'completed', created_at: today, incognito: true }),
    }
    mockPost.mockResolvedValue(exchange)
    mockGet.mockResolvedValue(
      detail(
        ...history.messages,
        { ...exchange.user_message, content: '(incognito)' },
        { ...exchange.assistant_message, content: '(incognito)' },
      ),
    )
    fireEvent.change(screen.getByLabelText('Message conductor'), { target: { value: 'plan the surprise party' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    await waitFor(() => expect(mockPost).toHaveBeenCalledWith(1, 'plan the surprise party', { incognito: true }))
    // The server kept "(incognito)"; this page shows what the person read.
    expect(await screen.findByText('Added it.')).toBeInTheDocument()
    const latest = screen.getByRole('article', { name: 'Latest turn' })
    expect(within(latest).getByText('plan the surprise party')).toBeInTheDocument()
    expect(within(latest).getByText('incognito')).toBeInTheDocument()

    // Flipped back, the next turn is kept as usual.
    fireEvent.click(toggle)
    expect(screen.queryByText(NOTICE)).not.toBeInTheDocument()
    mockPost.mockReturnValue(new Promise(() => {}))
    fireEvent.change(screen.getByLabelText('Message conductor'), { target: { value: 'thanks' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    expect(mockPost).toHaveBeenLastCalledWith(1, 'thanks')
  })

  it('shows a stored incognito turn as "(incognito)" with its receipts', async () => {
    mockGet.mockResolvedValue(
      detail(
        message({ id: 7, content: '(incognito)', incognito: true, created_at: today }),
        message({
          id: 8,
          role: 'assistant',
          content: '(incognito)',
          incognito: true,
          stop_reason: 'completed',
          created_at: today,
          tool_calls: [
            {
              tool: 'ask_tasks',
              arguments: {},
              result: '(incognito)',
              error: null,
              app_record: [{ tool: 'create_task', outcome: 'done', touched: [{ app: 'tasks', kind: 'task', id: '9', title: 'Buy a cake' }] }],
            },
          ],
        }),
      ),
    )
    renderPage()
    const latest = await screen.findByRole('article', { name: 'Latest turn' })
    expect(within(latest).getAllByText('(incognito)')).toHaveLength(2)
    expect(within(latest).getByText(/Buy a cake/)).toBeInTheDocument()
  })

  it('searches the timeline in a lazy panel and opens a hit', async () => {
    vi.mocked(searchTimeline).mockResolvedValue([
      {
        turn_id: 1,
        message_id: 1,
        at: yesterday,
        surface: null,
        kind: 'message',
        said: 'grab Odessa by Caribou',
        reply: 'Saved “Odessa”.',
        titles: ['Odessa'],
        matched: ['words', 'reply', 'receipt'],
      },
    ])
    renderPage()
    await screen.findByText('Nothing is due.')
    fireEvent.click(screen.getByRole('button', { name: 'Search the timeline' }))
    const box = await screen.findByRole('searchbox', { name: 'Search the timeline' })
    expect(screen.getByText(/Incognito turns aren’t searched/)).toBeInTheDocument()
    fireEvent.change(box, { target: { value: 'odessa' } })
    const hit = await screen.findByRole('button', { name: /grab Odessa by Caribou/ })
    expect(vi.mocked(searchTimeline).mock.calls[0][0]).toBe('odessa')
    fireEvent.click(hit)
    expect(await screen.findByText('Nothing is due.')).toBeInTheDocument()
    expect(screen.queryByRole('searchbox')).not.toBeInTheDocument()
  })
})
