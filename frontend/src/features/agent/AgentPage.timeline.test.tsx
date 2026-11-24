import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { getConversation, getTimeline, getTurnActivity, postMessage } from '../../api/agent'
import type { AgentMessage, ConversationDetail, MessageExchange } from '../../types/agent'
import { searchTimeline } from '../timeline/timelineApi'
import { AgentPage } from './AgentPage'

// The one timeline (#82): day separators, surface tags, incognito, search;
// the fold (#226): the page opens on the current window; and reading back
// (#233): earlier turns by scrolling, "Show earlier" in batches, the place kept.

vi.mock('../../api/sort', () => ({ getWorklist: vi.fn(), openGroup: vi.fn(), fileSongs: vi.fn() }))
vi.mock('../timeline/timelineApi', () => ({ searchTimeline: vi.fn() }))
vi.mock('../../api/agent', () => ({
  // Side threads (#139): none open.
  listThreads: vi.fn(() => Promise.resolve([])),
  // Waiting on you (#251): no card open.
  getCards: vi.fn(() => Promise.resolve([])),
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
  it('opens on the current window, with what came before the quiet gap behind "Show earlier"', async () => {
    renderPage()
    await screen.findByText('Nothing is due.')
    expect(screen.queryByText('Saved “Odessa”.')).not.toBeInTheDocument()
    expect(screen.getAllByRole('separator').map((el) => el.textContent)).toEqual(['Today'])
    fireEvent.click(screen.getByRole('button', { name: 'Show earlier' }))
    expect(screen.getByText('Saved “Odessa”.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Show earlier' })).not.toBeInTheDocument()
  })

  it('folds an old side thread’s note away with its day, and it still opens from there (#226)', async () => {
    const old = message({
      id: 20,
      role: 'assistant',
      content: 'Side thread “browser check”: added “Order the shelf brackets”.',
      stop_reason: 'thread_note',
      created_at: yesterday,
      thread: { id: 62, title: 'browser check' },
    })
    mockGet.mockResolvedValue(detail(old, ...history.messages))
    renderPage()
    await screen.findByText('Nothing is due.')
    // The note from before the gap doesn't head the page...
    expect(screen.queryByRole('group', { name: 'Side thread result: browser check' })).not.toBeInTheDocument()
    expect(screen.getByRole('article', { name: 'Latest turn' })).toHaveTextContent('Nothing is due.')
    // ...and is one tap away, with its Open.
    fireEvent.click(screen.getByRole('button', { name: 'Show earlier' }))
    const note = screen.getByRole('group', { name: 'Side thread result: browser check' })
    expect(within(note).getByRole('button', { name: 'Open' })).toBeInTheDocument()
  })

  it('opens empty a quiet gap after the last turn, and keeps a note that landed since', async () => {
    const landed = message({ id: 21, role: 'assistant', content: 'Side thread “Garage”: added “Order the shelf brackets”.', stop_reason: 'thread_note', created_at: today, thread: { id: 7, title: 'Garage' } })
    mockGet.mockResolvedValue(detail(...history.messages.slice(0, 2), landed))
    renderPage()
    expect(await screen.findByRole('group', { name: 'Side thread result: Garage' })).toBeInTheDocument()
    expect(screen.queryByText('Saved “Odessa”.')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Show earlier' })).toBeInTheDocument()
  })

  it('is one list with a separator for each day, each entry tagged with its surface', async () => {
    renderPage()
    await screen.findByText('Nothing is due.')
    fireEvent.click(screen.getByRole('button', { name: 'Show earlier' }))
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

/** `count` finished turns, the oldest first, all said at `at`. */
function turnsAt(count: number, at: string, firstId = 100): AgentMessage[] {
  return Array.from({ length: count }, (_, i) => [
    message({ id: firstId + i * 2, content: `ask ${i + 1}`, created_at: at }),
    message({ id: firstId + i * 2 + 1, role: 'assistant', content: `reply ${i + 1}.`, stop_reason: 'completed', created_at: at }),
  ]).flat()
}

/** jsdom lays nothing out: give the scroll area a content height and the
 * block of earlier turns a bottom edge, the way a browser would report them. */
function layout(box: { scrollHeight: number; clientHeight: number; aboveBottom: number }) {
  const isConvo = (el: Element) => el.classList.contains('shell-convo')
  vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockImplementation(function (this: HTMLElement) {
    return isConvo(this) ? box.scrollHeight : 0
  })
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockImplementation(function (this: HTMLElement) {
    return isConvo(this) ? box.clientHeight : 0
  })
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    const bottom = this.classList.contains('tl-above') ? box.aboveBottom - (this.parentElement?.scrollTop ?? 0) : 0
    return { top: 0, bottom, left: 0, right: 0, width: 0, height: 0, x: 0, y: 0, toJSON: () => ({}) }
  })
}

function onPhone() {
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: query.includes('max-width'),
    addEventListener: () => {},
    removeEventListener: () => {},
  }))
}

describe('AgentPage: reading back (#233)', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('on a phone, has the window’s earlier turns above the latest, the sheet fitted to the latest', async () => {
    onPhone()
    layout({ scrollHeight: 900, clientHeight: 300, aboveBottom: 600 })
    mockGet.mockResolvedValue(detail(...turnsAt(3, today)))
    renderPage()
    const latest = await screen.findByRole('article', { name: 'Latest turn' })
    expect(latest).toHaveTextContent('reply 3.')
    const convo = screen.getByLabelText('Timeline')
    // Both earlier turns are in the scroll area, in one block before the latest, with no button.
    const earlier = within(convo).getAllByRole('article', { name: 'Earlier turn' })
    expect(earlier).toHaveLength(2)
    expect(earlier[0]).toHaveTextContent('reply 1.')
    expect(earlier[1]).toHaveTextContent('reply 2.')
    const block = convo.querySelector('.tl-above')
    expect(earlier.every((el) => block?.contains(el))).toBe(true)
    expect(block?.contains(latest)).toBe(false)
    expect(screen.queryByRole('button', { name: 'Show earlier' })).not.toBeInTheDocument()
    // The sheet is as tall as what follows the block, opened at the end.
    expect(convo.style.maxHeight).toBe('300px')
    expect(convo.scrollTop).toBe(900)

    // Scrolling up by hand gives the sheet its full height.
    fireEvent.touchMove(convo)
    convo.scrollTop = 500
    fireEvent.scroll(convo)
    expect(convo.style.maxHeight).toBe('')
  })

  it('on a phone, a new turn fits the sheet to it again, at the end', async () => {
    onPhone()
    layout({ scrollHeight: 900, clientHeight: 300, aboveBottom: 600 })
    mockGet.mockResolvedValue(detail(...turnsAt(3, today)))
    renderPage()
    await screen.findByRole('article', { name: 'Latest turn' })
    const convo = screen.getByLabelText('Timeline')
    fireEvent.touchMove(convo)
    convo.scrollTop = 100
    fireEvent.scroll(convo)
    expect(convo.style.maxHeight).toBe('')

    mockPost.mockReturnValue(new Promise(() => {}))
    fireEvent.change(screen.getByLabelText('Message conductor'), { target: { value: 'and now?' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    await waitFor(() => expect(screen.getByRole('article', { name: 'Latest turn' })).toHaveTextContent('and now?'))
    expect(convo.style.maxHeight).not.toBe('')
    expect(convo.scrollTop).toBe(900)
    expect(within(convo).getAllByRole('article', { name: 'Earlier turn' })).toHaveLength(3)
  })

  it('"Show earlier" brings back a batch at a time and stays while more remains', async () => {
    mockGet.mockResolvedValue(detail(...turnsAt(25, yesterday), ...history.messages.slice(2)))
    renderPage()
    await screen.findByText('Nothing is due.')
    expect(screen.queryAllByRole('article', { name: 'Earlier turn' })).toHaveLength(0)
    fireEvent.click(screen.getByRole('button', { name: 'Show earlier' }))
    // The newest ten of the folded turns, in order, the button above them.
    const shown = screen.getAllByRole('article', { name: 'Earlier turn' })
    expect(shown).toHaveLength(10)
    expect(shown[0]).toHaveTextContent('reply 16.')
    expect(shown[9]).toHaveTextContent('reply 25.')
    const button = screen.getByRole('button', { name: 'Show earlier' })
    expect(button.compareDocumentPosition(shown[0]) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    fireEvent.click(button)
    expect(screen.getAllByRole('article', { name: 'Earlier turn' })).toHaveLength(20)
    fireEvent.click(screen.getByRole('button', { name: 'Show earlier' }))
    expect(screen.getAllByRole('article', { name: 'Earlier turn' })).toHaveLength(25)
    expect(screen.queryByRole('button', { name: 'Show earlier' })).not.toBeInTheDocument()
  })

  it('"Show earlier" keeps the place: what was on screen stays on screen', async () => {
    // Each earlier turn on the page adds 100px above the latest.
    layout({
      get scrollHeight() {
        return 500 + 100 * document.querySelectorAll('.shell-turn--old').length
      },
      clientHeight: 300,
      aboveBottom: 0,
    })
    mockGet.mockResolvedValue(detail(...turnsAt(25, yesterday), ...history.messages.slice(2)))
    renderPage()
    await screen.findByText('Nothing is due.')
    const convo = screen.getByLabelText('Timeline')
    convo.scrollTop = 40
    // The batch of ten adds 1000px above: the view moves down with it.
    fireEvent.click(screen.getByRole('button', { name: 'Show earlier' }))
    expect(convo.scrollTop).toBe(1040)
  })

  it('a search hit unfolds as far as its turn', async () => {
    vi.mocked(searchTimeline).mockResolvedValue([
      { turn_id: 1, message_id: 140, at: yesterday, surface: null, kind: 'message', said: 'ask 21', reply: 'reply 21.', titles: [], matched: ['words'] },
    ])
    mockGet.mockResolvedValue(detail(...turnsAt(25, yesterday), ...history.messages.slice(2)))
    renderPage()
    await screen.findByText('Nothing is due.')
    fireEvent.click(screen.getByRole('button', { name: 'Search the timeline' }))
    fireEvent.change(await screen.findByRole('searchbox', { name: 'Search the timeline' }), { target: { value: 'ask' } })
    fireEvent.click(await screen.findByRole('button', { name: /ask 21/ }))
    const shown = await screen.findAllByRole('article', { name: 'Earlier turn' })
    expect(shown).toHaveLength(5)
    expect(shown[0]).toHaveTextContent('reply 21.')
    expect(screen.getByRole('button', { name: 'Show earlier' })).toBeInTheDocument()
  })
})
