import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { answerConfirmation, cancelTurn, getConversation, getTimeline, getTurnActivity, postMessage } from '../../api/agent'
import { getWorklist } from '../../api/sort'
import type { AgentMessage, ConversationDetail } from '../../types/agent'
import { AgentPage } from './AgentPage'

vi.mock('../../api/sort', () => ({
  getWorklist: vi.fn(),
  openGroup: vi.fn(),
  fileSongs: vi.fn(),
}))

vi.mock('../memory/memoryApi', () => ({
  listFacts: vi.fn().mockResolvedValue([]),
  editFact: vi.fn(),
  forgetFact: vi.fn(),
}))

vi.mock('../../api/agent', () => ({
  // Side threads (#139): none open.
  listThreads: vi.fn(() => Promise.resolve([])),
  cancelTurn: vi.fn(),
  answerConfirmation: vi.fn(),
  getConversation: vi.fn(),
  getTurnActivity: vi.fn(),
  getTimeline: vi.fn(),
  postMessage: vi.fn(),
  turnEventsUrl: (id: number) => `/api/agent/conversations/${id}/events`,
}))

const mockTimeline = vi.mocked(getTimeline)
const mockGet = vi.mocked(getConversation)
const mockPost = vi.mocked(postMessage)
const mockAnswer = vi.mocked(answerConfirmation)
const mockActivity = vi.mocked(getTurnActivity)
const mockWorklist = vi.mocked(getWorklist)
const mockCancel = vi.mocked(cancelTurn)

function message(overrides: Partial<AgentMessage>): AgentMessage {
  return {
    id: 1,
    conversation_id: 1,
    role: 'user',
    content: 'hello',
    tool_calls: null,
    stop_reason: null,
    created_at: new Date().toISOString(),
    ...overrides,
  }
}

const assistantWithDelegateCalls = message({
  id: 2,
  role: 'assistant',
  content: 'Nothing is due today.',
  stop_reason: 'completed',
  tool_calls: [
    {
      tool: 'ask_tasks',
      arguments: { message: "what's due today?" },
      result: 'Nothing due.\n\n[tasks did: list_tasks]',
      error: null,
    },
    {
      tool: 'ask_chess',
      arguments: { message: 'status?' },
      result: null,
      error: 'chess is rate-limiting requests right now.',
    },
  ],
})

const detail: ConversationDetail = {
  id: 1,
  title: 'what tasks are due today?',
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  messages: [message({ content: 'what tasks are due today?' }), assistantWithDelegateCalls],
}

const INCOGNITO_NOTICE = 'Incognito: this turn’s words aren’t kept. Apps still see the request.'

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/" element={<AgentPage />} />
        <Route path="/c/:conversationId" element={<AgentPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  mockTimeline.mockResolvedValue({ ...detail, quiet_gap_seconds: 10_800, incognito_notice: INCOGNITO_NOTICE })
  mockGet.mockResolvedValue(detail)
  mockActivity.mockResolvedValue({
    active: false,
    kind: null,
    tool: null,
    iteration: null,
    elapsed_seconds: null,
  })
  mockWorklist.mockResolvedValue({
    filed_tracks: 0,
    filed_artist: null,
    filed_into: null,
    created_folder: false,
    not_found: [],
    opened: null,
    unsorted_tracks: 5,
    unsorted_artists: 1,
    folders: ['Dubstep', 'House'],
    next_up: [{ artist: 'Zeds Dead', tracks: 5, titles: ['Collapse'], tags_say: [] }],
  })
})

describe('AgentPage', () => {
  it('opens the memory list in the panel and closes it back to the thread', async () => {
    renderAt('/c/1')
    expect(await screen.findByText('Nothing is due today.')).toBeInTheDocument()
    const toggle = screen.getByRole('button', { name: 'Memory' })
    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'true')
    expect(await screen.findByRole('heading', { name: 'Memory' })).toBeInTheDocument()
    expect(await screen.findByText(/Nothing remembered/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Close memory' }))
    expect(await screen.findByText('Nothing is due today.')).toBeInTheDocument()
  })

  it('opens the sort panel on a turn where music ran its sorting tool', async () => {
    mockGet.mockResolvedValue({
      ...detail,
      messages: [
        message({ content: 'sort my music' }),
        message({
          id: 2,
          role: 'assistant',
          content: 'yo, Zeds Dead has 5 waiting. where they going?',
          stop_reason: 'completed',
          tool_calls: [
            {
              tool: 'ask_music',
              arguments: { message: 'sort my music' },
              result: 'five waiting',
              error: null,
              app_tools: ['sort_music'],
            },
          ],
        }),
      ],
    })

    renderAt('/c/1')

    expect(await screen.findByLabelText('Sort music')).toBeInTheDocument()
    // And typing still works while it is open — the panel is an input, not a
    // mode.
    expect(screen.getByLabelText('Message conductor')).toBeEnabled()
  })

  it('keeps the panel open through a turn that did not sort anything', async () => {
    // Saying "one at a time" out loud is a turn of its own and need not run the
    // sorting tool again. Keying the panel to the newest message made it vanish
    // mid-pass and stay gone — which is what happens the first time somebody
    // drives this by voice.
    mockGet.mockResolvedValue({
      ...detail,
      messages: [
        message({ content: 'sort my music' }),
        message({
          id: 2,
          role: 'assistant',
          content: 'yo, Zeds Dead has 5 waiting.',
          stop_reason: 'completed',
          tool_calls: [
            {
              tool: 'ask_music',
              arguments: { message: 'sort my music' },
              result: 'five waiting',
              error: null,
              app_tools: ['sort_music'],
            },
          ],
        }),
        message({ id: 3, content: 'one at a time' }),
        message({
          id: 4,
          role: 'assistant',
          content: 'aight, go song by song then',
          stop_reason: 'completed',
          tool_calls: null,
        }),
      ],
    })

    renderAt('/c/1')

    expect(await screen.findByLabelText('Sort music')).toBeInTheDocument()
  })

  it('can be dismissed and opened again without saying anything', async () => {
    mockGet.mockResolvedValue({
      ...detail,
      messages: [
        message({ content: 'sort my music' }),
        message({
          id: 2,
          role: 'assistant',
          content: 'yo, Zeds Dead has 5 waiting.',
          stop_reason: 'completed',
          tool_calls: [
            {
              tool: 'ask_music',
              arguments: {},
              result: 'five waiting',
              error: null,
              app_tools: ['sort_music'],
            },
          ],
        }),
      ],
    })

    renderAt('/c/1')
    fireEvent.click(await screen.findByLabelText('Hide the sorting panel'))
    expect(screen.queryByLabelText('Sort music')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /Back to sorting/ }))

    expect(await screen.findByLabelText('Sort music')).toBeInTheDocument()
  })

  it('leaves the panel shut on every other kind of turn', async () => {
    renderAt('/c/1')

    await screen.findByText('Nothing is due today.')
    expect(screen.queryByLabelText('Sort music')).not.toBeInTheDocument()
  })

  it('renders the thread with the delegate-call trajectory', async () => {
    renderAt('/c/1')

    expect(await screen.findByText('Nothing is due today.')).toBeInTheDocument()
    // Each delegate call is a plan step: kind, app (tasks shows as PCC), what
    // was asked, and its status as a word — a failure carries its reason.
    const steps = screen.getByRole('list', { name: 'Plan steps' })
    expect(steps).toHaveTextContent("askPCCwhat's due today?done")
    expect(steps).toHaveTextContent('askchessstatus?failed · chess is rate-limiting requests right now.')
  })

  it('sends a message, shows the live activity beat, then the refetched thread', async () => {
    let resolveRun: (value: never) => void = () => {}
    mockPost.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRun = resolve as (value: never) => void
        }),
    )
    // While the run is in flight, the poll reports a live delegate call.
    mockActivity.mockResolvedValue({
      active: true,
      kind: 'tool',
      tool: 'ask_chess',
      iteration: 2,
      elapsed_seconds: 14.5,
    })
    renderAt('/c/1')
    await screen.findByText('Nothing is due today.')

    fireEvent.change(screen.getByLabelText('Message conductor'), {
      target: { value: 'and how is the game going?' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    // Optimistic user bubble + the polled "asking chess" beat with elapsed time.
    expect(await screen.findByText('and how is the game going?')).toBeInTheDocument()
    expect(await screen.findByText(/Asking chess… · 15s/)).toBeInTheDocument()
    // Presence names the same beat, and Send has become Stop.
    expect(screen.getAllByRole('status').map((el) => el.textContent)).toContain('Asking chess')
    expect(screen.getByRole('button', { name: 'Stop' })).toBeInTheDocument()

    const followup: ConversationDetail = {
      ...detail,
      messages: [
        ...detail.messages,
        message({ id: 3, content: 'and how is the game going?' }),
        message({
          id: 4,
          role: 'assistant',
          content: 'Chess says you are up a pawn.',
          stop_reason: 'completed',
        }),
      ],
    }
    mockGet.mockResolvedValue(followup)
    resolveRun(undefined as never)

    expect(await screen.findByText('Chess says you are up a pawn.')).toBeInTheDocument()
    expect(screen.queryByText(/Asking chess…/)).not.toBeInTheDocument()
  })

  it('shows an app\'s pending question as a confirmation card and answers it with its token, not a turn', async () => {
    const waiting = message({
      id: 2,
      role: 'assistant',
      content: 'PCC wants to check: trash the Kitchen project and its 12 tasks?',
      stop_reason: 'completed',
      tool_calls: [
        {
          tool: 'ask_tasks',
          arguments: { message: 'wipe the kitchen project' },
          result: 'needs confirmation',
          error: null,
          app_record: [{ tool: 'trash_project', outcome: 'not_run', reason: 'needs confirmation' }],
          app_confirm: { app: 'tasks', thread_id: 3, token: 'tok', tool: 'trash_project', question: 'Trash the Kitchen project and its 12 tasks?' },
        },
      ],
    })
    mockGet.mockResolvedValue({ ...detail, messages: [message({ content: 'wipe the kitchen project' }), waiting] })
    mockPost.mockReturnValue(new Promise(() => {}))
    renderAt('/c/1')

    const card = await screen.findByRole('group', { name: 'Confirmation' })
    expect(card).toHaveTextContent('Waiting on you · PCC')
    expect(card).toHaveTextContent('Trash the Kitchen project and its 12 tasks?')
    expect(screen.getByRole('list', { name: 'Plan steps' })).toHaveTextContent('waiting on you')
    expect(screen.getAllByRole('status').map((el) => el.textContent)).toContain('Waiting on you')

    mockAnswer.mockReturnValue(new Promise(() => {}))
    fireEvent.click(screen.getByRole('button', { name: 'Yes, do it' }))
    expect(mockAnswer).toHaveBeenCalledWith(1, 'tok', 'yes')
    expect(mockPost).not.toHaveBeenCalled()
  })

  describe('several app questions', () => {
    const later = new Date(Date.now() + 600_000).toISOString()
    const armed = ['Test 1', 'Test 2', 'Test 3'].map((title, n) => ({
      app: 'tasks',
      thread_id: 3,
      token: `tok-${n + 1}`,
      tool: 'trash_task',
      question: `Trash the task “${title}”?`,
      expires_at: later,
    }))
    const asking = {
      tool: 'ask_tasks',
      arguments: { message: 'delete the test tasks' },
      result: 'needs confirmation',
      error: null,
      app_record: armed.map(() => ({ tool: 'trash_task', outcome: 'not_run', reason: 'needs confirmation' })),
      app_confirm: armed[0],
      app_confirms: armed,
    }
    const turn = (...answers: AgentMessage['tool_calls'] & object) =>
      ({ ...detail, messages: [message({ content: 'delete the test tasks' }), message({ id: 2, role: 'assistant', content: armed.map((a) => a.question).join('\n'), stop_reason: 'completed', tool_calls: [asking, ...answers] })] }) as ConversationDetail
    const yes = (token: string, over: object = {}) => ({ tool: 'confirm', arguments: { app: 'tasks' }, result: 'ok', error: null, answers: token, ...over })
    const trashed = { tool: 'trash_task', outcome: 'done', touched: [{ app: 'tasks', kind: 'task', id: '2', title: 'Test 2' }], undo: { handle: 'u2', how: 'restore the task' } }
    const cards = () => screen.findAllByRole('group', { name: /^Confirmation/ })

    it('shows one card per question, each with its own Yes and Not now', async () => {
      mockGet.mockResolvedValue(turn())
      renderAt('/c/1')
      const shown = await cards()
      expect(shown).toHaveLength(3)
      shown.forEach((card, n) => {
        expect(card).toHaveTextContent(armed[n].question)
        expect(within(card).getByRole('button', { name: 'Yes, do it' })).toBeEnabled()
        expect(within(card).getByRole('button', { name: 'Not now' })).toBeEnabled()
      })
    })

    it('answers one card with its token; it becomes that call’s receipt and the others stay open', async () => {
      mockGet.mockResolvedValueOnce(turn())
      mockGet.mockResolvedValue(turn(yes('tok-2', { app_record: [trashed] })))
      mockAnswer.mockResolvedValue({ token: 'tok-2', answer: 'yes', outcome: 'done', detail: null, message: message({ id: 2 }) })
      renderAt('/c/1')
      const shown = await cards()
      fireEvent.click(within(shown[1]).getByRole('button', { name: 'Yes, do it' }))
      expect(mockAnswer).toHaveBeenCalledWith(1, 'tok-2', 'yes')
      expect(mockPost).not.toHaveBeenCalled()

      const done = await screen.findByRole('group', { name: 'Confirmation: Done' })
      expect(done).toHaveTextContent('Trash the task “Test 2”?')
      expect(done).toHaveTextContent('Trashed · PCC · “Test 2”')
      expect(done).toHaveTextContent('Say “undo that” to take it back (restore the task).')
      expect(within(done).queryByRole('button')).toBeNull()
      const open = screen.getAllByRole('group', { name: 'Confirmation' })
      expect(open).toHaveLength(2)
      expect(within(open[0]).getByRole('button', { name: 'Yes, do it' })).toBeEnabled()
    })

    it('Not now closes that card alone, and nothing is sent as a turn', async () => {
      mockGet.mockResolvedValueOnce(turn())
      mockGet.mockResolvedValue(turn(yes('tok-1', { result: null, error: 'not run: declined' })))
      mockAnswer.mockResolvedValue({ token: 'tok-1', answer: 'no', outcome: 'declined', detail: null, message: message({ id: 2 }) })
      renderAt('/c/1')
      fireEvent.click(within((await cards())[0]).getByRole('button', { name: 'Not now' }))
      expect(mockAnswer).toHaveBeenCalledWith(1, 'tok-1', 'no')
      expect(await screen.findByRole('group', { name: 'Confirmation: Left as it is' })).toHaveTextContent('Trash the task “Test 1”?')
      expect(screen.getAllByRole('group', { name: 'Confirmation' })).toHaveLength(2)
      expect(mockPost).not.toHaveBeenCalled()
    })

    it('shows the app’s refusal of a token on that card, claiming nothing', async () => {
      const refusal = 'tasks didn’t take the yes (what it would change has changed since), so nothing was done. Ask tasks again.'
      mockGet.mockResolvedValueOnce(turn())
      mockGet.mockResolvedValue(turn(yes('tok-3', { result: null, error: refusal })))
      mockAnswer.mockResolvedValue({ token: 'tok-3', answer: 'yes', outcome: 'failed', detail: refusal, message: message({ id: 2 }) })
      renderAt('/c/1')
      fireEvent.click(within((await cards())[2]).getByRole('button', { name: 'Yes, do it' }))
      const failed = await screen.findByRole('group', { name: 'Confirmation: Not done' })
      expect(failed).toHaveTextContent('so nothing was done.')
      expect(failed).not.toHaveTextContent('Trashed')
    })

    it('shows a card past its expiry as expired, with no buttons', async () => {
      const past = new Date(Date.now() - 1000).toISOString()
      const stale = { ...asking, app_confirm: { ...armed[0], expires_at: past }, app_confirms: [{ ...armed[0], expires_at: past }, armed[1]] }
      mockGet.mockResolvedValue({ ...detail, messages: [message({}), message({ id: 2, role: 'assistant', content: 'q', stop_reason: 'completed', tool_calls: [stale] })] })
      renderAt('/c/1')
      const expired = await screen.findByRole('group', { name: 'Confirmation: Expired' })
      expect(expired).toHaveTextContent('Trash the task “Test 1”?')
      expect(within(expired).queryByRole('button')).toBeNull()
      expect(screen.getAllByRole('group', { name: 'Confirmation' })).toHaveLength(1)
    })
  })

  it('shows what the apps did, from their typed records', async () => {
    const done = message({
      id: 2,
      role: 'assistant',
      content: 'Added “Water the plants” for tomorrow.',
      stop_reason: 'completed',
      tool_calls: [
        {
          tool: 'ask_tasks',
          arguments: { message: 'add a task to water the plants tomorrow' },
          result: 'ok',
          error: null,
          app_record: [
            { tool: 'create_task', outcome: 'done', touched: [{ app: 'tasks', kind: 'task', id: '1290', title: 'Water the plants' }], undo: { handle: 'trash:task:1290', how: 'move it to the trash' } },
          ],
        },
      ],
    })
    mockGet.mockResolvedValue({ ...detail, messages: [message({ content: 'add a task to water the plants tomorrow' }), done] })
    renderAt('/c/1')

    const receipts = await screen.findByRole('list', { name: 'Receipts' })
    expect(receipts).toHaveTextContent('Created · PCC · “Water the plants”')
  })

  it('turns Send into Stop while a turn runs, and Stop cancels it', async () => {
    mockPost.mockReturnValue(new Promise(() => {}))
    mockCancel.mockResolvedValue({ cancel_requested: true })
    renderAt('/c/1')
    await screen.findByText('Nothing is due today.')

    fireEvent.change(screen.getByLabelText('Message conductor'), { target: { value: 'sort my whole library' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Stop' }))

    expect(mockCancel).toHaveBeenCalledWith(1)
    expect(screen.queryByRole('button', { name: 'Send message' })).not.toBeInTheDocument()
  })

  // Some apps are places the user goes: a turn that ends in `open_<app>` sends
  // the tab there once the send-off line has landed.
  describe('app handoff', () => {
    const handoffPayload = {
      handoff: 'chess',
      title: 'Chess',
      path: '/',
      upstream: '127.0.0.1:8000',
      intent_param: 'intent',
      intent: "let's play chess",
    }
    const handoffTurn = message({
      id: 6,
      role: 'assistant',
      content: "Board's up — go.",
      stop_reason: 'completed',
      tool_calls: [
        {
          tool: 'open_chess',
          arguments: { intent: "let's play chess" },
          result: JSON.stringify(handoffPayload),
          error: null,
        },
      ],
    })

    let assign: ReturnType<typeof vi.fn>

    beforeEach(() => {
      assign = vi.fn()
      Object.defineProperty(window, 'location', {
        value: Object.assign(new URL('http://localhost:3000/c/1'), { assign }),
        writable: true,
        configurable: true,
      })
    })

    async function play() {
      renderAt('/c/1')
      await screen.findByText('Nothing is due today.')
      mockPost.mockResolvedValue({
        user_message: message({ id: 5, content: "let's play chess" }),
        assistant_message: handoffTurn,
      })
      mockGet.mockResolvedValue({
        ...detail,
        messages: [...detail.messages, message({ id: 5, content: "let's play chess" }), handoffTurn],
      })
      fireEvent.change(screen.getByLabelText('Message conductor'), {
        target: { value: "let's play chess" },
      })
      fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    }

    it('shows the send-off, then navigates to the app with the intent', async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true })
      try {
        await play()

        // The reply lands and is readable before the tab goes anywhere.
        expect(await screen.findByText("Board's up — go.")).toBeInTheDocument()
        expect(screen.getByRole('list', { name: 'Plan steps' })).toHaveTextContent("openchesslet's play chessdone")
        expect(assign).not.toHaveBeenCalled()

        await vi.advanceTimersByTimeAsync(1_200)

        // Dev host (no domain to swap into) → the app's upstream, intent in tow.
        expect(assign).toHaveBeenCalledWith("http://127.0.0.1:8000/?intent=let%27s+play+chess")
      } finally {
        vi.useRealTimers()
      }
    })

    it('does not navigate when merely reopening the conversation later', async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true })
      try {
        // The handoff rides on the persisted trajectory, so history has it too
        // — but replaying history must never fling the user back into chess.
        mockGet.mockResolvedValue({ ...detail, messages: [...detail.messages, handoffTurn] })
        renderAt('/c/1')

        expect(await screen.findByText("Board's up — go.")).toBeInTheDocument()
        await vi.advanceTimersByTimeAsync(5_000)

        expect(assign).not.toHaveBeenCalled()
      } finally {
        vi.useRealTimers()
      }
    })
  })

  it('explains a run that stopped without a reply', async () => {
    mockGet.mockResolvedValue({
      ...detail,
      messages: [
        message({ content: 'Do something impossible' }),
        message({ id: 5, role: 'assistant', content: null, stop_reason: 'max_iterations' }),
      ],
    })
    renderAt('/c/1')

    expect(await screen.findByText(/hit its step limit before finishing/)).toBeInTheDocument()
  })

  it('surfaces a rate-limit rejection and reloads the thread', async () => {
    const { ApiError } = await import('../../api/client')
    mockPost.mockRejectedValue(new ApiError(429, { detail: 'rate limit exceeded' }))
    renderAt('/c/1')
    await screen.findByText('Nothing is due today.')

    fireEvent.change(screen.getByLabelText('Message conductor'), {
      target: { value: 'again' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/Rate limited/)
  })
})

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

describe('AgentPage: the live plan view (#89)', () => {
  beforeEach(() => {
    FakeEventSource.instances = []
    vi.stubGlobal('EventSource', FakeEventSource)
  })
  afterEach(() => vi.unstubAllGlobals())

  const beat = { kind: 'model', tool: null, iteration: 1, ack: 'On it: one question first, then PCC.', elapsed_seconds: 0.3 }
  const plan = {
    steps: [
      { id: 's1', do: 'ask_user', question: 'Which dentist task?', options: ['Dentist appointment', 'Book dentist'] },
      { id: 's2', do: 'ask', app: 'tasks', request: 'move {s1} to Monday', destructive: false },
    ],
    start: 0,
  }

  it('draws the plan beside the ack, keeps it through the question, and a tap answers the same plan', async () => {
    let resolveRun: (value: never) => void = () => {}
    mockPost.mockImplementationOnce(() => new Promise((resolve) => (resolveRun = resolve as (value: never) => void)))
    mockGet.mockResolvedValue({ ...detail, messages: [] })
    renderAt('/c/1')
    await screen.findByLabelText('Message conductor')

    fireEvent.change(screen.getByLabelText('Message conductor'), { target: { value: 'move the dentist thing to Monday' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    await screen.findByText('move the dentist thing to Monday')
    const es = FakeEventSource.instances.at(-1)!

    act(() => {
      es.emit('turn_started', beat, 1)
      es.emit('plan', plan, 2)
    })
    expect(screen.getByText('On it: one question first, then PCC.')).toBeInTheDocument()
    const steps = screen.getByRole('list', { name: 'Plan steps' })
    expect(steps).toHaveTextContent('ask_userconductorWhich dentist task?queued')
    // {s1} is a dashed ref, written in words for a reader.
    expect(within(steps).getByText('what step 1 finds')).toBeInTheDocument()

    act(() => es.emit('paused', { reason: 'ask_user', question: 'Which dentist task? (Dentist appointment / Book dentist)', app: null, step: 's1' }, 3))
    // While the turn is still wrapping up, the card is there but not yet tappable.
    expect(screen.getByRole('button', { name: 'Book dentist' })).toBeDisabled()

    const asked: ConversationDetail = {
      ...detail,
      messages: [
        message({ id: 10, content: 'move the dentist thing to Monday' }),
        message({ id: 11, role: 'assistant', content: 'Which dentist task? (Dentist appointment / Book dentist)', stop_reason: 'completed' }),
      ],
    }
    mockGet.mockResolvedValue(asked)
    act(() => es.emit('turn_ended', { stop_reason: 'completed', message_id: 11 }, 4))
    resolveRun(undefined as never)

    await waitFor(() => expect(screen.getByRole('button', { name: 'Dentist appointment' })).toBeEnabled())
    // The plan didn't vanish when the turn ended; the question waits on you.
    expect(screen.getByRole('list', { name: 'Plan steps' })).toHaveTextContent('waiting on you')
    expect(screen.getAllByRole('status').map((el) => el.textContent)).toContain('Waiting on you')

    mockPost.mockReturnValue(new Promise(() => {}))
    fireEvent.click(screen.getByRole('button', { name: 'Book dentist' }))
    expect(mockPost).toHaveBeenLastCalledWith(1, 'Book dentist')
  })
})
