import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getConversation, getTurnActivity, listConversations, postMessage } from '../../api/agent'
import { getWorklist } from '../../api/sort'
import type { AgentMessage, ConversationDetail } from '../../types/agent'
import { AgentPage } from './AgentPage'

vi.mock('../../api/sort', () => ({
  getWorklist: vi.fn(),
  openGroup: vi.fn(),
  fileSongs: vi.fn(),
}))

vi.mock('../../api/agent', () => ({
  createConversation: vi.fn(),
  deleteConversation: vi.fn(),
  getConversation: vi.fn(),
  getTurnActivity: vi.fn(),
  listConversations: vi.fn(),
  postMessage: vi.fn(),
}))

const mockList = vi.mocked(listConversations)
const mockGet = vi.mocked(getConversation)
const mockPost = vi.mocked(postMessage)
const mockActivity = vi.mocked(getTurnActivity)
const mockWorklist = vi.mocked(getWorklist)

function message(overrides: Partial<AgentMessage>): AgentMessage {
  return {
    id: 1,
    conversation_id: 1,
    role: 'user',
    content: 'hello',
    tool_calls: null,
    stop_reason: null,
    created_at: '2026-07-12T10:00:00Z',
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
  created_at: '2026-07-12T10:00:00Z',
  updated_at: '2026-07-12T10:00:00Z',
  messages: [message({ content: 'what tasks are due today?' }), assistantWithDelegateCalls],
}

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
  mockList.mockResolvedValue([
    {
      id: 1,
      title: 'what tasks are due today?',
      created_at: detail.created_at,
      updated_at: detail.updated_at,
    },
  ])
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
    expect(screen.getByText('Asked tasks')).toBeInTheDocument()
    expect(screen.getByText(/Asked chess/)).toBeInTheDocument()
    expect(screen.getByText('chess is rate-limiting requests right now.')).toBeInTheDocument()
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
    expect(await screen.findByRole('status')).toHaveTextContent('Asking chess… · 15s')

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
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
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
        expect(screen.getByText('Opened chess')).toBeInTheDocument()
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
