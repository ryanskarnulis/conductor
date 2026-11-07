import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { getTurnActivity } from '../../api/agent'
import { STREAM_RETRIES, TURN_ACTIVITY_POLL_MS, useTurnActivity } from './useTurnActivity'

vi.mock('../../api/agent', () => ({
  getTurnActivity: vi.fn(),
  turnEventsUrl: (id: number) => `/api/agent/conversations/${id}/events`,
}))

const mockPoll = vi.mocked(getTurnActivity)

/** A stand-in for the browser's EventSource: the test drives it. */
class FakeEventSource {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSED = 2
  static instances: FakeEventSource[] = []

  readonly url: string
  readyState = FakeEventSource.CONNECTING
  onerror: ((event: Event) => void) | null = null
  private listeners = new Map<string, ((event: MessageEvent<string>) => void)[]>()

  constructor(url: string) {
    this.url = url
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

  fail(state = FakeEventSource.CONNECTING) {
    this.readyState = state
    this.onerror?.(new Event('error'))
  }
}

const beat = { kind: 'tool', tool: 'ask_tasks', iteration: 1, ack: 'On it, asking tasks.', elapsed_seconds: 2 }
const pollBeat = { active: true, kind: 'tool' as const, tool: 'ask_chess', iteration: 1, elapsed_seconds: 4 }

beforeEach(() => {
  FakeEventSource.instances = []
  mockPoll.mockReset()
  mockPoll.mockResolvedValue(pollBeat)
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

function stream(): FakeEventSource {
  const source = FakeEventSource.instances.at(-1)
  if (!source) throw new Error('no stream opened')
  return source
}

describe('useTurnActivity over the stream', () => {
  beforeEach(() => vi.stubGlobal('EventSource', FakeEventSource))

  it('opens the conversation’s stream and folds its events into the turn', () => {
    const { result } = renderHook(() => useTurnActivity(5, true))
    expect(stream().url).toBe('/api/agent/conversations/5/events')
    expect(result.current).toBeNull()

    act(() => {
      stream().emit('snapshot', { active: false }, 1)
      stream().emit('turn_started', beat, 2)
      stream().emit('plan', { steps: [{ id: 's1', do: 'ask', app: 'tasks', request: 'add milk' }], start: 0 }, 3)
      stream().emit('step_started', { seq: 1, tool: 'ask_tasks', arguments: { message: 'add milk' } }, 4)
    })
    expect(result.current?.activity).toMatchObject({ active: true, kind: 'tool', tool: 'ask_tasks', ack: 'On it, asking tasks.' })
    expect(result.current?.stream?.steps).toEqual([{ seq: 1, tool: 'ask_tasks', arguments: { message: 'add milk' }, receipt: null }])
    expect(mockPoll).not.toHaveBeenCalled()
  })

  it('closes the stream when the turn ends, without falling back', () => {
    renderHook(() => useTurnActivity(5, true))
    act(() => {
      stream().emit('turn_started', beat, 1)
      stream().emit('turn_ended', { stop_reason: 'completed', message_id: 3 }, 2)
    })
    expect(stream().readyState).toBe(FakeEventSource.CLOSED)
    expect(mockPoll).not.toHaveBeenCalled()
  })

  it('falls back to the poll when the stream fails before it says anything', async () => {
    const { result } = renderHook(() => useTurnActivity(5, true))
    await act(async () => {
      stream().fail()
    })
    expect(stream().readyState).toBe(FakeEventSource.CLOSED)
    expect(mockPoll).toHaveBeenCalledWith(5)
    expect(result.current).toEqual({ activity: pollBeat, stream: null })
  })

  it('lets a working stream reconnect, and gives up after repeated failures', async () => {
    renderHook(() => useTurnActivity(5, true))
    act(() => stream().emit('turn_started', beat, 1))
    await act(async () => {
      stream().fail()
    })
    expect(mockPoll).not.toHaveBeenCalled() // the browser reconnects; a snapshot will catch it up
    await act(async () => {
      for (let i = 1; i < STREAM_RETRIES; i++) stream().fail()
    })
    expect(mockPoll).toHaveBeenCalled()
  })

  it('falls back when the server closes the stream for good', async () => {
    renderHook(() => useTurnActivity(5, true))
    act(() => stream().emit('turn_started', beat, 1))
    await act(async () => {
      stream().fail(FakeEventSource.CLOSED)
    })
    expect(mockPoll).toHaveBeenCalled()
  })

  it('keeps the elapsed counter moving between events', () => {
    vi.useFakeTimers()
    const { result } = renderHook(() => useTurnActivity(5, true))
    act(() => stream().emit('turn_started', { ...beat, elapsed_seconds: 0 }, 1))
    act(() => {
      vi.advanceTimersByTime(6_000)
    })
    expect(result.current?.activity?.elapsed_seconds).toBeGreaterThanOrEqual(5)
  })

  it('closes the stream when the send ends and returns nothing', () => {
    const { result, rerender } = renderHook(({ active }) => useTurnActivity(5, active), { initialProps: { active: true } })
    act(() => stream().emit('turn_started', beat, 1))
    rerender({ active: false })
    expect(stream().readyState).toBe(FakeEventSource.CLOSED)
    expect(result.current).toBeNull()
  })
})

describe('useTurnActivity without EventSource', () => {
  beforeEach(() => vi.stubGlobal('EventSource', undefined))

  it('polls the activity endpoint, as before', async () => {
    vi.useFakeTimers()
    const { result } = renderHook(() => useTurnActivity(5, true))
    await act(async () => {
      await Promise.resolve()
    })
    expect(FakeEventSource.instances).toEqual([])
    expect(result.current).toEqual({ activity: pollBeat, stream: null })
    await act(async () => {
      vi.advanceTimersByTime(TURN_ACTIVITY_POLL_MS)
      await Promise.resolve()
    })
    expect(mockPoll).toHaveBeenCalledTimes(2)
  })

  it('keeps the last beat through a failed poll', async () => {
    vi.useFakeTimers()
    const { result } = renderHook(() => useTurnActivity(5, true))
    await act(async () => {
      await Promise.resolve()
    })
    mockPoll.mockRejectedValueOnce(new Error('blip'))
    await act(async () => {
      vi.advanceTimersByTime(TURN_ACTIVITY_POLL_MS)
      await Promise.resolve()
    })
    expect(result.current?.activity).toEqual(pollBeat)
  })

  it('does nothing while idle', () => {
    const { result } = renderHook(() => useTurnActivity(5, false))
    expect(result.current).toBeNull()
    expect(mockPoll).not.toHaveBeenCalled()
  })
})
