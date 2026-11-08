import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cancelJob } from '../../api/agent'
import { ApiError } from '../../api/client'
import type { JobCancelResult, JobEvent } from '../../types/agent'
import { JOB_STREAM_RETRIES, useJobs } from './useJobs'

vi.mock('../../api/agent', () => ({
  cancelJob: vi.fn(),
  turnEventsUrl: (id: number) => `/api/agent/conversations/${id}/events`,
}))

const mockCancel = vi.mocked(cancelJob)

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

const job = (over: Partial<JobEvent> = {}): JobEvent => ({ app: 'music', ref: '118', status: 'working', done: 12, total: 40, title: 'Late Night Drive', untrusted: true, ended: false, ...over })
const latest = () => FakeEventSource.instances[FakeEventSource.instances.length - 1]

beforeEach(() => {
  FakeEventSource.instances = []
  mockCancel.mockReset()
  vi.stubGlobal('EventSource', FakeEventSource)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useJobs', () => {
  it('follows the stream while no send is in flight, and not during one', () => {
    const refresh = vi.fn()
    const { rerender } = renderHook(({ follow }) => useJobs(1, follow, refresh), { initialProps: { follow: true } })
    expect(FakeEventSource.instances).toHaveLength(1)
    expect(latest().url).toBe('/api/agent/conversations/1/events')
    rerender({ follow: false })
    expect(latest().readyState).toBe(FakeEventSource.CLOSED)
    rerender({ follow: true })
    expect(FakeEventSource.instances).toHaveLength(2)
  })

  it('finds the running jobs in the first snapshot, after a reload', () => {
    const { result } = renderHook(() => useJobs(1, true, vi.fn()))
    act(() => latest().emit('snapshot', { active: false, jobs: [job({ done: 20 })] }, 5))
    expect(result.current.state.jobs.map((j) => [j.ref, j.done])).toEqual([['118', 20]])
    expect(latest().readyState).toBe(FakeEventSource.OPEN)
  })

  it('closes the stream at once when there is nothing to follow', () => {
    renderHook(() => useJobs(1, true, vi.fn()))
    act(() => latest().emit('snapshot', { active: false }, 5))
    expect(latest().readyState).toBe(FakeEventSource.CLOSED)
  })

  it('moves a card on each progress event, and reads the thread again when the job ends', () => {
    const refresh = vi.fn()
    const { result } = renderHook(() => useJobs(1, true, refresh))
    act(() => latest().emit('snapshot', { active: false, jobs: [job()] }, 5))
    act(() => latest().emit('job', job({ done: 30 }), 6))
    expect(result.current.state.jobs[0].done).toBe(30)
    expect(refresh).not.toHaveBeenCalled()
    act(() => latest().emit('job', job({ done: 38, status: 'completed', ended: true, how: 'ended', note_message_id: 9 }), 7))
    expect(refresh).toHaveBeenCalledOnce()
    expect(result.current.state.jobs[0]).toMatchObject({ ended: true, note_message_id: 9 })
    expect(latest().readyState).toBe(FakeEventSource.CLOSED)
  })

  it('notices a job that ended while no stream was open', () => {
    const refresh = vi.fn()
    const { rerender } = renderHook(({ follow }) => useJobs(1, follow, refresh), { initialProps: { follow: true } })
    act(() => latest().emit('snapshot', { active: false, jobs: [job()] }, 5))
    rerender({ follow: false })
    rerender({ follow: true })
    act(() => latest().emit('snapshot', { active: false }, 2))
    expect(refresh).toHaveBeenCalledOnce()
  })

  it('stops reconnecting a stream that keeps failing', () => {
    renderHook(() => useJobs(1, true, vi.fn()))
    act(() => latest().emit('snapshot', { active: false, jobs: [job()] }, 5))
    for (let i = 0; i < JOB_STREAM_RETRIES; i++) act(() => latest().fail())
    expect(latest().readyState).toBe(FakeEventSource.CLOSED)
  })

  it('asks the app to stop on Cancel: asking, then asked, and reads the thread for the note', async () => {
    let answer: () => void = () => {}
    mockCancel.mockReturnValue(new Promise<JobCancelResult>((resolve) => (answer = () => resolve({} as JobCancelResult))))
    const refresh = vi.fn()
    const { result } = renderHook(() => useJobs(1, true, refresh))
    let done: Promise<void> = Promise.resolve()
    act(() => {
      done = result.current.cancel('music', '118')
    })
    expect(mockCancel).toHaveBeenCalledWith(1, 'music', '118')
    expect(result.current.asks).toEqual({ 'music:118': 'sending' })
    await act(async () => {
      answer()
      await done
    })
    expect(result.current.asks).toEqual({ 'music:118': 'asked' })
    expect(refresh).toHaveBeenCalledOnce()
  })

  it('a Cancel that fails is the card’s error, never a stopped job', async () => {
    mockCancel.mockRejectedValue(new ApiError(502, { detail: 'music couldn’t be reached' }))
    const { result } = renderHook(() => useJobs(1, true, vi.fn()))
    await act(async () => {
      await result.current.cancel('music', '118')
    })
    expect(result.current.asks).toEqual({})
    expect(result.current.errors).toEqual({ 'music:118': 'music couldn’t be reached' })
  })
})
