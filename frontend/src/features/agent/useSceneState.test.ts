import { renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { JobCardData } from './jobStream'
import { HOLD_MS, RELEASE_MS } from './sceneState'
import type { ShellTurn } from './turnView'
import { useSceneState } from './useSceneState'

/** The scene lets a finished turn's strand go on its own clock (#232). */

const turn = (over: Partial<ShellTurn> = {}): ShellTurn => ({
  key: 't',
  said: 'x',
  note: false,
  ack: null,
  resumedNote: null,
  steps: [{ appKey: 'music', kind: 'ask', app: 'music', text: '', state: 'done' }],
  receipts: [],
  pause: null,
  announce: null,
  reply: 'Done.',
  stopReason: null,
  live: false,
  ...over,
})
const NO_JOBS: JobCardData[] = []

describe('useSceneState', () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'performance'] })
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('releases a finished turn’s strand a few seconds after the reply, with no new turn and no render', () => {
    let renders = 0
    const live = turn({ live: true, reply: null })
    const { result, rerender } = renderHook(
      ({ latest }) => {
        renders++
        return useSceneState(latest, 'idle', null, NO_JOBS)
      },
      { initialProps: { latest: live } },
    )
    vi.advanceTimersByTime(60_000)
    expect(result.current().strands.music.release).toBeUndefined() // still running: it keeps its strand
    rerender({ latest: turn() })
    const seen = renders
    vi.advanceTimersByTime(HOLD_MS - 1)
    expect(result.current().strands.music).toMatchObject({ state: 'done' })
    expect(result.current().strands.music.release).toBeUndefined()
    vi.advanceTimersByTime(1)
    expect(result.current().strands.music.release).toBeDefined()
    vi.advanceTimersByTime(RELEASE_MS)
    expect(result.current().strands).toEqual({})
    expect(renders).toBe(seen)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('keeps the strands of a plan that waits on the person', () => {
    const paused = turn({ pause: { kind: 'question', question: 'Which?', options: [] } })
    const { result } = renderHook(() => useSceneState(paused, 'wait', null, NO_JOBS))
    vi.advanceTimersByTime(HOLD_MS * 10)
    expect(result.current().strands.music).toMatchObject({ state: 'done' })
    expect(result.current().strands.music.release).toBeUndefined()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('drops its timer when the page goes away', () => {
    const { unmount } = renderHook(() => useSceneState(turn(), 'idle', null, NO_JOBS))
    expect(vi.getTimerCount()).toBe(1)
    unmount()
    expect(vi.getTimerCount()).toBe(0)
  })
})
