import { describe, expect, it } from 'vitest'
import { HOLD_MS, nextSceneState, nextSceneWake, RELEASE_MS } from './sceneState'
import type { ShellTurn } from './turnView'

const turn = (steps: ShellTurn['steps']): ShellTurn => ({ key: 't', said: 'x', note: false, ack: null, resumedNote: null, steps, receipts: [], pause: null, announce: null, reply: null, stopReason: null, live: false })
const step = (appKey: string | null, state: ShellTurn['steps'][number]['state']) => ({ appKey, kind: 'ask', app: appKey ?? 'conductor', text: '', state })

describe('nextSceneState', () => {
  it('gives each web app on the plan a strand in its state', () => {
    const s = nextSceneState(null, turn([step('music', 'run'), step('tasks', 'done'), step(null, 'done')]), 'work', null, 100)
    expect(s.strands).toEqual({ music: { state: 'run', since: 100 }, tasks: { state: 'done', since: 100 } })
    expect(s.pres).toBe('work')
  })

  it('keeps when a strand entered its state, so the scene plays the change once', () => {
    const a = nextSceneState(null, turn([step('music', 'run')]), 'work', null, 100)
    const b = nextSceneState(a, turn([step('music', 'run')]), 'work', null, 900)
    const c = nextSceneState(b, turn([step('music', 'done')]), 'idle', null, 1500)
    expect(b.strands.music.since).toBe(100)
    expect(c.strands.music).toEqual({ state: 'done', since: 1500 })
    expect(c.lastReceipt).toEqual({ app: 'music', t: 1500 })
  })

  it('bumps idx only when something visible changes (reduced motion redraws on it)', () => {
    const a = nextSceneState(null, turn([step('music', 'run')]), 'work', null, 100)
    const b = nextSceneState(a, turn([step('music', 'run')]), 'work', null, 200)
    const c = nextSceneState(b, turn([step('tasks', 'wait')]), 'wait', 'tasks', 300)
    expect(b.idx).toBe(a.idx)
    expect(c.idx).toBe(a.idx + 1)
    expect(c.confirm).toEqual({ app: 'tasks' })
  })

  it('staggers queued strands by their place on the plan', () => {
    const s = nextSceneState(null, turn([step('music', 'queued'), step('tasks', 'queued')]), 'think', null, 100)
    expect(s.strands).toEqual({ music: { state: 'queued', since: 100, k: 0 }, tasks: { state: 'queued', since: 100, k: 1 } })
  })

  it('never dims a lit strand for a later step still queued', () => {
    const s = nextSceneState(null, turn([step('tasks', 'done'), step('music', 'run'), step('tasks', 'queued')]), 'work', null, 100)
    expect(s.strands.tasks.state).toBe('done')
  })

  it('shows a failed step as a frayed strand and has Glitch glitch once', () => {
    const a = nextSceneState(null, turn([step('music', 'run')]), 'work', null, 100)
    const b = nextSceneState(a, turn([step('music', 'failed')]), 'idle', null, 500)
    expect(b.strands.music.state).toBe('failed')
    expect(b.lastFail).toEqual({ app: 'music', t: 500 })
    expect(b.steps).toEqual({ music: { state: 'failed' } })
    expect(nextSceneState(b, turn([step('music', 'failed')]), 'idle', null, 900).lastFail).toEqual({ app: 'music', t: 500 })
  })

  it('draws no strand for a step that did not run', () => {
    const s = nextSceneState(null, turn([step('music', 'not_run')]), 'idle', null, 100)
    expect(s.strands).toEqual({})
  })
})

describe('a finished turn lets its strands go (#232)', () => {
  const ended = turn([step('music', 'done')])
  const live = { ...ended, live: true }
  const at = (prev: ReturnType<typeof nextSceneState> | null, t: ShellTurn | null, now: number, confirm: string | null = null) =>
    nextSceneState(prev, t, confirm ? 'wait' : 'idle', confirm, now)

  it('holds the result, releases it, then is at rest, with no new turn', () => {
    const a = at(at(null, live, 100), ended, 1000)
    expect(a.strands.music).toEqual({ state: 'done', since: 100 })
    expect(nextSceneWake(a, 1000)).toBe(1000 + HOLD_MS)
    const held = at(a, ended, 1000 + HOLD_MS - 1)
    expect(held.strands.music.release).toBeUndefined()
    expect(held.idx).toBe(a.idx)
    const going = at(held, ended, 1000 + HOLD_MS)
    expect(going.strands.music).toEqual({ state: 'done', since: 100, release: 1000 + HOLD_MS })
    expect(going.idx).toBe(a.idx + 1)
    expect(nextSceneWake(going, 1000 + HOLD_MS)).toBe(1000 + HOLD_MS + RELEASE_MS)
    const gone = at(going, ended, 1000 + HOLD_MS + RELEASE_MS)
    expect(gone.strands).toEqual({})
    expect(gone.idx).toBe(going.idx + 1)
    expect(nextSceneWake(gone, 1000 + HOLD_MS + RELEASE_MS)).toBeNull()
    // The same finished turn never re-attaches.
    expect(at(gone, ended, 60_000).strands).toEqual({})
    expect(HOLD_MS + RELEASE_MS).toBeLessThanOrEqual(5000)
  })

  it('counts the hold from the end of the turn, not from the step', () => {
    const a = at(null, live, 100)
    const b = at(a, live, 100 + HOLD_MS * 3)
    expect(b.strands.music).toEqual({ state: 'done', since: 100 })
    expect(nextSceneWake(b, 100 + HOLD_MS * 3)).toBeNull()
    const c = at(b, ended, 20_000)
    expect(at(c, ended, 20_000 + HOLD_MS - 1).strands.music.release).toBeUndefined()
  })

  it('lets a failed strand go the same way', () => {
    const failed = turn([step('music', 'failed')])
    const a = at(null, failed, 100)
    expect(at(a, failed, 100 + HOLD_MS).strands.music).toEqual({ state: 'failed', since: 100, release: 100 + HOLD_MS })
    expect(at(a, failed, 100 + HOLD_MS + RELEASE_MS).strands).toEqual({})
  })

  it('keeps every strand of a plan that waits on the person', () => {
    const paused = { ...turn([step('music', 'done'), step('tasks', 'wait')]), pause: { kind: 'question' as const, question: 'Which?', options: [] } }
    const a = at(null, paused, 100)
    const b = at(a, paused, 100 + HOLD_MS * 10)
    expect(b.strands).toEqual({ music: { state: 'done', since: 100 }, tasks: { state: 'wait', since: 100 } })
    expect(nextSceneWake(b, 100 + HOLD_MS * 10)).toBeNull()
    // A write waiting on a yes is the same.
    const confirm = turn([step('music', 'done'), step('tasks', 'wait')])
    expect(at(at(null, confirm, 100, 'tasks'), confirm, 100 + HOLD_MS * 10, 'tasks').strands.music).toEqual({ state: 'done', since: 100 })
  })

  it('starts the hold again when the plan resumes and ends', () => {
    const paused = { ...ended, pause: { kind: 'question' as const, question: 'Which?', options: [] } }
    const a = at(at(null, paused, 100), paused, 50_000)
    const b = at(a, ended, 50_100)
    expect(b.strands.music).toEqual({ state: 'done', since: 100 })
    expect(nextSceneWake(b, 50_100)).toBe(50_100 + HOLD_MS)
  })

  it('lights a new turn on the same app afresh after a release', () => {
    const gone = at(at(null, ended, 100), ended, 100 + HOLD_MS + RELEASE_MS)
    const again = at(gone, { ...turn([step('music', 'run')]), live: true }, 30_000)
    expect(again.strands.music).toEqual({ state: 'run', since: 30_000 })
    const done = at(again, live, 31_000)
    expect(done.strands.music).toEqual({ state: 'done', since: 31_000 })
    expect(at(done, ended, 32_000).rest.music).toEqual({ state: 'done', t: 32_000 })
  })
})
