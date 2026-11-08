import { describe, expect, it } from 'vitest'
import { nextSceneState } from './sceneState'
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
