import { describe, expect, it } from 'vitest'
import type { LiveReceipt } from '../../types/agent'
import { applyTurnEvent, EMPTY_STREAM, queuedPlanSteps, streamActivity, type TurnEventName, type TurnStream } from './turnStream'

const beat = { kind: 'model' as const, tool: null, iteration: 1, ack: null, elapsed_seconds: 0.2 }
const receipt = (seq: number, over: Partial<LiveReceipt> = {}): LiveReceipt => ({
  seq,
  kind: 'ask',
  app: 'tasks',
  tool: 'ask_tasks',
  arguments: { message: 'add milk' },
  outcome: 'done',
  error: null,
  receipts: [],
  confirm: null,
  ...over,
})

function fold(events: [TurnEventName, unknown][]): TurnStream {
  return events.reduce((state, [name, data], i) => applyTurnEvent(state, name, data, i + 1), EMPTY_STREAM)
}

describe('applyTurnEvent', () => {
  it('folds a turn in order: started, plan, a step, its receipt, the end', () => {
    const state = fold([
      ['snapshot', { active: false }],
      ['turn_started', beat],
      ['plan', { steps: [{ id: 's1', do: 'ask', app: 'tasks' }], start: 0 }],
      ['step_started', { seq: 1, tool: 'ask_tasks', arguments: { message: 'add milk' } }],
      ['receipt', receipt(1)],
      ['turn_ended', { stop_reason: 'completed', message_id: 9 }],
    ])
    expect(state.active).toBe(false)
    expect(state.plan?.steps).toHaveLength(1)
    expect(state.steps).toEqual([{ seq: 1, tool: 'ask_tasks', arguments: { message: 'add milk' }, receipt: receipt(1) }])
    expect(state.ended).toEqual({ stop_reason: 'completed', message_id: 9 })
    expect(state.lastId).toBe(6)
  })

  it('a step that started has no receipt until its receipt event arrives', () => {
    const state = fold([
      ['turn_started', beat],
      ['step_started', { seq: 1, tool: 'ask_tasks', arguments: {} }],
    ])
    expect(state.steps[0].receipt).toBeNull()
  })

  it('a snapshot replaces what came before, and older events are ignored', () => {
    let state = fold([
      ['turn_started', beat],
      ['step_started', { seq: 1, tool: 'ask_tasks', arguments: {} }],
    ])
    state = applyTurnEvent(state, 'snapshot', { active: true, activity: beat, plan: null, steps: [{ seq: 1, receipt: receipt(1) }], paused: null }, 10)
    expect(state.steps).toEqual([{ seq: 1, receipt: receipt(1) }])
    expect(applyTurnEvent(state, 'step_started', { seq: 2, tool: 'ask_music', arguments: {} }, 9)).toBe(state)
  })

  it('keeps steps in seq order and the first plan of the turn', () => {
    const state = fold([
      ['turn_started', beat],
      ['plan', { steps: [{ id: 's1', do: 'ask' }], start: 0 }],
      ['plan', { steps: [{ id: 's9', do: 'ask' }], start: 4 }],
      ['receipt', receipt(2)],
      ['step_started', { seq: 1, tool: 'ask_music', arguments: {} }],
    ])
    expect(state.plan?.steps[0].id).toBe('s1')
    expect(state.steps.map((s) => s.seq)).toEqual([1, 2])
  })

  it('notes a resumed plan, and a snapshot on reconnect carries it', () => {
    const resumed = { reason: 'ask_user', step: 's1', answer: 'Book dentist' }
    const state = fold([
      ['turn_started', beat],
      ['resumed', resumed],
    ])
    expect(state.resumed).toEqual(resumed)
    const again = applyTurnEvent(EMPTY_STREAM, 'snapshot', { active: true, activity: beat, plan: null, steps: [], paused: null, resumed }, 4)
    expect(again.resumed).toEqual(resumed)
    expect(applyTurnEvent(again, 'turn_started', beat, 5).resumed).toBeNull()
  })

  it('a new turn starts clean', () => {
    const state = fold([
      ['turn_started', beat],
      ['receipt', receipt(1)],
      ['turn_ended', { stop_reason: 'completed', message_id: 1 }],
      ['turn_started', beat],
    ])
    expect(state.steps).toEqual([])
    expect(state.ended).toBeNull()
    expect(state.active).toBe(true)
  })
})

describe('streamActivity and queuedPlanSteps', () => {
  it('gives the beat in the poll shape while active, nothing after', () => {
    const live = fold([['turn_started', { ...beat, kind: 'tool', tool: 'ask_tasks', ack: 'On it, asking tasks.' }]])
    expect(streamActivity(live, 7.5)).toEqual({ active: true, kind: 'tool', tool: 'ask_tasks', iteration: 1, ack: 'On it, asking tasks.', elapsed_seconds: 7.5 })
    expect(streamActivity({ ...live, active: false }, 7.5)).toBeNull()
  })

  it('lists the plan steps past the ones started, from where this turn begins', () => {
    const state = fold([
      ['turn_started', beat],
      ['plan', { steps: [{ id: 's1', do: 'ask_user' }, { id: 's2', do: 'ask' }, { id: 's3', do: 'open' }], start: 1 }],
      ['step_started', { seq: 1, tool: 'ask_tasks', arguments: {} }],
    ])
    expect(queuedPlanSteps(state).map((s) => s.id)).toEqual(['s3'])
  })
})
