import { describe, expect, it } from 'vitest'
import type { AgentMessage, LiveReceipt, ToolCallRecord } from '../../types/agent'
import { nextSceneState } from './sceneState'
import { applyTurnEvent, EMPTY_STREAM, type TurnEventName, type TurnStream } from './turnStream'
import { buildTurns, pendingConfirm, presenceFor, receiptFrom } from './turnView'

function msg(over: Partial<AgentMessage>): AgentMessage {
  return { id: 1, conversation_id: 1, role: 'user', content: 'hi', tool_calls: null, stop_reason: null, created_at: '2026-11-06T18:00:00Z', ...over }
}
const askTasks = (over: Partial<ToolCallRecord> = {}): ToolCallRecord => ({ tool: 'ask_tasks', arguments: { message: 'wipe the kitchen project' }, result: 'ok', error: null, ...over })
const confirm = { app: 'tasks', thread_id: 9, token: 't', tool: 'trash_project', question: 'Trash the Kitchen project and its 12 tasks?' }

describe('receiptFrom', () => {
  it('names the verb, the entity titles and the undo', () => {
    expect(
      receiptFrom({ tool: 'create_task', outcome: 'done', touched: [{ app: 'tasks', kind: 'task', id: '1291', title: 'Sort Honey' }], undo: { handle: 'trash:task:1291', how: 'move it to the trash' } }, 'tasks'),
    ).toEqual({ appKey: 'tasks', verb: 'Created', app: 'PCC', entity: '“Sort Honey”', undoHow: 'move it to the trash', noUndo: undefined })
  })

  it('counts what a read found beyond the titles it lists', () => {
    const r = receiptFrom({ tool: 'list_tasks', outcome: 'looked_up', count: 31, touched: [{ app: 'tasks', kind: 'task', id: '41', title: 'Pay rent' }] }, 'tasks')
    expect(r?.verb).toBe('Found')
    expect(r?.entity).toBe('31 · “Pay rent”')
  })

  it('says when there is no undo, and why', () => {
    expect(receiptFrom({ tool: 'download_playlist', outcome: 'done', touched: [], undo: { none: 'downloaded files stay on disk' } }, 'music')?.noUndo).toBe('downloaded files stay on disk')
  })

  it('leaves a call waiting on confirmation to the confirmation card', () => {
    expect(receiptFrom({ tool: 'trash_project', outcome: 'not_run', reason: 'needs confirmation' }, 'tasks')).toBeNull()
  })

  it('reports a refusal with its reason', () => {
    expect(receiptFrom({ tool: 'complete_task', outcome: 'refused', reason: 'task 9 is already done' }, 'tasks')).toMatchObject({ verb: 'Refused', entity: 'task 9 is already done' })
  })
})

describe('pendingConfirm', () => {
  it('finds the question the last turn left waiting', () => {
    const messages = [msg({ id: 1 }), msg({ id: 2, role: 'assistant', tool_calls: [askTasks({ app_confirm: confirm })] })]
    expect(pendingConfirm(messages)).toMatchObject({ app: 'PCC', question: confirm.question })
  })

  it('treats a later call to the same app in that turn as voiding it', () => {
    const messages = [msg({ id: 1 }), msg({ id: 2, role: 'assistant', tool_calls: [askTasks({ app_confirm: confirm }), askTasks()] })]
    expect(pendingConfirm(messages)).toBeNull()
  })

  it('lapses once anything else is said', () => {
    const messages = [msg({ id: 1 }), msg({ id: 2, role: 'assistant', tool_calls: [askTasks({ app_confirm: confirm })] }), msg({ id: 3, content: 'what’s due today?' })]
    expect(pendingConfirm(messages)).toBeNull()
  })
})

describe('buildTurns', () => {
  it('pairs what was said with the reply, its steps and its receipts', () => {
    const turns = buildTurns(
      [
        msg({ id: 1, content: 'download the new Caribou album' }),
        msg({
          id: 2,
          role: 'assistant',
          content: 'Queued Honey.',
          stop_reason: 'completed',
          tool_calls: [{ tool: 'ask_music', arguments: { message: 'download the new Caribou album' }, result: 'ok', error: null, app_record: [{ tool: 'download_album', outcome: 'done', touched: [{ app: 'music', kind: 'download_job', id: '118', title: 'Honey' }], undo: { none: 'files stay' } }] }],
        }),
      ],
      null,
      null,
    )
    expect(turns).toHaveLength(1)
    expect(turns[0]).toMatchObject({ said: 'download the new Caribou album', reply: 'Queued Honey.', live: false })
    expect(turns[0].steps[0]).toMatchObject({ kind: 'ask', app: 'music', appKey: 'music', state: 'done' })
    expect(turns[0].receipts[0]).toMatchObject({ verb: 'Queued', entity: '“Honey”' })
  })

  it('marks the step an app is waiting on, and only on the last turn', () => {
    const waiting = buildTurns([msg({ id: 1 }), msg({ id: 2, role: 'assistant', tool_calls: [askTasks({ app_confirm: confirm })] })], null, null)
    expect(waiting[0].steps[0].state).toBe('wait')
    const lapsed = buildTurns([msg({ id: 1 }), msg({ id: 2, role: 'assistant', tool_calls: [askTasks({ app_confirm: confirm })] }), msg({ id: 3 }), msg({ id: 4, role: 'assistant', content: 'ok' })], null, null)
    expect(lapsed[0].steps[0].state).toBe('not_run')
  })

  it('shows a failed delegate call as failed, with its error', () => {
    const turns = buildTurns([msg({ id: 1 }), msg({ id: 2, role: 'assistant', tool_calls: [{ tool: 'ask_chess', arguments: { message: 'status?' }, result: null, error: 'chess timed out' }] })], null, null)
    expect(turns[0].steps[0]).toMatchObject({ state: 'failed', detail: 'chess timed out' })
  })

  it('keeps a note as your own turn with nothing put in conductor’s mouth', () => {
    const turns = buildTurns([msg({ id: 1, content: 'Filed Zeds Dead into Dubstep' })], null, null)
    expect(turns[0]).toMatchObject({ note: true, reply: null })
  })

  it('appends the in-flight turn with the running step from the activity poll', () => {
    const turns = buildTurns([], 'how is the game going?', { active: true, kind: 'tool', tool: 'ask_chess', iteration: 2, elapsed_seconds: 3 })
    expect(turns[0]).toMatchObject({ said: 'how is the game going?', live: true })
    expect(turns[0].steps[0]).toMatchObject({ kind: 'ask', appKey: 'chess', state: 'run', text: 'Asking chess' })
  })
})

describe('presenceFor', () => {
  it('follows the turn: thinking, working, waiting, ready', () => {
    expect(presenceFor(true, null, false)).toEqual({ state: 'think', label: 'Thinking' })
    expect(presenceFor(true, { active: true, kind: 'tool', tool: 'ask_music', iteration: 1, elapsed_seconds: 1 }, false)).toEqual({ state: 'work', label: 'Asking music' })
    expect(presenceFor(false, null, true)).toEqual({ state: 'wait', label: 'Waiting on you' })
    expect(presenceFor(false, null, false)).toEqual({ state: 'idle', label: 'Ready' })
  })
})

// --- the live channel (#38): nothing looks done before its receipt says so ---------------

const beat = { kind: 'model' as const, tool: null, iteration: 1, ack: 'On it, asking music and tasks.', elapsed_seconds: 0 }
const plan = {
  steps: [
    { id: 's1', do: 'ask', app: 'music', request: 'grab Swim', destructive: false },
    { id: 's2', do: 'ask', app: 'tasks', request: 'sort {s1}', destructive: false },
  ],
  start: 0,
}
const landed = (seq: number, over: Partial<LiveReceipt> = {}): LiveReceipt => ({
  seq,
  kind: 'ask',
  app: 'music',
  tool: 'ask_music',
  arguments: { message: 'grab Swim' },
  outcome: 'done',
  error: null,
  receipts: [{ source: 'app', tool: 'download_album', outcome: 'done', touched: [{ app: 'music', kind: 'album', id: '7', title: 'Swim' }] }],
  confirm: null,
  ...over,
})

function streamOf(events: [TurnEventName, unknown][]): TurnStream {
  return events.reduce((state, [name, data], i) => applyTurnEvent(state, name, data, i + 1), EMPTY_STREAM)
}

const live = (stream: TurnStream) => buildTurns([], 'grab Swim and sort it', null, stream)[0]

describe('buildTurns from the stream', () => {
  it('no step shows a state ahead of its receipt', () => {
    const planned = streamOf([['turn_started', beat], ['plan', plan]])
    expect(live(planned).steps.map((s) => s.state)).toEqual(['queued', 'queued'])

    const started = streamOf([['turn_started', beat], ['plan', plan], ['step_started', { seq: 1, tool: 'ask_music', arguments: { message: 'grab Swim' } }]])
    expect(live(started).steps.map((s) => s.state)).toEqual(['run', 'queued'])
    expect(live(started).receipts).toEqual([])

    const done = applyTurnEvent(started, 'receipt', landed(1), 99)
    expect(live(done).steps.map((s) => s.state)).toEqual(['done', 'queued'])
    expect(live(done).steps[0]).toMatchObject({ kind: 'ask', appKey: 'music', text: 'grab Swim' })
    expect(live(done).receipts).toEqual([expect.objectContaining({ appKey: 'music', verb: 'Queued', entity: '“Swim”' })])
  })

  it('the scene reads the same steps: a running strand is not done', () => {
    const started = streamOf([['turn_started', beat], ['plan', plan], ['step_started', { seq: 1, tool: 'ask_music', arguments: {} }]])
    const scene = nextSceneState(null, live(started), 'work', null, 10)
    expect(scene.strands).toEqual({ music: { state: 'run', since: 10 }, tasks: { state: 'queued', since: 10 } })
  })

  it('shows a failed or unrun step plainly, from its receipt', () => {
    const stream = streamOf([
      ['turn_started', beat],
      ['plan', plan],
      ['receipt', landed(1, { outcome: 'failed', error: 'music timed out', receipts: [] })],
      ['receipt', landed(2, { app: 'tasks', tool: 'ask_tasks', outcome: 'not_run', error: "not run: an earlier step of the plan didn't finish", receipts: [] })],
    ])
    expect(live(stream).steps.map((s) => [s.state, s.detail])).toEqual([
      ['failed', 'music timed out'],
      ['failed', "not run: an earlier step of the plan didn't finish"],
    ])
  })

  it('a step an app asked about waits on you once the plan pauses', () => {
    const asked = landed(1, { app: 'tasks', tool: 'ask_tasks', receipts: [{ source: 'app', tool: 'trash_project', outcome: 'not_run', reason: 'needs confirmation' }], confirm: { app: 'tasks', tool: 'trash_project', question: 'Trash it?' } })
    const stream = streamOf([['turn_started', beat], ['receipt', asked], ['paused', { reason: 'app_confirm', question: 'Trash it?', app: 'tasks', step: 's1' }]])
    expect(live(stream).steps[0].state).toBe('wait')
    expect(live(stream).receipts).toEqual([]) // the confirmation card's business
  })

  it('a step at code’s gate waits on you; nothing ran', () => {
    const gated = { steps: [{ id: 's1', do: 'ask', app: 'music', request: 'delete the album', destructive: true }], start: 0 }
    const stream = streamOf([['turn_started', beat], ['plan', gated], ['paused', { reason: 'gate', question: 'Just to check?', app: 'music', step: 's1' }]])
    expect(live(stream).steps).toEqual([{ appKey: 'music', kind: 'ask', app: 'music', text: 'delete the album', state: 'wait' }])
  })

  it('conductor’s own unknown receipt is never shown as a chip', () => {
    const stream = streamOf([['turn_started', beat], ['receipt', landed(1, { receipts: [{ source: 'conductor', tool: '', outcome: 'unknown', reason: 'no record' }] })]])
    expect(live(stream).receipts).toEqual([])
  })

  it('falls back to the beat while the stream has no steps yet', () => {
    const turns = buildTurns([], 'hi', { active: true, kind: 'tool', tool: 'ask_chess', iteration: 1, elapsed_seconds: 1 }, streamOf([['turn_started', beat]]))
    expect(turns[0].steps[0]).toMatchObject({ appKey: 'chess', state: 'run' })
  })
})
