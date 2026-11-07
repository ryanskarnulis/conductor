import { describe, expect, it } from 'vitest'
import type { AgentMessage, ToolCallRecord } from '../../types/agent'
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

  it('shows a look as a look step with a found receipt, and no ask', () => {
    const turns = buildTurns(
      [
        msg({ id: 1, content: 'what is due today?' }),
        msg({
          id: 2,
          role: 'assistant',
          content: 'Pay rent is due.',
          stop_reason: 'completed',
          tool_calls: [{ tool: 'look_tasks', arguments: {}, result: '{}', error: null, app_record: [{ tool: 'glance', outcome: 'looked_up', count: 1, touched: [{ app: 'tasks', kind: 'task', id: '41', title: 'Pay rent' }] }] }],
        }),
      ],
      null,
      null,
    )
    expect(turns[0].steps[0]).toMatchObject({ kind: 'look', app: 'PCC', appKey: 'tasks', state: 'done' })
    expect(turns[0].receipts[0]).toMatchObject({ verb: 'Found', app: 'PCC', entity: '“Pay rent”' })
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
