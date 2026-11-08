import { describe, expect, it } from 'vitest'
import type { AgentMessage, JobEvent, LiveReceipt, ToolCallRecord } from '../../types/agent'
import { applyJobEvent, EMPTY_JOBS, jobCards, type JobsState } from './jobStream'
import { nextSceneState } from './sceneState'
import { applyTurnEvent, EMPTY_STREAM, type TurnEventName } from './turnStream'
import { buildTurns, pendingConfirm } from './turnView'

/** How a background job shows in the thread, the plan and the scene (#90). */

function msg(over: Partial<AgentMessage>): AgentMessage {
  return { id: 1, conversation_id: 1, role: 'user', content: 'hi', tool_calls: null, stop_reason: null, created_at: '2026-11-06T18:00:00Z', ...over }
}
const job = (over: Partial<JobEvent> = {}): JobEvent => ({ app: 'music', ref: '118', status: 'working', done: 12, total: 40, title: 'Late Night Drive', untrusted: true, ended: false, ...over })
const state = (...jobs: JobEvent[]): JobsState => jobs.reduce((s, j, i) => applyJobEvent(s, 'job', j, i + 1), EMPTY_JOBS)
const cardsOf = (...jobs: JobEvent[]) => jobCards(state(...jobs), {}, new Set())
const note = (id: number): AgentMessage => msg({ id, role: 'assistant', content: 'Late Night Drive is in: 38 of 40.', stop_reason: 'job_note' })
const confirm = { app: 'tasks', thread_id: 9, token: 't', tool: 'trash_project', question: 'Trash the Kitchen project?' }
const ask = (over: Partial<ToolCallRecord> = {}): ToolCallRecord => ({ tool: 'ask_tasks', arguments: { message: 'wipe it' }, result: 'ok', error: null, ...over })

describe('a job’s note in the thread', () => {
  it('is its own turn, never the reply to what you said last', () => {
    const turns = buildTurns([msg({ id: 1, content: 'stop it' }), note(2)], null, null)
    expect(turns).toHaveLength(2)
    expect(turns[0]).toMatchObject({ said: 'stop it', note: true, reply: null, jobNote: false })
    expect(turns[1]).toMatchObject({ said: '', jobNote: true, reply: 'Late Night Drive is in: 38 of 40.', stopReason: null, at: '2026-11-06T18:00:00Z' })
  })

  it('does not take the place of the last turn: a waiting question still shows, and still takes a yes', () => {
    const messages = [msg({ id: 1 }), msg({ id: 2, role: 'assistant', content: 'Trash it?', stop_reason: 'completed', tool_calls: [ask({ app_confirm: confirm })] }), note(3)]
    expect(pendingConfirm(messages)).toMatchObject({ app: 'PCC', question: confirm.question })
    const turns = buildTurns(messages, null, null)
    expect(turns.map((t) => t.jobNote)).toEqual([false, true])
  })

  it('still lapses a question when the person says something else', () => {
    const messages = [msg({ id: 1 }), msg({ id: 2, role: 'assistant', tool_calls: [ask({ app_confirm: confirm })] }), note(3), msg({ id: 4, content: 'what’s due?' })]
    expect(pendingConfirm(messages)).toBeNull()
  })
})

describe('a step that started a job', () => {
  const beat = { kind: 'model' as const, tool: null, iteration: 1, ack: null, elapsed_seconds: 0.2 }
  const receipt = (status: string): LiveReceipt => ({
    seq: 1,
    kind: 'ask',
    app: 'music',
    tool: 'ask_music',
    arguments: { message: 'download this playlist' },
    outcome: 'done',
    error: null,
    receipts: [{ source: 'app', tool: 'queue_download', outcome: 'done', job: { ref: '118', status, done: 0, total: 40 } }],
    confirm: null,
  })
  const live = (status: string) => {
    const events: [TurnEventName, unknown][] = [
      ['turn_started', beat],
      ['plan', { steps: [{ id: 's1', do: 'ask', app: 'music', request: 'download this playlist' }], start: 0 }],
      ['step_started', { seq: 1, tool: 'ask_music', arguments: {} }],
      ['receipt', receipt(status)],
    ]
    const stream = events.reduce((s, [name, data], i) => applyTurnEvent(s, name, data, i + 1), EMPTY_STREAM)
    return buildTurns([], 'download this playlist', null, stream)[0]
  }

  it('is running in the background, not done, while its job runs', () => {
    expect(live('submitted').steps.map((s) => s.state)).toEqual(['job'])
    expect(live('working').steps.map((s) => s.state)).toEqual(['job'])
  })

  it('is done when the app says the job it started had already ended', () => {
    expect(live('completed').steps.map((s) => s.state)).toEqual(['done'])
  })
})

describe('the Living Web follows the job cards', () => {
  const idle = (cards = cardsOf(), prev = null as ReturnType<typeof nextSceneState> | null, now = 100) => nextSceneState(prev, null, 'idle', null, now, cards)

  it('hums the strand of the app that has a job running, with a ring from its done and total', () => {
    const scene = idle(cardsOf(job({ done: 12, total: 40 })))
    expect(scene.strands.music).toEqual({ state: 'job', since: 100 })
    expect(scene.jobs).toEqual({ music: { n: 12, of: 40 } })
    expect(scene.job).toEqual({ n: 12, of: 40, app: 'music' })
  })

  it('draws a queued job as a queued strand, not a working one, and no ring', () => {
    const scene = idle(cardsOf(job({ status: 'submitted', done: 0 })))
    expect(scene.strands.music.state).toBe('queued')
    expect(scene.jobs).toEqual({})
  })

  it('gives each app its own ring, and sums the jobs of one app', () => {
    const scene = idle(cardsOf(job({ done: 10, total: 40 }), job({ ref: '119', done: 5, total: 10 }), job({ app: 'tasks', ref: '7', done: 1, total: 4 })))
    expect(scene.jobs).toEqual({ music: { n: 15, of: 50 }, tasks: { n: 1, of: 4 } })
  })

  it('has no ring for a job whose total is unknown, but still hums', () => {
    const scene = idle(cardsOf(job({ total: null })))
    expect(scene.jobs).toEqual({})
    expect(scene.strands.music.state).toBe('job')
    expect(scene.job).toEqual({ n: 12, of: 0, app: 'music' })
  })

  it('is done only on the end event, and failed when the job failed', () => {
    const running = idle(cardsOf(job()))
    const finished = idle(cardsOf(job({ status: 'completed', done: 40, ended: true, note_message_id: 3 })), running, 900)
    expect(finished.strands.music).toEqual({ state: 'done', since: 900 })
    expect(finished.lastReceipt).toEqual({ app: 'music', t: 900 })
    const failed = idle(cardsOf(job({ status: 'failed', ended: true, note_message_id: 3 })), running, 900)
    expect(failed.strands.music.state).toBe('failed')
    expect(failed.lastFail).toEqual({ app: 'music', t: 900 })
    // The report of a job that has finished, before its end event, is not done.
    expect(idle(cardsOf(job({ status: 'completed' }))).strands.music.state).toBe('job')
  })

  it('never overrides a running turn’s own strand, and keeps when it began', () => {
    const turn = { key: 't', said: 'x', note: false, ack: null, resumedNote: null, steps: [{ appKey: 'music', kind: 'ask', app: 'music', text: '', state: 'run' as const }], receipts: [], pause: null, announce: null, reply: null, stopReason: null, live: true }
    expect(nextSceneState(null, turn, 'work', null, 100, cardsOf(job())).strands.music.state).toBe('run')
    const a = idle(cardsOf(job({ done: 1 })))
    const b = idle(cardsOf(job({ done: 2 })), a, 700)
    expect(b.strands.music.since).toBe(100)
    expect(b.idx).toBe(a.idx + 1) // the ring moved: a reduced-motion scene redraws
  })

  it('draws nothing for an app that is not on the web', () => {
    expect(idle(cardsOf(job({ app: 'weather' }))).strands).toEqual({})
  })
})
