import { describe, expect, it } from 'vitest'
import type { JobEvent } from '../../types/agent'
import {
  applyJobEvent,
  EMPTY_JOBS,
  following,
  groupJobNotes,
  jobCards,
  jobTitle,
  jobWord,
  type JobsState,
  type JobStreamEvent,
} from './jobStream'
import type { ShellTurn } from './turnView'

const job = (over: Partial<JobEvent> = {}): JobEvent => ({
  app: 'music',
  ref: '118',
  status: 'working',
  done: 0,
  total: 40,
  title: 'Late Night Drive',
  untrusted: true,
  ended: false,
  ...over,
})

function fold(events: [JobStreamEvent, unknown][], from: JobsState = EMPTY_JOBS): JobsState {
  return events.reduce((state, [name, data], i) => applyJobEvent(state, name, data, from.lastId + i + 1), from)
}

const NONE = {}
const SHOWN = new Set<number>()

describe('applyJobEvent', () => {
  it('a snapshot lists the running jobs; progress events move them; the end keeps the job until its note shows', () => {
    const state = fold([
      ['snapshot', { active: false, jobs: [job()] }],
      ['job', job({ done: 12 })],
    ])
    expect(state.jobs).toHaveLength(1)
    expect(state.jobs[0].done).toBe(12)
    const ended = fold([['job', job({ done: 38, status: 'completed', ended: true, how: 'ended', note_message_id: 9 })]], state)
    expect(ended.jobs[0]).toMatchObject({ ended: true, status: 'completed', note_message_id: 9 })
    expect(following(ended)).toBe(false)
  })

  it('ignores an event older than what it has applied, except a snapshot (ids start over after a restart)', () => {
    const state = applyJobEvent(EMPTY_JOBS, 'job', job({ done: 20 }), 10)
    expect(applyJobEvent(state, 'job', job({ done: 5 }), 9)).toBe(state)
    const fresh = applyJobEvent(state, 'snapshot', { active: false, jobs: [job({ done: 3 })] }, 1)
    expect(fresh.jobs[0].done).toBe(3)
  })

  it('follows while a turn is in flight or a job runs, and stops when neither is left', () => {
    expect(following(EMPTY_JOBS)).toBe(false)
    expect(following(fold([['turn_started', {}]]))).toBe(true)
    expect(following(fold([['turn_started', {}], ['turn_ended', {}]]))).toBe(false)
    expect(following(fold([['snapshot', { active: false, jobs: [job()] }]]))).toBe(true)
  })

  it('a snapshot that drops a running job says it ended while nobody listened', () => {
    const before = fold([['snapshot', { active: false, jobs: [job(), job({ ref: '119' })] }]])
    const after = fold([['snapshot', { active: false, jobs: [job({ ref: '119' })] }]], before)
    expect(after.dropped.map((j) => j.ref)).toEqual(['118'])
    expect(applyJobEvent(after, 'job', job({ ref: '119', done: 4 }), 99).dropped).toEqual([])
  })

  it('announces a job appearing, changing status or ending, never a progress tick', () => {
    const queued = fold([['job', job({ status: 'submitted' })]])
    expect(queued.announce).toBe('music: “Late Night Drive”, queued, 0 of 40.')
    const working = fold([['job', job({ status: 'working', done: 1 })]], queued)
    expect(working.announce).toBe('music: “Late Night Drive”, working, 1 of 40.')
    const tick = fold([['job', job({ status: 'working', done: 2 })]], working)
    expect(tick.announce).toBe(working.announce)
    const done = fold([['job', job({ status: 'completed', done: 38, ended: true, note_message_id: 4 })]], tick)
    expect(done.announce).toBe('music: “Late Night Drive”, finished.')
  })

  it('keeps a Cancel’s answer when a later poll leaves it out', () => {
    const asked = fold([['job', job({ done: 5, cancel_requested: true })]])
    const polled = fold([['job', job({ done: 6 })]], asked)
    expect(polled.jobs[0].cancel_requested).toBe(true)
  })
})

describe('jobWord', () => {
  it('queued is not working, and only the end event makes a job finished, failed or stopped', () => {
    expect(jobWord(job({ status: 'submitted' }), false)).toBe('queued')
    expect(jobWord(job({ status: 'working' }), false)).toBe('working')
    expect(jobWord(job({ status: 'input_required' }), false)).toBe('waiting on you')
    // The app's report of a finished job before its end event is not "finished" yet.
    expect(jobWord(job({ status: 'completed' }), false)).toBe('ending')
    expect(jobWord(job({ status: 'failed' }), false)).toBe('ending')
    expect(jobWord(job({ status: 'completed', ended: true }), false)).toBe('finished')
    expect(jobWord(job({ status: 'failed', ended: true }), false)).toBe('failed')
    expect(jobWord(job({ status: 'canceled', ended: true }), false)).toBe('stopped')
    expect(jobWord(job({ status: 'working', ended: true, how: 'lost' }), false)).toBe('lost track of it')
    expect(jobWord(job({ status: 'working', ended: true, how: 'gone' }), false)).toBe('lost track of it')
  })

  it('a tap makes a working job "stopping", never "stopped"', () => {
    expect(jobWord(job(), true)).toBe('stopping')
    expect(jobWord(job({ status: 'submitted' }), true)).toBe('queued')
  })
})

describe('jobTitle', () => {
  it('shows an outside title as one plain line, clipped, never markup', () => {
    expect(jobTitle(job({ title: 'Mix\n\u0007<b>Hi</b>\t  2024 ' }))).toBe('Mix <b>Hi</b> 2024')
    expect(jobTitle(job({ title: 'x'.repeat(200) }))).toHaveLength(80)
    expect(jobTitle(job({ title: null }))).toBe('music job')
  })
})

describe('jobCards: a card never runs ahead of its events', () => {
  // One job's life as the stream tells it, with a Cancel tapped midway.
  const life: [JobStreamEvent, JobEvent | { active: boolean; jobs: JobEvent[] }][] = [
    ['snapshot', { active: false, jobs: [job({ status: 'submitted' })] }],
    ['job', job({ status: 'working', done: 0 })],
    ['job', job({ status: 'working', done: 12 })],
    ['job', job({ status: 'working', done: 12, cancel_requested: true })],
    ['job', job({ status: 'canceled', done: 12 })],
    ['job', job({ status: 'canceled', done: 12, ended: true, how: 'ended', note_message_id: 7 })],
  ]

  it('shows at every step exactly the last event’s words and numbers', () => {
    let state = EMPTY_JOBS
    const seen: string[] = []
    life.forEach(([name, data], i) => {
      state = applyJobEvent(state, name, data, i + 1)
      const [card] = jobCards(state, NONE, SHOWN)
      const last = (name === 'snapshot' ? (data as { jobs: JobEvent[] }).jobs[0] : data) as JobEvent
      expect(card.done).toBe(last.done)
      expect(card.total).toBe(last.total)
      // Finished, failed or stopped only on an event that says the job ended or was stopped.
      if (['finished', 'failed', 'stopped', 'lost track of it'].includes(card.word)) {
        expect(last.ended || last.status === 'canceled').toBe(true)
      }
      seen.push(card.word)
    })
    expect(seen).toEqual(['queued', 'working', 'working', 'stopping', 'stopped', 'stopped'])
  })

  it('never says finished before an event with status completed that ended', () => {
    for (const status of ['submitted', 'working', 'input_required', 'completed']) {
      const [card] = jobCards(fold([['job', job({ status })]]), NONE, SHOWN)
      expect(card.word).not.toBe('finished')
    }
    const [card] = jobCards(fold([['job', job({ status: 'completed', ended: true, note_message_id: 3 })]]), NONE, SHOWN)
    expect(card.word).toBe('finished')
  })

  it('a tap on Cancel changes nothing but the asking state, until the app answers', () => {
    const state = fold([['job', job({ done: 7 })]])
    const [plain] = jobCards(state, NONE, SHOWN)
    const [sending] = jobCards(state, { 'music:118': 'sending' }, SHOWN)
    const [asked] = jobCards(state, { 'music:118': 'asked' }, SHOWN)
    expect([plain.word, sending.word, asked.word]).toEqual(['working', 'working', 'stopping'])
    expect([plain.asking, sending.asking, asked.asking]).toEqual(['idle', 'sending', 'asked'])
    expect([sending.done, asked.done]).toEqual([7, 7])
  })

  it('a finished job leaves once its note is in the thread, and not before', () => {
    const state = fold([['job', job({ status: 'completed', done: 38, ended: true, note_message_id: 9 })]])
    expect(jobCards(state, NONE, new Set([1, 2]))).toHaveLength(1)
    expect(jobCards(state, NONE, new Set([1, 9]))).toHaveLength(0)
  })

  it('can’t stop a job that isn’t running, and has no ratio without a total', () => {
    const state = fold([
      ['job', job({ ref: '1', status: 'completed', ended: true, note_message_id: 3 })],
      ['job', job({ ref: '2', status: 'working', total: null, done: 5 })],
    ])
    const cards = jobCards(state, NONE, SHOWN)
    expect(cards.map((c) => c.cancelable)).toEqual([false, true])
    expect(cards[1]).toMatchObject({ ratio: null, progress: '5 done' })
  })

  it('writes progress with thousands separators and marks outside titles', () => {
    const [card] = jobCards(fold([['job', job({ done: 164, total: 1480 })]]), NONE, SHOWN)
    expect(card.progress).toBe('164 of 1,480')
    expect(card.outside).toBe(true)
    expect(jobCards(fold([['job', job({ untrusted: false })]]), NONE, SHOWN)[0].outside).toBe(false)
  })
})

describe('groupJobNotes', () => {
  const turn = (key: string, jobNote = false): ShellTurn => ({
    key,
    said: '',
    note: false,
    ack: null,
    resumedNote: null,
    steps: [],
    receipts: [],
    pause: null,
    announce: null,
    reply: key,
    stopReason: null,
    live: false,
    jobNote,
  })

  it('collects each run of job notes into one group and keeps real turns apart', () => {
    const items = groupJobNotes([turn('a'), turn('n1', true), turn('n2', true), turn('b'), turn('n3', true)])
    expect(items.map((i) => (i.kind === 'notes' ? i.turns.map((t) => t.key) : i.turn.key))).toEqual(['a', ['n1', 'n2'], 'b', ['n3']])
    const turns = items.filter((i) => i.kind === 'turn')
    expect(turns.map((i) => i.index)).toEqual([0, 3])
  })
})
