import type { JobEvent, TurnSnapshot } from '../../types/agent'
import { appLabel, type ShellTurn } from './turnView'

/** The jobs a conversation started, folded from the live channel (#90): the
 * snapshot's running jobs, then each `job` event. Pure, like `turnStream`:
 * `useJobs` feeds it, `jobCards` reads it. It holds only what the server
 * sent, so a card can never run ahead of its events. */
export interface JobsState {
  lastId: number
  /** A turn is in flight (another tab's, or this one's before it ended). */
  active: boolean
  /** Running jobs, plus jobs that just ended and whose note isn't on screen yet. */
  jobs: JobEvent[]
  /** One line for a screen reader, set when a job appears, changes status or ends. */
  announce: string | null
  /** Jobs the last snapshot no longer lists though they were running: they
   * ended while no stream was open, so their notes are on the server. */
  dropped: JobEvent[]
}

export const JOB_EVENTS = ['snapshot', 'job', 'turn_started', 'turn_ended'] as const
export type JobStreamEvent = (typeof JOB_EVENTS)[number]

export const EMPTY_JOBS: JobsState = { lastId: 0, active: false, jobs: [], announce: null, dropped: [] }

export const jobKey = (job: Pick<JobEvent, 'app' | 'ref'>) => `${job.app}:${job.ref}`

/** What a job is, in one phrase. Only an event's own words make a job
 * finished, failed or stopped: a tap on Cancel only ever says `stopping`. */
export type JobWord = 'queued' | 'working' | 'waiting on you' | 'stopping' | 'stopped' | 'ending' | 'finished' | 'failed' | 'lost track of it'

export function jobWord(job: JobEvent, asked: boolean): JobWord {
  if (job.ended) {
    if (job.how === 'lost' || job.how === 'gone') return 'lost track of it'
    if (job.status === 'completed') return 'finished'
    if (job.status === 'failed') return 'failed'
    return job.status === 'canceled' ? 'stopped' : 'lost track of it'
  }
  switch (job.status) {
    case 'submitted':
      return 'queued'
    case 'input_required':
      return 'waiting on you'
    case 'canceled':
      return 'stopped'
    case 'completed':
    case 'failed':
      return 'ending'
    default:
      return asked || job.cancel_requested ? 'stopping' : 'working'
  }
}

const fmt = (n: number) => n.toLocaleString('en-US')

/** `164 of 1,480`, or `164 done` when the total isn't known. */
export function jobProgress(job: Pick<JobEvent, 'done' | 'total'>): string {
  return job.total !== null ? `${fmt(job.done)} of ${fmt(job.total)}` : `${fmt(job.done)} done`
}

/** An untrusted or trusted title as a plain label: one line, no control
 * characters, clipped. Rendered as text, never markup. */
export function jobTitle(job: Pick<JobEvent, 'app' | 'title'>): string {
  // eslint-disable-next-line no-control-regex
  const flat = (job.title ?? '').replace(/[\u0000-\u001f\u007f]+/g, ' ').replace(/\s+/g, ' ').trim()
  if (flat === '') return `${appLabel(job.app)} job`
  return flat.length > 80 ? `${flat.slice(0, 79)}…` : flat
}

function sentence(job: JobEvent, asked = false): string {
  const label = job.untrusted && job.title ? `“${jobTitle(job)}”` : jobTitle(job)
  const word = jobWord(job, asked)
  return `${appLabel(job.app)}: ${label}, ${word}${job.ended ? '' : `, ${jobProgress(job)}`}.`
}

function upsert(jobs: JobEvent[], job: JobEvent): JobEvent[] {
  const at = jobs.findIndex((j) => jobKey(j) === jobKey(job))
  return at === -1 ? [...jobs, job] : jobs.map((j, i) => (i === at ? job : j))
}

/** The state after one event. A snapshot replaces the jobs (and is always
 * taken: after a restart event ids start over); any other event older than
 * what was applied is ignored. */
export function applyJobEvent(state: JobsState, name: JobStreamEvent, data: unknown, id: number): JobsState {
  if (id <= state.lastId && name !== 'snapshot') return state
  const next = { ...state, lastId: id, dropped: [] }
  switch (name) {
    case 'snapshot': {
      const snap = data as TurnSnapshot
      const jobs = (snap.jobs ?? []).map((job) => ({ ...job, ended: false }))
      const kept = new Set(jobs.map(jobKey))
      return { ...next, active: snap.active, jobs, dropped: state.jobs.filter((j) => !j.ended && !kept.has(jobKey(j))) }
    }
    case 'turn_started':
      return { ...next, active: true }
    case 'turn_ended':
      return { ...next, active: false }
    case 'job': {
      const job = data as JobEvent
      const before = state.jobs.find((j) => jobKey(j) === jobKey(job))
      // A cancel's answer says so once; later polls that omit it don't undo it.
      const merged = before?.cancel_requested && !job.ended ? { ...job, cancel_requested: job.cancel_requested ?? true } : job
      const changed = !before || before.status !== merged.status || before.ended !== merged.ended
      return { ...next, jobs: upsert(state.jobs, merged), announce: changed ? sentence(merged) : state.announce }
    }
  }
}

/** Whether the stream still has something to follow. */
export function following(state: JobsState): boolean {
  return state.active || state.jobs.some((job) => !job.ended)
}

export type CardState = 'queued' | 'run' | 'wait' | 'ok' | 'bad' | 'stop'

export interface JobCardData {
  key: string
  app: string
  appName: string
  ref: string
  title: string
  /** The title is an outside app's text: shown as a label, quoted. */
  outside: boolean
  word: JobWord
  state: CardState
  done: number
  total: number | null
  progress: string
  /** 0..1 when the total is known. */
  ratio: number | null
  ended: boolean
  noteId: number | null
  /** The app was asked to stop (by the tap), or says it is stopping. */
  asking: 'idle' | 'sending' | 'asked'
  cancelable: boolean
}

/** The thread's turns in order, each run of job notes collected into one
 * group: ten jobs ending at once read as one quiet block, not ten. */
export type TurnItem = { kind: 'turn'; turn: ShellTurn; index: number } | { kind: 'notes'; turns: ShellTurn[]; key: string }

export function groupJobNotes(turns: ShellTurn[]): TurnItem[] {
  const items: TurnItem[] = []
  turns.forEach((turn, index) => {
    const last = items[items.length - 1]
    if (!turn.jobNote) items.push({ kind: 'turn', turn, index })
    else if (last?.kind === 'notes') last.turns.push(turn)
    else items.push({ kind: 'notes', turns: [turn], key: `notes-${turn.key}` })
  })
  return items
}

const STATE_OF: Record<JobWord, CardState> = {
  queued: 'queued',
  working: 'run',
  'waiting on you': 'wait',
  stopping: 'stop',
  stopped: 'stop',
  ending: 'run',
  finished: 'ok',
  failed: 'bad',
  'lost track of it': 'bad',
}

/** The cards for the jobs the stream knows, oldest first. `asks` is where the
 * person's taps on Cancel stand (`sending` until the app answers, then
 * `asked`); `shown` are the message ids on screen, so a job that ended
 * leaves once its note has arrived. Every word and number is the last
 * event's: the derivation never runs ahead of what the stream said. */
export function jobCards(state: JobsState, asks: Readonly<Record<string, 'sending' | 'asked'>>, shown: ReadonlySet<number>): JobCardData[] {
  return state.jobs
    .filter((job) => !(job.ended && job.note_message_id != null && shown.has(job.note_message_id)))
    .map((job) => {
      const key = jobKey(job)
      const ask = asks[key]
      const word = jobWord(job, ask === 'asked')
      return {
        key,
        app: job.app,
        appName: appLabel(job.app),
        ref: job.ref,
        title: jobTitle(job),
        outside: job.untrusted && job.title !== null,
        word,
        state: STATE_OF[word],
        done: job.done,
        total: job.total,
        progress: jobProgress(job),
        ratio: job.total !== null && job.total > 0 ? Math.min(1, job.done / job.total) : null,
        ended: job.ended,
        noteId: job.note_message_id ?? null,
        asking: ask ?? (job.cancel_requested && !job.ended ? 'asked' : 'idle'),
        cancelable: !job.ended && job.status !== 'canceled' && job.status !== 'completed' && job.status !== 'failed',
      }
    })
}
