import type { PresenceState } from '../../design/status'
import type { JobCardData } from './jobStream'
import type { ShellTurn } from './turnView'

/** What the Living Web scene reads each frame: the same facts as the panel,
 * reduced to strands per app plus Glitch's pose. A strand's `since` is when
 * it entered its state (ms, the scene's clock), so the scene can play the
 * transition once and then hold.
 *
 * A settled strand lets go (#232): once its turn has ended (or its job has),
 * the result holds for `HOLD_MS`, then the strand carries `release` (when it
 * began to let go) for `RELEASE_MS`, then it is gone and the web is at rest.
 * A turn still running or waiting on the person keeps every strand. */

/** How long a finished turn's strands and done marks stay before letting go. */
export const HOLD_MS = 4000
/** How long the letting go takes: the scene's animation fits inside it. */
export const RELEASE_MS = 900

export type StrandState = 'queued' | 'run' | 'job' | 'wait' | 'done' | 'failed'

export interface SceneState {
  pres: 'idle' | 'listen' | 'think' | 'work' | 'wait' | 'speak'
  /** `k` is the strand's place on the plan, so queued strands reach out one after another. */
  strands: Record<string, { state: StrandState; since: number; k?: number; release?: number }>
  /** When each settled strand came to rest (its turn or job over), kept after
   * it is gone so the same result never re-attaches. */
  rest: Record<string, { state: StrandState; t: number }>
  recall: { state: 'run' | 'done'; since: number } | null
  /** The job Glitch hums over (`of` 0: its total isn't known). */
  job: { n: number; of: number; app: string } | null
  /** The progress ring on each app's anchor, from its running jobs with a known total (#90). */
  jobs: Record<string, { n: number; of: number }>
  confirm: { app: string } | null
  lastStart: { app: string; t: number } | null
  lastReceipt: { app: string; t: number } | null
  lastFail: { app: string; t: number } | null
  /** The apps whose step failed, which is what Glitch reads to glitch. */
  steps: Record<string, { state: 'failed' }>
  screen: boolean
  /** Changes whenever anything visible changes: the reduced-motion scene redraws on it. */
  idx: number
}

/** The fleet apps that have a place on the web. */
export const WEB_APPS = ['tasks', 'music', 'chess'] as const

const PRES: Record<PresenceState, SceneState['pres']> = {
  idle: 'idle',
  listen: 'listen',
  think: 'think',
  work: 'work',
  wait: 'wait',
  speak: 'speak',
  failed: 'idle',
  done: 'idle',
}

function strandFor(state: ShellTurn['steps'][number]['state']): StrandState | null {
  switch (state) {
    case 'run':
      return 'run'
    case 'wait':
      return 'wait'
    case 'done':
      return 'done'
    case 'failed':
      return 'failed'
    case 'job':
      return 'job'
    case 'queued':
      return 'queued'
    default:
      return null
  }
}

/** What a job card makes of its app's strand, from its own words only: queued
 * is not working, and nothing is done before the end event says so. */
function strandOfJob(job: JobCardData): StrandState | null {
  switch (job.word) {
    case 'queued':
      return 'queued'
    case 'working':
    case 'stopping':
    case 'ending':
      return 'job'
    case 'waiting on you':
      return 'wait'
    case 'finished':
      return 'done'
    case 'failed':
    case 'lost track of it':
      return 'failed'
    default:
      return null
  }
}

const JOB_RANK: Record<StrandState, number> = { failed: 6, wait: 5, job: 4, queued: 3, done: 2, run: 0 }

/** Next scene state from the latest turn and the conversation's job cards,
 * carrying `since` forward for any strand whose state did not change. A
 * running turn's own strand wins; a job lights the strand of an app the turn
 * has left done or never touched, with a progress ring (#90). Pure given `now`. */
export function nextSceneState(
  prev: SceneState | null,
  latest: ShellTurn | null,
  presence: PresenceState,
  confirmApp: string | null,
  now: number,
  jobs: readonly JobCardData[] = [],
): SceneState {
  const strands: SceneState['strands'] = {}
  let lastStart: SceneState['lastStart'] = prev?.lastStart ?? null
  let lastReceipt: SceneState['lastReceipt'] = prev?.lastReceipt ?? null
  let lastFail: SceneState['lastFail'] = prev?.lastFail ?? null
  const failed: SceneState['steps'] = {}
  for (const step of latest?.steps ?? []) {
    const app = step.appKey
    if (!app || !(WEB_APPS as readonly string[]).includes(app)) continue
    const state = strandFor(step.state)
    if (!state) continue
    // A step still queued never dims a strand an earlier step already lit:
    // the strand shows the latest step that has really begun.
    if (state === 'queued' && strands[app] && strands[app].state !== 'queued') continue
    const before = prev?.strands[app]
    const since = before && before.state === state ? before.since : now
    strands[app] = state === 'queued' ? { state, since, k: Object.keys(strands).length } : { state, since }
    if (state === 'failed') failed[app] = { state: 'failed' }
    if (since === now && state === 'run') lastStart = { app, t: now }
    if (since === now && (state === 'done' || state === 'failed')) lastReceipt = { app, t: now }
    if (since === now && state === 'failed') lastFail = { app, t: now }
  }
  const lit: Record<string, StrandState> = {}
  const fromJob = new Set<string>()
  const rings: SceneState['jobs'] = {}
  for (const job of jobs) {
    const state = strandOfJob(job)
    if (!state || !(WEB_APPS as readonly string[]).includes(job.app)) continue
    if (!lit[job.app] || JOB_RANK[state] > JOB_RANK[lit[job.app]]) lit[job.app] = state
    if (state === 'job' && job.total !== null) {
      const ring = rings[job.app] ?? { n: 0, of: 0 }
      rings[job.app] = { n: ring.n + job.done, of: ring.of + job.total }
    }
  }
  for (const [app, state] of Object.entries(lit)) {
    const have = strands[app]
    if (have && have.state !== 'done') continue
    const before = prev?.strands[app]
    const since = before && before.state === state ? before.since : now
    strands[app] = state === 'queued' ? { state, since, k: Object.keys(strands).length } : { state, since }
    fromJob.add(app)
    if (since === now && state === 'done') lastReceipt = { app, t: now }
    if (since === now && state === 'failed') {
      failed[app] = { state: 'failed' }
      lastFail = { app, t: now }
    }
  }
  // A strand that only says "this is over" lets go after a hold. The turn's
  // own strands settle when the turn ends with nothing waiting on the person;
  // a job's settles with its end event.
  const ended = !!latest && !latest.live && latest.pause === null && confirmApp === null
  const rest: SceneState['rest'] = {}
  for (const [app, strand] of Object.entries(strands)) {
    if (strand.state !== 'done' && strand.state !== 'failed') continue
    if (!fromJob.has(app) && !ended) continue
    const before = prev?.rest[app]
    const at = before && before.state === strand.state ? before.t : now
    rest[app] = { state: strand.state, t: at }
    if (now >= at + HOLD_MS + RELEASE_MS) delete strands[app]
    else if (now >= at + HOLD_MS) strand.release = at + HOLD_MS
  }
  const hum = jobs.find((job) => strandOfJob(job) === 'job')
  const pres = PRES[presence]
  const changed =
    !prev ||
    prev.pres !== pres ||
    (prev.confirm?.app ?? null) !== confirmApp ||
    Object.keys(strands).length !== Object.keys(prev.strands).length ||
    JSON.stringify(rings) !== JSON.stringify(prev.jobs) ||
    Object.entries(strands).some(([app, s]) => prev.strands[app]?.state !== s.state || prev.strands[app]?.release !== s.release)
  return {
    pres,
    strands,
    rest,
    recall: null,
    job: hum ? { n: hum.done, of: hum.total ?? 0, app: hum.app } : null,
    jobs: rings,
    confirm: confirmApp ? { app: confirmApp } : null,
    lastStart,
    lastReceipt,
    lastFail,
    steps: failed,
    screen: true,
    idx: (prev?.idx ?? 0) + (changed ? 1 : 0),
  }
}

/** When the scene state next changes by the clock alone (a hold ending, a
 * release finishing), or null when nothing is pending. */
export function nextSceneWake(state: SceneState, now: number): number | null {
  let wake: number | null = null
  for (const app of Object.keys(state.strands)) {
    const rest = state.rest[app]
    if (!rest) continue
    const at = now < rest.t + HOLD_MS ? rest.t + HOLD_MS : rest.t + HOLD_MS + RELEASE_MS
    if (wake === null || at < wake) wake = at
  }
  return wake
}
