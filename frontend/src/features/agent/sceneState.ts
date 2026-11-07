import type { PresenceState } from '../../design/status'
import type { ShellTurn } from './turnView'

/** What the Living Web scene reads each frame: the same facts as the panel,
 * reduced to strands per app plus Glitch's pose. A strand's `since` is when
 * it entered its state (ms, the scene's clock), so the scene can play the
 * transition once and then hold. */

export type StrandState = 'queued' | 'run' | 'job' | 'wait' | 'done' | 'failed'

export interface SceneState {
  pres: 'idle' | 'listen' | 'think' | 'work' | 'wait' | 'speak'
  strands: Record<string, { state: StrandState; since: number }>
  recall: { state: 'run' | 'done'; since: number } | null
  job: { n: number; of: number; app: string } | null
  confirm: { app: string } | null
  lastStart: { app: string; t: number } | null
  lastReceipt: { app: string; t: number } | null
  lastFail: null
  steps: Record<string, never>
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

/** Next scene state from the latest turn, carrying `since` forward for any
 * strand whose state did not change. Pure given `now`. */
export function nextSceneState(prev: SceneState | null, latest: ShellTurn | null, presence: PresenceState, confirmApp: string | null, now: number): SceneState {
  const strands: SceneState['strands'] = {}
  let lastStart: SceneState['lastStart'] = prev?.lastStart ?? null
  let lastReceipt: SceneState['lastReceipt'] = prev?.lastReceipt ?? null
  for (const step of latest?.steps ?? []) {
    const app = step.appKey
    if (!app || !(WEB_APPS as readonly string[]).includes(app)) continue
    const state = strandFor(step.state)
    if (!state) continue
    const before = prev?.strands[app]
    const since = before && before.state === state ? before.since : now
    strands[app] = { state, since }
    if (since === now && state === 'run') lastStart = { app, t: now }
    if (since === now && (state === 'done' || state === 'failed')) lastReceipt = { app, t: now }
  }
  const pres = PRES[presence]
  const changed =
    !prev ||
    prev.pres !== pres ||
    (prev.confirm?.app ?? null) !== confirmApp ||
    Object.keys(strands).length !== Object.keys(prev.strands).length ||
    Object.entries(strands).some(([app, s]) => prev.strands[app]?.state !== s.state)
  return {
    pres,
    strands,
    recall: null,
    job: null,
    confirm: confirmApp ? { app: confirmApp } : null,
    lastStart,
    lastReceipt,
    lastFail: null,
    steps: {},
    screen: true,
    idx: (prev?.idx ?? 0) + (changed ? 1 : 0),
  }
}
