/** The status vocabulary every Conductor surface shares (design system #86):
 * one state, always written as a word, never colour alone. */
export type StepState = 'queued' | 'run' | 'job' | 'wait' | 'done' | 'failed' | 'cancelled' | 'not_run'

export const STEP_WORD: Record<StepState, string> = {
  queued: 'queued',
  run: 'running',
  job: 'running · job',
  wait: 'waiting on you',
  done: 'done',
  failed: 'failed',
  cancelled: 'cancelled',
  not_run: 'not run',
}

export type PresenceState = 'idle' | 'listen' | 'think' | 'work' | 'wait' | 'speak' | 'failed' | 'done'
