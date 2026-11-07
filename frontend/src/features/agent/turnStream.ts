import type {
  LivePlan,
  LiveReceipt,
  LiveStep,
  PlanBodyStep,
  TurnActivity,
  TurnBeat,
  TurnEnded,
  TurnPause,
  TurnSnapshot,
} from '../../types/agent'

/** The live channel's view of the turn in flight (#38), folded from
 * `GET …/events` one event at a time. Pure: `useTurnActivity` feeds it and
 * `turnView.buildTurns` reads it, so the panel and the scene see one source.
 * It holds only what the server sent: a step is landed only once its
 * receipt event arrived. */
export interface TurnStream {
  /** The newest event id applied; anything at or below it is old news. */
  lastId: number
  active: boolean
  activity: TurnBeat | null
  plan: LivePlan | null
  steps: LiveStep[]
  paused: TurnPause | null
  ended: TurnEnded | null
}

/** The event names the server sends, each a `MessageEvent` with JSON data. */
export const TURN_EVENTS = [
  'snapshot',
  'turn_started',
  'activity',
  'plan',
  'step_started',
  'receipt',
  'paused',
  'turn_ended',
] as const

export type TurnEventName = (typeof TURN_EVENTS)[number]

export const EMPTY_STREAM: TurnStream = {
  lastId: 0,
  active: false,
  activity: null,
  plan: null,
  steps: [],
  paused: null,
  ended: null,
}

function upsert(steps: LiveStep[], seq: number, patch: Partial<LiveStep>): LiveStep[] {
  const at = steps.findIndex((step) => step.seq === seq)
  if (at === -1) return [...steps, { seq, receipt: null, ...patch }].sort((a, b) => a.seq - b.seq)
  return steps.map((step, i) => (i === at ? { ...step, ...patch } : step))
}

/** The stream after one event. A snapshot replaces everything; an event
 * older than what was applied is ignored. */
export function applyTurnEvent(state: TurnStream, name: TurnEventName, data: unknown, id: number): TurnStream {
  if (id <= state.lastId) return state
  const next = { ...state, lastId: id }
  switch (name) {
    case 'snapshot': {
      const snap = data as TurnSnapshot
      return {
        ...EMPTY_STREAM,
        lastId: id,
        active: snap.active,
        activity: snap.activity ?? null,
        plan: snap.plan ?? null,
        steps: snap.steps ?? [],
        paused: snap.paused ?? null,
      }
    }
    case 'turn_started':
      return { ...EMPTY_STREAM, lastId: id, active: true, activity: data as TurnBeat }
    case 'activity':
      return { ...next, activity: data as TurnBeat }
    case 'plan':
      // A turn has one plan: the first announced stands.
      return next.plan ? next : { ...next, plan: data as LivePlan }
    case 'step_started': {
      const d = data as { seq: number; tool: string; arguments: Record<string, unknown> }
      return { ...next, steps: upsert(next.steps, d.seq, { tool: d.tool, arguments: d.arguments }) }
    }
    case 'receipt': {
      const d = data as LiveReceipt
      return { ...next, steps: upsert(next.steps, d.seq, { tool: d.tool, arguments: d.arguments, receipt: d }) }
    }
    case 'paused':
      return { ...next, paused: data as TurnPause }
    case 'turn_ended':
      return { ...next, active: false, ended: data as TurnEnded }
  }
}

/** The stream's beat in the poll's shape, with elapsed time as of `elapsed`. */
export function streamActivity(stream: TurnStream, elapsed: number | null): TurnActivity | null {
  if (!stream.active || stream.activity === null) return null
  const { kind, tool, iteration, ack } = stream.activity
  return { active: true, kind, tool, iteration, ack, elapsed_seconds: elapsed ?? stream.activity.elapsed_seconds }
}

/** The plan steps not yet started, in order (one call per step from `start`). */
export function queuedPlanSteps(stream: TurnStream): PlanBodyStep[] {
  if (stream.plan === null) return []
  return stream.plan.steps.slice(stream.plan.start + stream.steps.length)
}
