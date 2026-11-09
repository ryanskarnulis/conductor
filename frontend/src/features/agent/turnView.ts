import type { PlanStepData, ReceiptData } from '../../design/components'
import { STEP_WORD, type PresenceState } from '../../design/status'
import type {
  AgentMessage,
  AgentStopReason,
  AppConfirm,
  AppRecordEntry,
  LiveReceipt,
  PlanBodyStep,
  ToolCallRecord,
  TurnActivity,
} from '../../types/agent'
import { activityLabel, appFromTool, lookedAtFromTool, openedAppFromTool } from './delegateCalls'
import { answerValue, fillParts, valueFromRecord, type RefValue } from './planRefs'
import { queuedPlanSteps, type TurnStream } from './turnStream'
import { JOB_NOTE } from '../../types/agent'

/** The shell's view of a conversation, derived from today's persisted turns
 * (messages, their delegate calls and the apps' typed records) plus the live
 * channel: the stream's steps and receipts (#38), or the poll's beat. Pure:
 * the page and the 3D scene both read this, so they can never disagree about
 * what happened. */

/** How an app is named on screen. `tasks` is PCC's fleet name. */
export function appLabel(app: string): string {
  return app === 'tasks' ? 'PCC' : app
}

/** What a paused plan waits on, as a card: conductor's own question (its
 * options are bare titles), or a write someone must say yes to. */
export type PauseView =
  | { kind: 'question'; question: string; options: string[] }
  | { kind: 'confirm'; appKey: string | null; app: string; question: string; why: string }

export interface ShellTurn {
  key: string
  /** What the person said, or the note they left from the UI. */
  said: string
  note: boolean
  /** Conductor's code-built ack of the plan, said before anything ran. */
  ack: string | null
  /** How this turn picked up a plan that was waiting on the person. */
  resumedNote: string | null
  steps: (PlanStepData & { appKey: string | null })[]
  receipts: (ReceiptData & { appKey: string })[]
  /** What the plan waits on, while it waits. */
  pause: PauseView | null
  /** One line for a screen reader: what the plan just did or waits on. */
  announce: string | null
  reply: string | null
  stopReason: AgentStopReason | null
  /** True while the turn is the optimistic, in-flight one. */
  live: boolean
  /** The reply is a background job's one note (#90), not an answer to anything. */
  jobNote?: boolean
  /** When a job's note was posted. */
  at?: string | null
  /** When the turn began (#82: the timeline's day separators). */
  when?: string | null
  /** The surface it came from (#82), null for the web page typed. */
  surface?: string | null
  /** Incognito (#82): its words and reply were never written. */
  incognito?: boolean
}

export interface PendingConfirmView {
  app: string
  question: string
}

/** The confirm card's line for a think's proposal (#78). */
export const PROPOSAL_QUESTION = 'Run the plan above?'

const VERBS: [RegExp, string][] = [
  [/^(create|add|new)/, 'Created'],
  [/^(update|edit|rename|set|reschedule|move)/, 'Updated'],
  [/^(trash|delete|remove|wipe)/, 'Trashed'],
  [/^complete/, 'Completed'],
  [/^reopen/, 'Reopened'],
  [/^restore/, 'Restored'],
  [/^(download|queue|grab|save)/, 'Queued'],
  [/^(sort|file)/, 'Filed'],
  [/^cancel/, 'Cancelled'],
]

function verbFor(entry: AppRecordEntry): string {
  if (entry.outcome === 'looked_up') return 'Found'
  if (entry.outcome === 'refused') return 'Refused'
  if (entry.outcome === 'not_run') return 'Not run'
  const hit = VERBS.find(([re]) => re.test(entry.tool))
  return hit ? hit[1] : 'Done'
}

/** A receipt chip from one record entry. Calls that never ran because they
 * wait on a confirmation are the confirmation card's business, not a receipt. */
export function receiptFrom(entry: AppRecordEntry, app: string): (ReceiptData & { appKey: string }) | null {
  if (entry.outcome === 'not_run' && entry.reason === 'needs confirmation') return null
  const titles = (entry.touched ?? [])
    .map((ref) => (ref.title ? `“${ref.title}”` : `${ref.kind} ${ref.id}`))
    .slice(0, 3)
  const listed = entry.touched?.length ?? 0
  const count = entry.count ?? null
  const parts: string[] = []
  if (count !== null && count > listed) parts.push(String(count))
  if (titles.length) parts.push(titles.join(', ') + (listed > titles.length ? ` and ${listed - titles.length} more` : ''))
  if (entry.outcome === 'refused' || entry.outcome === 'not_run') if (entry.reason) parts.push(entry.reason)
  const undo = entry.outcome === 'done' ? entry.undo : null
  return {
    appKey: app,
    verb: verbFor(entry),
    app: appLabel(app),
    entity: parts.join(' · '),
    undoHow: undo && 'handle' in undo ? undo.how : undefined,
    noUndo: undo && 'none' in undo ? undo.none : undefined,
  }
}

type StepSource = Pick<ToolCallRecord, 'tool' | 'arguments' | 'error'> & { app_confirm?: unknown }

/** An error text the loop writes for a call it never dispatched (backend
 * `run_record.classify`): that step didn't run, which is not the same as failing. */
const NEVER_RAN = ['not run: ', 'Invalid arguments: ', 'Unknown tool ']

/** A call's end, from its error: failed, or not run (with its reason). */
function endedBadly(error: string | null): { state: 'failed' | 'not_run'; detail: string } | null {
  if (error === null) return null
  const prefix = NEVER_RAN.find((p) => error.startsWith(p))
  if (prefix === undefined) return { state: 'failed', detail: error }
  return { state: 'not_run', detail: prefix === NEVER_RAN[0] ? error.slice(prefix.length) : error }
}

function stepFrom(call: StepSource, waiting: boolean): ShellTurn['steps'][number] {
  const asked = appFromTool(call.tool)
  const opened = openedAppFromTool(call.tool)
  const bad = endedBadly(call.error)
  const message = typeof call.arguments.message === 'string' ? call.arguments.message : ''
  const outcome = (done: ShellTurn['steps'][number]['state']) => ({ state: bad ? bad.state : done, detail: bad?.detail })
  if (asked !== null) {
    const appKey = call.tool.slice('ask_'.length)
    return { appKey, kind: 'ask', app: appLabel(asked), text: message || `ask ${asked}`, ...outcome(call.app_confirm ? (waiting ? 'wait' : 'not_run') : 'done') }
  }
  if (opened !== null) {
    const appKey = call.tool.slice('open_'.length)
    return { appKey, kind: 'open', app: appLabel(opened), text: typeof call.arguments.intent === 'string' ? call.arguments.intent : `open ${opened}`, ...outcome('done') }
  }
  if (call.tool === 'confirm') {
    const app = typeof call.arguments.app === 'string' ? call.arguments.app : ''
    return { appKey: app || null, kind: 'confirm', app: appLabel(app || 'conductor'), text: 'yes', ...outcome('done') }
  }
  const looked = lookedAtFromTool(call.tool)
  if (looked !== null) {
    const appKey = call.tool.slice('look_'.length)
    return { appKey, kind: 'look', app: appLabel(looked), text: `look at ${looked}`, ...outcome('done') }
  }
  if (call.tool === 'list_agents') return { appKey: null, kind: 'look', app: 'conductor', text: 'check the fleet', ...outcome('done') }
  // A think (#78) detached as a job: it runs on, its proposal comes later.
  if (call.tool === 'think') return { appKey: null, kind: 'think', app: 'conductor', text: typeof call.arguments.problem === 'string' ? call.arguments.problem : 'think it over', ...outcome('job') }
  return { appKey: null, kind: 'local', app: 'conductor', text: call.tool, ...outcome('done') }
}

/** A landed step as the record it is: the same derivation as a stored turn. */
function recordOf(receipt: LiveReceipt): StepSource & { app_record: AppRecordEntry[] } {
  return {
    tool: receipt.tool,
    arguments: receipt.arguments,
    error: receipt.error,
    app_confirm: receipt.confirm,
    app_record: receipt.receipts.filter((entry) => entry.source === 'app'),
  }
}

/** A plan step no call has started yet. Its words keep any `{sN}` dashed
 * until that step's receipt exists. */
function queuedFrom(step: PlanBodyStep, waiting: boolean, values: Record<string, RefValue>): ShellTurn['steps'][number] {
  const app = typeof step.app === 'string' ? step.app : null
  const said =
    step.do === 'ask' ? step.request : step.do === 'open' ? step.intent : step.do === 'ask_user' ? step.question : step.do === 'confirm' ? step.answer : undefined
  const words = typeof said === 'string' && said ? fillParts(said, values) : { text: step.do, parts: undefined }
  return {
    appKey: app ? app.replace(/-/g, '_') : null,
    kind: step.do,
    app: appLabel(app ?? 'conductor'),
    text: words.text,
    parts: words.parts,
    state: waiting ? 'wait' : 'queued',
  }
}

/** What each earlier step gave later ones as `{sN}`, only from receipts that
 * exist (the backend's `from_record`), plus an `ask_user` answer once resumed. */
function refValues(stream: TurnStream): Record<string, RefValue> {
  const values: Record<string, RefValue> = {}
  const plan = stream.plan
  if (plan) {
    for (const step of stream.steps) {
      const receipt = step.receipt
      const id = plan.steps[plan.start + step.seq - 1]?.id
      if (!receipt || !id || (receipt.outcome !== 'done' && receipt.outcome !== 'looked_up')) continue
      if (receipt.kind !== 'ask' && receipt.kind !== 'look' && receipt.kind !== 'open') continue
      const value = valueFromRecord(receipt.receipts.filter((entry) => entry.source === 'app'))
      if (value) values[id] = value
    }
  }
  if (stream.resumed?.reason === 'ask_user' && stream.resumed.answer !== null) values[stream.resumed.step ?? 's1'] = answerValue(stream.resumed.answer, stream.resumed.untrusted ?? false)
  return values
}

/** Whether a landed step started long work that is still going (#76). */
function detached(receipt: LiveReceipt): boolean {
  return receipt.receipts.some((entry) => entry.source === 'app' && entry.job != null && !['completed', 'failed', 'canceled'].includes(entry.job.status))
}

/** The in-flight turn's steps from the stream: each landed step from its
 * receipt, a started one as running, and the plan's remaining steps as
 * queued. Nothing is shown done before its receipt arrived. */
export function streamSteps(stream: TurnStream): ShellTurn['steps'] {
  const waitingOnApp = stream.paused?.reason === 'app_confirm'
  const steps = stream.steps.map((step) => {
    if (step.receipt) {
      const landed = stepFrom(recordOf(step.receipt), waitingOnApp)
      // A step whose receipt carries a job still running went to the
      // background: queued, not done (#90).
      return landed.state === 'done' && detached(step.receipt) ? { ...landed, state: 'job' as const } : landed
    }
    return { ...stepFrom({ tool: step.tool ?? '', arguments: step.arguments ?? {}, error: null }, false), state: 'run' as const }
  })
  const paused = stream.paused && stream.paused.reason !== 'app_confirm' ? stream.paused.step : null
  const values = refValues(stream)
  return [...steps, ...queuedPlanSteps(stream).map((step) => queuedFrom(step, step.id === paused, values))]
}

/** What the stream's plan waits on, as a card. */
export function streamPause(stream: TurnStream): PauseView | null {
  const paused = stream.paused
  if (!paused) return null
  if (paused.reason === 'ask_user') {
    const step = stream.plan?.steps.find((s) => s.id === paused.step)
    const question = typeof step?.question === 'string' && step.question ? step.question : (paused.question ?? '')
    if (!question) return null
    return { kind: 'question', question, options: (step?.options ?? []).filter((o): o is string => typeof o === 'string' && o !== '') }
  }
  if (!paused.question) return null
  const app = appLabel(paused.app ?? 'conductor')
  const why = paused.reason === 'gate' ? `Conductor checks before asking ${app}. Say “yes”, or tap.` : `${app} asks before this write. Say “yes”, or tap.`
  return { kind: 'confirm', appKey: paused.app ? paused.app.replace(/-/g, '_') : null, app, question: paused.question, why }
}

/** How a resumed turn says it picked the plan up again. */
export function streamResumedNote(stream: TurnStream): string | null {
  const resumed = stream.resumed
  if (!resumed) return null
  if (resumed.reason === 'ask_user') return resumed.answer ? `Resumed the plan · you answered ${resumed.answer}` : 'Resumed the plan · you answered'
  if (resumed.reason === 'gate') return 'Resumed the plan · you said yes'
  if (resumed.reason === 'proposal') return 'Running the proposed plan · you said yes'
  return 'You said yes to the app’s question'
}

/** What a screen reader hears as the plan moves: the question it waits on, or
 * the latest step that is past queued, in the same words as the panel. */
function announceFor(steps: ShellTurn['steps'], pause: PauseView | null): string | null {
  if (pause) {
    const options = pause.kind === 'question' && pause.options.length > 0 ? ` ${pause.options.join(', or ')}?` : ''
    return `Waiting on you: ${pause.question}${options}`
  }
  if (steps.length === 0) return null
  let at = -1
  steps.forEach((step, i) => {
    if (step.state !== 'queued') at = i
  })
  if (at === -1) return `Plan: ${steps.length} step${steps.length === 1 ? '' : 's'}, none started`
  const step = steps[at]
  return `Step ${at + 1} of ${steps.length}, ${step.app}: ${STEP_WORD[step.state]}${step.detail ? ` · ${step.detail}` : ''}`
}

/** The receipt chips the stream's landed steps carry. */
export function streamReceipts(stream: TurnStream): ShellTurn['receipts'] {
  return stream.steps.flatMap((step) => {
    const receipt = step.receipt
    if (!receipt?.app || (receipt.kind !== 'ask' && receipt.kind !== 'confirm')) return []
    const app = receipt.app
    return recordOf(receipt)
      .app_record.map((entry) => receiptFrom(entry, app))
      .filter((r): r is NonNullable<typeof r> => r !== null)
  })
}

/** The newest message that is part of the conversation itself: a background
 * job's note (#76) is neither said nor answered, so it never takes the place
 * of the last turn (it doesn't void a waiting question on the server either). */
export function lastExchange(messages: AgentMessage[]): AgentMessage | undefined {
  for (let i = messages.length - 1; i >= 0; i--) if (messages[i].stop_reason !== JOB_NOTE) return messages[i]
  return undefined
}

/** The question the conversation's last turn leaves waiting, if any — the
 * frontend twin of backend `confirmation.pending`: a later call to the same
 * app in that turn voids it. */
export function pendingConfirm(messages: AgentMessage[]): (PendingConfirmView & { raw: Pick<AppConfirm, 'app'> }) | null {
  const last = lastExchange(messages)
  if (!last || last.role !== 'assistant') return null
  // A think's proposal (#78), posted as its job's note: the reply lists the
  // plan, and a yes runs exactly that.
  if ((last.stop_reason as string | null) === 'think_proposal') return { app: 'conductor', question: PROPOSAL_QUESTION, raw: { app: 'conductor' } }
  let live: AppConfirm | null = null
  for (const call of last.tool_calls ?? []) {
    const app = call.tool.startsWith('ask_') ? call.tool.slice(4) : call.tool === 'confirm' ? String(call.arguments.app ?? '') : null
    if (live !== null && app === live.app) live = null
    if (call.app_confirm) live = call.app_confirm
  }
  return live ? { app: appLabel(live.app), question: live.question, raw: live } : null
}

/** Turns, oldest first: each user message with the assistant reply that
 * answered it (a note has none). The in-flight send is appended as live: its
 * steps and receipts from the stream when there is one, else the poll's beat. */
export function buildTurns(
  messages: AgentMessage[],
  pendingText: string | null,
  activity: TurnActivity | null,
  stream: TurnStream | null = null,
  held: TurnStream | null = null,
): ShellTurn[] {
  const turns: ShellTurn[] = []
  const waitingOn = pendingConfirm(messages)
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i]
    // A reply with no message of yours before it (history that starts
    // mid-thread) still shows, as a turn with nothing said.
    const orphan = m.role === 'assistant'
    const next = orphan ? m : messages[i + 1]
    // A job's note (#76) answers nothing: it is never the reply to what you said.
    const reply = next && next.role === 'assistant' && (orphan || next.stop_reason !== JOB_NOTE) ? next : null
    const isLast = reply !== null && reply === lastExchange(messages)
    const calls = reply?.tool_calls ?? []
    // The turn that ended on a pause keeps the plan the stream drew, so the
    // plan doesn't vanish the moment its question appears.
    const kept = held !== null && isLast && reply !== null && held.paused !== null && held.ended?.message_id === reply.id ? held : null
    const pause = kept ? streamPause(kept) : null
    const steps = kept ? streamSteps(kept) : calls.map((call) => stepFrom(call, isLast && waitingOn !== null))
    turns.push({
      key: `m${m.id}`,
      said: orphan ? '' : (m.content ?? ''),
      note: !orphan && reply === null,
      ack: kept?.activity?.ack ?? null,
      resumedNote: kept ? streamResumedNote(kept) : null,
      steps,
      pause,
      announce: kept ? announceFor(steps, pause) : null,
      receipts: calls.flatMap((call) => {
        const app = appFromTool(call.tool)
          ? call.tool.slice(4)
          : lookedAtFromTool(call.tool)
            ? call.tool.slice('look_'.length)
            : call.tool === 'confirm'
              ? String(call.arguments.app ?? '')
              : null
        if (!app) return []
        return (call.app_record ?? []).map((entry) => receiptFrom(entry, app)).filter((r): r is NonNullable<typeof r> => r !== null)
      }),
      reply: reply?.content ?? null,
      stopReason: reply?.stop_reason === JOB_NOTE ? null : (reply?.stop_reason ?? null),
      live: false,
      jobNote: reply?.stop_reason === JOB_NOTE,
      at: reply?.stop_reason === JOB_NOTE ? reply.created_at : null,
      when: m.created_at,
      surface: m.surface ?? null,
      incognito: m.incognito === true,
    })
    if (reply && !orphan) i++
  }
  if (pendingText !== null && stream !== null && (stream.steps.length > 0 || stream.plan !== null)) {
    const steps = streamSteps(stream)
    const pause = streamPause(stream)
    turns.push({
      key: 'pending',
      said: pendingText,
      note: false,
      ack: stream.activity?.ack ?? activity?.ack ?? null,
      resumedNote: streamResumedNote(stream),
      steps,
      receipts: streamReceipts(stream),
      pause,
      announce: announceFor(steps, pause),
      reply: null,
      stopReason: null,
      live: true,
    })
  } else if (pendingText !== null) {
    const steps: ShellTurn['steps'] = []
    if (activity?.kind === 'tool' && activity.tool) {
      const step = stepFrom({ tool: activity.tool, arguments: {}, error: null }, false)
      steps.push({ ...step, state: 'run', text: activityLabel(activity).replace(/…$/, '') })
    }
    turns.push({ key: 'pending', said: pendingText, note: false, ack: activity?.ack ?? null, resumedNote: null, steps, receipts: [], pause: null, announce: null, reply: null, stopReason: null, live: true })
  }
  return turns
}

export interface PresenceView {
  state: PresenceState
  label: string
}

/** One state word for what Glitch is doing, from the same data as the panel. */
export function presenceFor(sending: boolean, activity: TurnActivity | null, waiting: boolean): PresenceView {
  if (sending) {
    if (activity?.kind === 'tool') return { state: 'work', label: activityLabel(activity).replace(/…$/, '') }
    return { state: 'think', label: 'Thinking' }
  }
  if (waiting) return { state: 'wait', label: 'Waiting on you' }
  return { state: 'idle', label: 'Ready' }
}
