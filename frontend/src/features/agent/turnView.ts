import type { PlanStepData, ReceiptData } from '../../design/components'
import type { PresenceState } from '../../design/status'
import type {
  AgentMessage,
  AgentStopReason,
  AppConfirm,
  AppRecordEntry,
  ToolCallRecord,
  TurnActivity,
} from '../../types/agent'
import { activityLabel, appFromTool, openedAppFromTool } from './delegateCalls'

/** The shell's view of a conversation, derived from today's persisted turns
 * (messages, their delegate calls and the apps' typed records) plus the live
 * activity poll. Pure: the page and the 3D scene both read this, so they can
 * never disagree about what happened. */

/** How an app is named on screen. `tasks` is PCC's fleet name. */
export function appLabel(app: string): string {
  return app === 'tasks' ? 'PCC' : app
}

export interface ShellTurn {
  key: string
  /** What the person said, or the note they left from the UI. */
  said: string
  note: boolean
  steps: (PlanStepData & { appKey: string | null })[]
  receipts: (ReceiptData & { appKey: string })[]
  reply: string | null
  stopReason: AgentStopReason | null
  /** True while the turn is the optimistic, in-flight one. */
  live: boolean
}

export interface PendingConfirmView {
  app: string
  question: string
}

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

function stepFrom(call: ToolCallRecord, waiting: boolean): ShellTurn['steps'][number] {
  const asked = appFromTool(call.tool)
  const opened = openedAppFromTool(call.tool)
  const failed = call.error !== null
  const message = typeof call.arguments.message === 'string' ? call.arguments.message : ''
  if (asked !== null) {
    const appKey = call.tool.slice('ask_'.length)
    const state = failed ? 'failed' : call.app_confirm ? (waiting ? 'wait' : 'not_run') : 'done'
    return { appKey, kind: 'ask', app: appLabel(asked), text: message || `ask ${asked}`, state, detail: failed ? (call.error ?? undefined) : undefined }
  }
  if (opened !== null) {
    const appKey = call.tool.slice('open_'.length)
    return { appKey, kind: 'open', app: appLabel(opened), text: typeof call.arguments.intent === 'string' ? call.arguments.intent : `open ${opened}`, state: failed ? 'failed' : 'done' }
  }
  if (call.tool === 'confirm') {
    const app = typeof call.arguments.app === 'string' ? call.arguments.app : ''
    return { appKey: app || null, kind: 'confirm', app: appLabel(app || 'conductor'), text: 'yes', state: failed ? 'failed' : 'done' }
  }
  if (call.tool === 'list_agents') return { appKey: null, kind: 'look', app: 'conductor', text: 'check the fleet', state: failed ? 'failed' : 'done' }
  return { appKey: null, kind: 'local', app: 'conductor', text: call.tool, state: failed ? 'failed' : 'done' }
}

/** The question the conversation's last turn leaves waiting, if any — the
 * frontend twin of backend `confirmation.pending`: a later call to the same
 * app in that turn voids it. */
export function pendingConfirm(messages: AgentMessage[]): (PendingConfirmView & { raw: AppConfirm }) | null {
  const last = messages[messages.length - 1]
  if (!last || last.role !== 'assistant') return null
  let live: AppConfirm | null = null
  for (const call of last.tool_calls ?? []) {
    const app = call.tool.startsWith('ask_') ? call.tool.slice(4) : call.tool === 'confirm' ? String(call.arguments.app ?? '') : null
    if (live !== null && app === live.app) live = null
    if (call.app_confirm) live = call.app_confirm
  }
  return live ? { app: appLabel(live.app), question: live.question, raw: live } : null
}

/** Turns, oldest first: each user message with the assistant reply that
 * answered it (a note has none). The in-flight send is appended as live. */
export function buildTurns(messages: AgentMessage[], pendingText: string | null, activity: TurnActivity | null): ShellTurn[] {
  const turns: ShellTurn[] = []
  const waitingOn = pendingConfirm(messages)
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i]
    // A reply with no message of yours before it (history that starts
    // mid-thread) still shows, as a turn with nothing said.
    const orphan = m.role === 'assistant'
    const next = orphan ? m : messages[i + 1]
    const reply = next && next.role === 'assistant' ? next : null
    const isLast = reply !== null && reply === messages[messages.length - 1]
    const calls = reply?.tool_calls ?? []
    turns.push({
      key: `m${m.id}`,
      said: orphan ? '' : (m.content ?? ''),
      note: !orphan && reply === null,
      steps: calls.map((call) => stepFrom(call, isLast && waitingOn !== null)),
      receipts: calls.flatMap((call) => {
        const app = appFromTool(call.tool) ? call.tool.slice(4) : call.tool === 'confirm' ? String(call.arguments.app ?? '') : null
        if (!app) return []
        return (call.app_record ?? []).map((entry) => receiptFrom(entry, app)).filter((r): r is NonNullable<typeof r> => r !== null)
      }),
      reply: reply?.content ?? null,
      stopReason: reply?.stop_reason ?? null,
      live: false,
    })
    if (reply && !orphan) i++
  }
  if (pendingText !== null) {
    const steps: ShellTurn['steps'] = []
    if (activity?.kind === 'tool' && activity.tool) {
      const step = stepFrom({ tool: activity.tool, arguments: {}, result: null, error: null }, false)
      steps.push({ ...step, state: 'run', text: activityLabel(activity).replace(/…$/, '') })
    }
    turns.push({ key: 'pending', said: pendingText, note: false, steps, receipts: [], reply: null, stopReason: null, live: true })
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
