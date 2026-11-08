// Mirrors backend/app/schemas/conversations.py.

export interface Conversation {
  id: number
  title: string | null
  created_at: string
  updated_at: string
}

/** One dispatched tool call persisted on an assistant message; exactly one of
 * result/error is set. Mirrors the backend's ToolCallRecord. */
export interface ToolCallRecord {
  tool: string
  arguments: Record<string, unknown>
  result: string | null
  error: string | null
  /** For a delegate call, the tools the *app* ran while answering. Persisted
   * for the UI and never sent to the model — it is how a panel can tell what a
   * turn actually did without parsing the reply text for it. */
  app_tools?: string[] | null
  /** The app's typed record of what its tools did (delegate v1.1): one entry
   * per tool call. Persisted for the UI, never sent to the model. */
  app_record?: AppRecordEntry[] | null
  /** The question the app is waiting on (#68), with its single-use token.
   * Persisted and shown, never sent to the model. */
  app_confirm?: AppConfirm | null
}

/** One entity a tool call changed or found (delegate v1.1 `touched`). The
 * title is the app's text: render it as text, never as markup. */
export interface TouchedRef {
  app: string
  kind: string
  id: string
  title?: string | null
  /** Set when the title is outside text (a web title, a file's tags). */
  untrusted?: boolean
}

/** One entry of an app's typed record (agent-standard delegate-api v1.1). */
export interface AppRecordEntry {
  tool: string
  outcome: 'done' | 'refused' | 'looked_up' | 'not_run' | string
  touched?: TouchedRef[] | null
  count?: number | null
  reason?: string | null
  undo?: { handle: string; how: string } | { none: string } | null
  untrusted?: boolean
  /** Set when the call started long work that went on after the turn (#76). */
  job?: { ref: string; status: string; done?: number | null; total?: number | null } | null
}

/** What conductor keeps of an app's `confirm_required` (backend AppConfirm). */
export interface AppConfirm {
  app: string
  thread_id: number
  token: string
  tool: string
  question: string
  expires_at?: string | null
}

/** STANDARD.md §3's stop vocabulary, the same strings in every app. */
export type AgentStopReason =
  | 'completed'
  | 'max_iterations'
  | 'correction_limit'
  | 'truncated'
  | 'no_progress'
  | 'tool_call_limit'
  | 'cancelled'
  | 'provider_error'
  | 'timed_out'
  | 'internal_error'

/** The `stop_reason` of the one note a background job posts when it ends (#76). */
export const JOB_NOTE = 'job_note'

export interface AgentMessage {
  id: number
  conversation_id: number
  role: 'user' | 'assistant'
  content: string | null
  tool_calls: ToolCallRecord[] | null
  stop_reason: AgentStopReason | typeof JOB_NOTE | null
  created_at: string
}

export interface ConversationDetail extends Conversation {
  messages: AgentMessage[]
}

export interface MessageExchange {
  user_message: AgentMessage
  assistant_message: AgentMessage
}

/** What the in-flight turn is doing right now (GET …/activity — the poll
 * fallback while a send blocks; the stream is `GET …/events`). All fields but `active` are
 * null when the conversation has no turn in flight. */
export interface TurnActivity {
  active: boolean
  kind: 'model' | 'tool' | null
  tool: string | null
  iteration: number | null
  elapsed_seconds: number | null
  /** A compiled plan's honest ack (#70): what's about to happen, never what's done. */
  ack?: string | null
}

// --- The live channel (#38): GET …/events, mirrors backend app/api/turn_activity.py.

/** One beat on the stream (`turn_started`, `activity`, a snapshot's `activity`). */
export interface TurnBeat {
  kind: 'model' | 'tool'
  tool: string | null
  iteration: number
  ack: string | null
  elapsed_seconds: number
}

/** One step of a validated plan, as the planner wrote it (backend `plan.py`). */
export interface PlanBodyStep {
  id: string
  do: string
  app?: string
  request?: string
  intent?: string
  question?: string
  answer?: string
  /** An `ask_user` step's offered choices: bare titles. */
  options?: string[]
  [field: string]: unknown
}

/** A step as the ledger journals it (backend `ledger.live_step`): sent only
 * once its record exists. `receipts` are the app's own entries, or
 * conductor's `unknown` one for an app call with no readable record. */
export interface LiveReceipt {
  seq: number
  kind: string
  app: string | null
  tool: string
  arguments: Record<string, unknown>
  outcome: 'done' | 'failed' | 'looked_up' | 'not_run'
  error: string | null
  receipts: (AppRecordEntry & { source: 'app' | 'conductor' })[]
  /** An app's confirm question, without its token. */
  confirm: { app: string; tool: string; question: string } | null
}

/** The `seq`-th call of the turn: started, and its receipt once it landed. */
export interface LiveStep {
  seq: number
  tool?: string
  arguments?: Record<string, unknown>
  receipt: LiveReceipt | null
}

export interface LivePlan {
  steps: PlanBodyStep[]
  /** The plan step this turn's first call runs (a resumed plan starts past 0). */
  start: number
}

export interface TurnPause {
  reason: 'ask_user' | 'gate' | 'app_confirm'
  question: string | null
  app: string | null
  step: string | null
}

/** This turn picks up a plan that was waiting on the person (#89). `step` is
 * the step it waited on (null for an app's own question); `answer` is an
 * `ask_user` answer exactly as `{s1}` will read it. */
export interface TurnResumed {
  reason: 'ask_user' | 'gate' | 'app_confirm'
  step: string | null
  answer: string | null
}

export interface TurnSnapshot {
  active: boolean
  activity?: TurnBeat
  plan?: LivePlan | null
  steps?: LiveStep[]
  paused?: TurnPause | null
  resumed?: TurnResumed | null
  /** The jobs still running (#76, #90), whether or not a turn is in flight. */
  jobs?: JobEvent[]
}

/** A background job as the stream reports it (backend `job` event, #76):
 * its A2A status, how far it got, and once it ended, the note's message id.
 * `title` is the app's text when `untrusted`: a label, never markup. */
export interface JobEvent {
  app: string
  ref: string
  status: 'submitted' | 'working' | 'input_required' | 'completed' | 'failed' | 'canceled' | string
  done: number
  total: number | null
  title: string | null
  untrusted: boolean
  ended: boolean
  /** On an end: `ended` (the app said so), or `lost`/`gone` (conductor lost track). */
  how?: 'ended' | 'lost' | 'gone' | null
  note_message_id?: number | null
  /** Set by a card's Cancel: the app was asked to stop. */
  cancel_requested?: boolean
}

/** The app's answer to a card's Cancel (backend `JobCancelRead`). */
export interface JobCancelResult {
  app: string
  ref: string
  status: string
  done: number
  total: number | null
  cancel_requested: boolean
  message: AgentMessage
}

export interface TurnEnded {
  stop_reason: AgentStopReason | null
  message_id: number | null
}
