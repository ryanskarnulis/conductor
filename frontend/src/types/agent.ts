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
}

/** One entry of an app's typed record (agent-standard delegate-api v1.1). */
export interface AppRecordEntry {
  tool: string
  outcome: 'done' | 'refused' | 'looked_up' | 'not_run' | string
  touched?: TouchedRef[] | null
  count?: number | null
  reason?: string | null
  undo?: { handle: string; how: string } | { none: string } | null
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

export interface AgentMessage {
  id: number
  conversation_id: number
  role: 'user' | 'assistant'
  content: string | null
  tool_calls: ToolCallRecord[] | null
  stop_reason: AgentStopReason | null
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
 * target while a send blocks; v1 has no SSE). All fields but `active` are
 * null when the conversation has no turn in flight. */
export interface TurnActivity {
  active: boolean
  kind: 'model' | 'tool' | null
  tool: string | null
  iteration: number | null
  elapsed_seconds: number | null
}
