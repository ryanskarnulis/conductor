import { apiClient, apiUrl } from './client'
import type { Heard } from '../voice/tts'
import type {
  AgentMessage,
  ConversationDetail,
  JobCancelResult,
  MessageExchange,
  Surface,
  ThreadClosed,
  ThreadInfo,
  Timeline,
  TurnActivity,
} from '../types/agent'

// Posting a message runs the master loop synchronously: each iteration may
// wrap a full subagent loop, and a cold model load is ~100 s before the first
// token. Match the backend provider's own read timeout, not the 30 s default.
const AGENT_RUN_TIMEOUT_MS = 300_000

/** The one timeline (#82), created on first use: every surface writes to it. */
export async function getTimeline(): Promise<Timeline> {
  const res = await apiClient('/api/agent/timeline')
  return (await res.json()) as Timeline
}

/** Record something the person did in the UI (filing a song from the sort
 * panel). No model runs — it appends the turn and returns it, so a click that
 * bypassed the loop still shows up in the thread as the answer it was. */
export async function postNote(id: number, content: string): Promise<AgentMessage> {
  const res = await apiClient(`/api/agent/conversations/${id}/notes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  })
  return (await res.json()) as AgentMessage
}

export async function getConversation(id: number): Promise<ConversationDetail> {
  const res = await apiClient(`/api/agent/conversations/${id}`)
  return (await res.json()) as ConversationDetail
}

/** How a turn is sent: the surface it came from, and incognito (#82): its
 * words and reply are never written; the apps still see the request. */
export interface SendOptions {
  surface?: Surface
  incognito?: boolean
  /** Barge-in (#77): how much of the last spoken reply the person heard
   * before they spoke over it, as the voice client reports it. */
  heard?: Heard
}

/** A heard report as the API takes it: the voice client's fields, in snake case. */
function heardBody(heard: Heard) {
  return {
    played_ms: heard.playedMs,
    total_ms: heard.totalMs,
    heard_chars: heard.heardChars,
    total_chars: heard.totalChars,
  }
}

export async function postMessage(
  conversationId: number,
  content: string,
  options: SendOptions = {},
): Promise<MessageExchange> {
  const { heard, ...rest } = options
  const res = await apiClient(`/api/agent/conversations/${conversationId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, ...rest, ...(heard ? { heard: heardBody(heard) } : {}) }),
    timeoutMs: AGENT_RUN_TIMEOUT_MS,
  })
  return (await res.json()) as MessageExchange
}

export async function getTurnActivity(conversationId: number): Promise<TurnActivity> {
  const res = await apiClient(`/api/agent/conversations/${conversationId}/activity`)
  return (await res.json()) as TurnActivity
}

/** The conversation's live channel (#38): Server-Sent Events for the turn in
 * flight. `useTurnActivity` opens it with `EventSource`. */
export function turnEventsUrl(conversationId: number): string {
  return apiUrl(`/api/agent/conversations/${conversationId}/events`)
}

/** Stop the conversation's in-flight turn (#66). Idempotent; the blocked
 * `postMessage` then answers `cancelled` with the record of what ran. */
export async function cancelTurn(conversationId: number): Promise<{ cancel_requested: boolean }> {
  const res = await apiClient(`/api/agent/conversations/${conversationId}/cancel`, { method: 'POST' })
  return (await res.json()) as { cancel_requested: boolean }
}

/** Stop a background job from its card (#90): the app's cancel, no model, and
 * the tap recorded as the person's own note. The result carries the job's
 * `done` count at the moment of the cancel. */
export async function cancelJob(conversationId: number, app: string, ref: string): Promise<JobCancelResult> {
  const res = await apiClient(`/api/agent/conversations/${conversationId}/jobs/cancel`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ app, ref }),
  })
  return (await res.json()) as JobCancelResult
}

/** The side threads still open (#139), oldest first. */
export async function listThreads(): Promise<ThreadInfo[]> {
  const res = await apiClient('/api/agent/threads')
  return (await res.json()) as ThreadInfo[]
}

/** Start a side thread carrying what the timeline's window touched and saved.
 * No model runs. Incognito keeps no name. */
export async function startThread(title: string | null, incognito = false): Promise<ThreadInfo> {
  const res = await apiClient('/api/agent/threads', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...(title ? { title } : {}), ...(incognito ? { incognito: true } : {}) }),
  })
  return (await res.json()) as ThreadInfo
}

export async function getThread(id: number): Promise<ThreadInfo> {
  const res = await apiClient(`/api/agent/threads/${id}`)
  return (await res.json()) as ThreadInfo
}

/** Close a side thread: its result, narrated from its receipts, is posted
 * into the timeline. The narrator runs, so this waits like a turn. */
export async function closeThread(id: number, surface?: Surface): Promise<ThreadClosed> {
  const res = await apiClient(`/api/agent/threads/${id}/close`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(surface ? { surface } : {}),
    timeoutMs: AGENT_RUN_TIMEOUT_MS,
  })
  return (await res.json()) as ThreadClosed
}
