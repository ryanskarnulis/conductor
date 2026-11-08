import { apiClient, apiUrl } from './client'
import type {
  AgentMessage,
  Conversation,
  ConversationDetail,
  MessageExchange,
  TurnActivity,
} from '../types/agent'

// Posting a message runs the master loop synchronously: each iteration may
// wrap a full subagent loop, and a cold model load is ~100 s before the first
// token. Match the backend provider's own read timeout, not the 30 s default.
const AGENT_RUN_TIMEOUT_MS = 300_000

export async function listConversations(): Promise<Conversation[]> {
  const res = await apiClient('/api/agent/conversations')
  return (await res.json()) as Conversation[]
}

export async function createConversation(): Promise<Conversation> {
  const res = await apiClient('/api/agent/conversations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  })
  return (await res.json()) as Conversation
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

export async function deleteConversation(id: number): Promise<void> {
  await apiClient(`/api/agent/conversations/${id}`, { method: 'DELETE' })
}

export async function postMessage(
  conversationId: number,
  content: string,
): Promise<MessageExchange> {
  const res = await apiClient(`/api/agent/conversations/${conversationId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
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
