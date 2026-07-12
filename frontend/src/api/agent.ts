import { apiClient } from './client'
import type {
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
