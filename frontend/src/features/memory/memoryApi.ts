import { apiClient } from '../../api/client'
import type { AgentMessage } from '../../types/agent'

/** Where a saved fact came from (#81): the turn, what the person said then
 * (null once that turn was forgotten), when, and in which conversation. */
export interface FactSource {
  turn_id: number
  kind: 'message' | 'note' | string
  words: string | null
  said_at: string
  surface: string | null
  thread: { id: number; title: string | null } | null
}

export interface MemoryFact {
  id: number
  text: string
  created_at: string
  updated_at: string
  /** null: changed in the memory list outside any conversation. */
  source: FactSource | null
}

export interface FactChange {
  fact: MemoryFact | null
  /** The person's note turn, when a conversation was open. */
  message: AgentMessage | null
}

export async function listFacts(): Promise<MemoryFact[]> {
  const res = await apiClient('/api/memory/facts')
  return (await res.json()) as MemoryFact[]
}

export async function editFact(id: number, text: string, conversationId: number | null): Promise<FactChange> {
  const res = await apiClient(`/api/memory/facts/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, conversation_id: conversationId }),
  })
  return (await res.json()) as FactChange
}

export async function forgetFact(id: number, conversationId: number | null): Promise<FactChange> {
  const query = conversationId !== null ? `?conversation_id=${conversationId}` : ''
  const res = await apiClient(`/api/memory/facts/${id}${query}`, { method: 'DELETE' })
  return (await res.json()) as FactChange
}
