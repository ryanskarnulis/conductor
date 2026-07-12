import { useCallback, useEffect, useState } from 'react'
import { getConversation, postMessage } from '../../api/agent'
import { ApiError } from '../../api/client'
import type { ConversationDetail } from '../../types/agent'

interface UseConversation {
  detail: ConversationDetail | null
  loading: boolean
  error: string | null
  /** The optimistic user bubble + progress indicator while a run is in flight. */
  pendingText: string | null
  send: (content: string) => Promise<boolean>
}

/** Human-readable failure line for a `postMessage` run. */
export function sendErrorMessage(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 429) {
      return 'Rate limited — give conductor a moment before sending more.'
    }
    const detail = (e.body as { detail?: unknown } | null)?.detail
    if (typeof detail === 'string') return detail
  }
  return e instanceof Error ? e.message : 'The conductor run failed'
}

// All state is tagged with the conversation it belongs to and the exposed
// values are derived by id match — switching conversations "resets" the view
// without effect-time setState, and a slow send's refetch can never clobber a
// thread the user has since navigated away from. (PCC's pattern.)
interface Tagged<T> {
  id: number
  value: T
}

function forId<T>(tagged: Tagged<T> | null, id: number | null): T | null {
  return tagged !== null && tagged.id === id ? tagged.value : null
}

/**
 * One open conversation: history plus the send → loop → exchange round trip.
 *
 * The run is synchronous server-side (no streaming in v1), so `send` keeps an
 * optimistic user bubble + progress state up while it waits, then refetches
 * the thread from the server — the source of truth for what was persisted
 * (on failure the user turn may or may not have landed; the refetch shows
 * exactly what did). `onExchange` lets the page refresh the sidebar's
 * recency/title.
 */
export function useConversation(
  conversationId: number | null,
  onExchange?: () => void,
): UseConversation {
  const [loaded, setLoaded] = useState<Tagged<ConversationDetail> | null>(null)
  const [errorState, setErrorState] = useState<Tagged<string> | null>(null)
  const [pending, setPending] = useState<Tagged<string> | null>(null)

  const detail = forId(loaded, conversationId)
  const error = forId(errorState, conversationId)
  const pendingText = forId(pending, conversationId)
  const loading = conversationId !== null && detail === null && error === null

  useEffect(() => {
    if (conversationId === null) return
    let active = true
    getConversation(conversationId)
      .then((data) => {
        if (active) setLoaded({ id: conversationId, value: data })
      })
      .catch((e: unknown) => {
        if (active) {
          setErrorState({
            id: conversationId,
            value: e instanceof Error ? e.message : 'Failed to load conversation',
          })
        }
      })
    return () => {
      active = false
    }
  }, [conversationId])

  const send = useCallback(
    async (content: string): Promise<boolean> => {
      if (conversationId === null) return false
      setErrorState(null)
      setPending({ id: conversationId, value: content })
      let ok = true
      try {
        await postMessage(conversationId, content)
      } catch (e: unknown) {
        ok = false
        setErrorState({ id: conversationId, value: sendErrorMessage(e) })
      }
      // Success or not, the server is the source of truth for the thread
      // (on failure the user turn may or may not have been persisted).
      try {
        const fresh = await getConversation(conversationId)
        setLoaded({ id: conversationId, value: fresh })
      } catch {
        // The send error (if any) is already surfaced; keep it.
      }
      setPending((prev) => (prev !== null && prev.id === conversationId ? null : prev))
      onExchange?.()
      return ok
    },
    [conversationId, onExchange],
  )

  return { detail, loading, error, pendingText, send }
}
