import { useCallback, useEffect, useState } from 'react'
import { createConversation, deleteConversation, listConversations } from '../../api/agent'
import type { Conversation } from '../../types/agent'

interface UseConversations {
  conversations: Conversation[]
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
  create: () => Promise<Conversation>
  remove: (id: number) => Promise<void>
}

/** The sidebar list: active conversations, most recently touched first. */
export function useConversations(): UseConversations {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setConversations(await listConversations())
      setError(null)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load conversations')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    listConversations()
      .then((data) => {
        if (active) setConversations(data)
      })
      .catch((e: unknown) => {
        if (active) {
          setError(e instanceof Error ? e.message : 'Failed to load conversations')
        }
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  const create = useCallback(async () => {
    const conversation = await createConversation()
    await refresh()
    return conversation
  }, [refresh])

  const remove = useCallback(
    async (id: number) => {
      await deleteConversation(id)
      await refresh()
    },
    [refresh],
  )

  return { conversations, loading, error, refresh, create, remove }
}
