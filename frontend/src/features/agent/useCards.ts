import { useCallback, useEffect, useState } from 'react'
import { getCards } from '../../api/agent'
import type { WaitingCard } from '../../types/agent'

interface UseCards {
  cards: WaitingCard[]
  /** Read the cards again: a turn ended, a card was tapped, a note landed. */
  refresh: () => Promise<void>
  /** Take the server's answer as the cards now open. */
  set: (cards: WaitingCard[]) => void
}

/**
 * The cards waiting on the person (#251), read from the server: it alone
 * says which questions are open (answered, replaced and lapsed ones are
 * gone). Read when the conversation opens and whenever `syncKey` changes
 * (the thread's newest message), and again when a card's own window closes.
 */
export function useCards(conversationId: number | null, syncKey: unknown): UseCards {
  const [state, setState] = useState<{ id: number; cards: WaitingCard[] } | null>(null)
  const cards = state !== null && state.id === conversationId ? state.cards : NONE

  const refresh = useCallback(async (): Promise<void> => {
    if (conversationId === null) return
    try {
      const fresh = await getCards(conversationId)
      setState({ id: conversationId, cards: fresh })
    } catch {
      // The cards stay as they were; the next change reads them again.
    }
  }, [conversationId])

  useEffect(() => {
    if (conversationId === null) return
    let active = true
    getCards(conversationId)
      .then((fresh) => {
        if (active) setState({ id: conversationId, cards: fresh })
      })
      .catch(() => {
        // The cards stay as they were; the next change reads them again.
      })
    return () => {
      active = false
    }
  }, [conversationId, syncKey])

  // A card whose token's window closes lapses without anything being said.
  const nextExpiry = Math.min(...cards.flatMap((card) => (card.expires_at ? [Date.parse(card.expires_at)] : [])))
  useEffect(() => {
    if (!Number.isFinite(nextExpiry)) return
    const timer = setTimeout(() => void refresh(), Math.max(0, nextExpiry - Date.now()) + 500)
    return () => clearTimeout(timer)
  }, [nextExpiry, refresh])

  const set = useCallback(
    (fresh: WaitingCard[]) => {
      if (conversationId !== null) setState({ id: conversationId, cards: fresh })
    },
    [conversationId],
  )
  return { cards, refresh, set }
}

const NONE: WaitingCard[] = []
