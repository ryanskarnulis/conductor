import { useCallback, useEffect, useState } from 'react'
import { answerCards, getConversation, postMessage, postNote, type SendOptions } from '../../api/agent'
import { ApiError } from '../../api/client'
import type { CardsAnswer, ConversationDetail, Surface } from '../../types/agent'
import { handoffFrom, type Handoff } from './handoff'
import { shownText } from './turnView'

interface UseConversation {
  detail: ConversationDetail | null
  loading: boolean
  error: string | null
  /** The optimistic user bubble + progress indicator while a run is in flight. */
  pendingText: string | null
  /** Set when the turn just answered asked to hand the user over to an app
   * (`open_<app>`) — the page navigates there. Only ever set by a live `send`,
   * never by loading history: reopening an old conversation must not fling the
   * user back into chess. */
  handoff: Handoff | null
  /** Resolves to the assistant's reply text on success (possibly ''), or
   * null when the run failed — voice uses the text to speak the reply. */
  send: (content: string, options?: SendOptions) => Promise<string | null>
  /** This session's incognito turns (#82): message id → the words or reply
   * the person read live. The server never wrote them; a reload forgets them. */
  liveWords: ReadonlyMap<number, string | null>
  /** Record an action taken in the UI as a turn of its own. No model runs —
   * the sort panel files through the app directly, and the thread still has to
   * show that the question was answered. */
  note: (content: string) => Promise<void>
  /** Read the thread again: a job's note (#90) was posted without a send. */
  refresh: () => Promise<void>
  /** Tap cards (#251): one, or a group's "Approve all". The only way
   * anything is approved. `resumes` is the tap as the thread will record it,
   * when it picks a waiting plan up: the turn then shows live, like a send.
   * The thread is read again either way; null when the tap failed. */
  tap: (cards: string[], answer: 'yes' | 'no', how?: { option?: number; surface?: Surface; resumes?: string }) => Promise<CardsAnswer | null>
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
  const [handoffState, setHandoffState] = useState<Tagged<Handoff> | null>(null)
  // Incognito turns' words and replies, by message id: this page's memory only.
  const [liveWords, setLiveWords] = useState<ReadonlyMap<number, string | null>>(() => new Map())

  const detail = forId(loaded, conversationId)
  const error = forId(errorState, conversationId)
  const pendingText = forId(pending, conversationId)
  const handoff = forId(handoffState, conversationId)
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
    async (content: string, options?: SendOptions): Promise<string | null> => {
      if (conversationId === null) return null
      setErrorState(null)
      setPending({ id: conversationId, value: content })
      let reply: string | null = null
      try {
        const exchange = options ? await postMessage(conversationId, content, options) : await postMessage(conversationId, content)
        // A voice turn says an app's passed answer in its short form (#241).
        reply = exchange.assistant_message.spoken ?? exchange.assistant_message.content ?? ''
        if (exchange.user_message.incognito) {
          // Incognito (#82): the server wrote "(incognito)". The person reads
          // the words and the reply in this page only, until it reloads; an
          // app's answer (#241) isn't stored either, so it rides the reply.
          const { user_message: said, assistant_message: answered } = exchange
          const shown = shownText(answered.content, answered.tool_calls)
          setLiveWords((prev) => new Map(prev).set(said.id, said.content).set(answered.id, shown))
        }
        // Read the handoff off the live turn only. It rides on the persisted
        // trajectory, so history has it too — but replaying history must not
        // re-navigate, which is why it's captured here and not derived from
        // `detail`.
        const target = handoffFrom(exchange.assistant_message.tool_calls)
        if (target) setHandoffState({ id: conversationId, value: target })
      } catch (e: unknown) {
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
      return reply
    },
    [conversationId, onExchange],
  )

  const note = useCallback(
    async (content: string): Promise<void> => {
      if (conversationId === null) return
      try {
        await postNote(conversationId, content)
        const fresh = await getConversation(conversationId)
        setLoaded({ id: conversationId, value: fresh })
        onExchange?.()
      } catch (e: unknown) {
        // The action itself already succeeded — only its record failed, so this
        // must not read as "the song was not filed".
        setErrorState({
          id: conversationId,
          value: `Filed, but the note did not reach the thread: ${sendErrorMessage(e)}`,
        })
      }
    },
    [conversationId, onExchange],
  )

  const refresh = useCallback(async (): Promise<void> => {
    if (conversationId === null) return
    try {
      const fresh = await getConversation(conversationId)
      setLoaded({ id: conversationId, value: fresh })
      onExchange?.()
    } catch {
      // The thread stays as it was; the next event reads it again.
    }
  }, [conversationId, onExchange])

  const tap = useCallback(
    async (cards: string[], answer: 'yes' | 'no', how: { option?: number; surface?: Surface; resumes?: string } = {}): Promise<CardsAnswer | null> => {
      if (conversationId === null) return null
      const { resumes, ...options } = how
      setErrorState(null)
      if (resumes) setPending({ id: conversationId, value: resumes })
      let done: CardsAnswer | null = null
      try {
        done = await answerCards(conversationId, cards, answer, options)
        const handed = done.exchanges.map((exchange) => handoffFrom(exchange.assistant_message.tool_calls)).find((target) => target)
        if (handed) setHandoffState({ id: conversationId, value: handed })
      } catch (e: unknown) {
        // Nothing is claimed: the card stays as the server has it.
        setErrorState({ id: conversationId, value: sendErrorMessage(e) })
      }
      await refresh()
      setPending((prev) => (prev !== null && prev.id === conversationId ? null : prev))
      return done
    },
    [conversationId, refresh],
  )

  return { detail, loading, error, pendingText, handoff, send, liveWords, note, refresh, tap }
}
