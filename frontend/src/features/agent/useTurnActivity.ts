import { useEffect, useState } from 'react'
import { getTurnActivity, turnEventsUrl } from '../../api/agent'
import type { TurnActivity } from '../../types/agent'
import { applyTurnEvent, EMPTY_STREAM, streamActivity, TURN_EVENTS, type TurnStream } from './turnStream'

// The poll fallback's pace. The backend beat changes at provider-turn /
// tool-dispatch granularity (seconds apart at minimum), so a relaxed poll is
// plenty.
export const TURN_ACTIVITY_POLL_MS = 1_500
// How often the stream path refreshes the elapsed counter (no network).
export const ELAPSED_TICK_MS = 1_000
// Errors in a row after which a stream that was working is given up for the poll.
export const STREAM_RETRIES = 3

/** What the page knows about the turn in flight. */
export interface TurnLive {
  /** The current beat, in the poll's shape, from whichever channel is live. */
  activity: TurnActivity | null
  /** The push channel's view (#38): the plan, each step's start and receipt,
   * a pause. Null on the poll path. */
  stream: TurnStream | null
}

/**
 * Follows the conversation's turn while `active` (a send is in flight).
 *
 * The live channel is `GET …/events` over `EventSource` (#38): a snapshot,
 * then each beat, plan, step and receipt as the backend publishes it. If the
 * browser has no `EventSource`, or the stream fails before it delivers
 * anything, closes, or keeps failing, this falls back to polling
 * `GET …/activity` — the beat only, as before. A reconnecting stream starts
 * from a fresh snapshot, so nothing is lost. Poll failures keep the previous
 * beat — a blip must not blank the progress line mid-run.
 *
 * The stored state is tagged with its conversation and the return value is
 * derived (`active` + id match), so going idle or switching threads clears
 * the line without effect-time setState.
 */
export function useTurnActivity(conversationId: number | null, active: boolean): TurnLive | null {
  const [live, setLive] = useState<{ id: number; value: TurnLive } | null>(null)

  useEffect(() => {
    if (!active || conversationId === null) return
    let closed = false
    let source: EventSource | null = null
    let poller: ReturnType<typeof setInterval> | null = null
    let ticker: ReturnType<typeof setInterval> | null = null
    let stream: TurnStream = EMPTY_STREAM
    // Client time the turn began, from the beat's own elapsed seconds.
    let startedAt: number | null = null

    const publish = () => {
      if (closed) return
      const elapsed = startedAt === null ? null : Math.round((Date.now() - startedAt) / 100) / 10
      setLive({ id: conversationId, value: { activity: streamActivity(stream, elapsed), stream } })
    }

    const poll = () => {
      getTurnActivity(conversationId)
        .then((value) => {
          // A not-active response means the run just finished (or hasn't hit
          // the registry yet) — keep the last beat; the send resolving tears
          // the whole indicator down.
          if (!closed && value.active) setLive({ id: conversationId, value: { activity: value, stream: null } })
        })
        .catch(() => {
          // Keep the previous beat; the POST owns failure reporting.
        })
    }

    const fallBack = () => {
      source?.close()
      source = null
      if (ticker !== null) clearInterval(ticker)
      ticker = null
      if (closed || poller !== null) return
      poll()
      poller = setInterval(poll, TURN_ACTIVITY_POLL_MS)
    }

    if (typeof EventSource === 'undefined') {
      fallBack()
    } else {
      const es = new EventSource(turnEventsUrl(conversationId))
      source = es
      let received = false
      let errors = 0
      const onEvent = (name: (typeof TURN_EVENTS)[number]) => (event: MessageEvent<string>) => {
        received = true
        errors = 0
        let data: unknown
        try {
          data = JSON.parse(event.data)
        } catch {
          return
        }
        const id = Number(event.lastEventId) || stream.lastId + 1
        const before = stream.activity
        stream = applyTurnEvent(stream, name, data, id)
        if (stream.activity !== before) startedAt = stream.activity ? Date.now() - stream.activity.elapsed_seconds * 1000 : null
        if (name === 'turn_ended') es.close()
        publish()
      }
      for (const name of TURN_EVENTS) es.addEventListener(name, onEvent(name) as EventListener)
      es.onerror = () => {
        errors += 1
        // Never worked, gave up, or keeps failing: the poll takes over.
        if (!received || es.readyState === EventSource.CLOSED || errors >= STREAM_RETRIES) fallBack()
      }
      ticker = setInterval(() => {
        if (stream.active) publish()
      }, ELAPSED_TICK_MS)
    }

    return () => {
      closed = true
      source?.close()
      if (poller !== null) clearInterval(poller)
      if (ticker !== null) clearInterval(ticker)
    }
  }, [conversationId, active])

  return active && live !== null && live.id === conversationId ? live.value : null
}
