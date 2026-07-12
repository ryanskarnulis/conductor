import { useEffect, useState } from 'react'
import { getTurnActivity } from '../../api/agent'
import type { TurnActivity } from '../../types/agent'

// v1 has no SSE: while a send blocks, this is the progress channel. The
// backend beat changes at provider-turn / tool-dispatch granularity (seconds
// apart at minimum), so a relaxed poll is plenty.
export const TURN_ACTIVITY_POLL_MS = 1_500

/**
 * Polls `GET …/activity` while `active` (a send is in flight) and returns the
 * latest in-flight beat, or null before the first one lands. Poll failures
 * keep the previous beat — a blip must not blank the progress line mid-run.
 *
 * The stored beat is tagged with its conversation and the return value is
 * derived (`active` + id match), so going idle or switching threads clears
 * the line without effect-time setState.
 */
export function useTurnActivity(
  conversationId: number | null,
  active: boolean,
): TurnActivity | null {
  const [beat, setBeat] = useState<{ id: number; value: TurnActivity } | null>(null)

  useEffect(() => {
    if (!active || conversationId === null) return
    let live = true
    const poll = () => {
      getTurnActivity(conversationId)
        .then((value) => {
          // A not-active response means the run just finished (or hasn't hit
          // the registry yet) — keep the last beat; the send resolving tears
          // the whole indicator down.
          if (live && value.active) setBeat({ id: conversationId, value })
        })
        .catch(() => {
          // Keep the previous beat; the POST owns failure reporting.
        })
    }
    poll()
    const timer = setInterval(poll, TURN_ACTIVITY_POLL_MS)
    return () => {
      live = false
      clearInterval(timer)
    }
  }, [conversationId, active])

  return active && beat !== null && beat.id === conversationId ? beat.value : null
}
