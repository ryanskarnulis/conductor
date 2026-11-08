import { useCallback, useEffect, useRef, useState } from 'react'
import { cancelJob, turnEventsUrl } from '../../api/agent'
import { applyJobEvent, EMPTY_JOBS, following, JOB_EVENTS, jobKey, type JobsState } from './jobStream'
import { sendErrorMessage } from './useConversation'

// Errors in a row after which a stream that keeps failing is given up.
export const JOB_STREAM_RETRIES = 3

const NONE: Readonly<Record<string, never>> = {}

interface Tagged<T> {
  id: number
  value: T
}

/** What the page knows about the jobs this conversation started (#90).
 *
 * While no send is in flight, the page follows the live channel
 * (`GET …/events`): its first snapshot lists the jobs still running (from the
 * ledger, so a reload mid-job finds them), then each `job` event moves a card.
 * The stream is closed as soon as nothing is left to follow, and opened again
 * when a turn ends (a new job may have started). A job that ends has its note
 * on the server already, so `refresh` brings it into the thread; so does a
 * snapshot that no longer lists a job the page thought was running.
 *
 * `cancel` is a card's Cancel: the direct route, no model; the tap shows as
 * `sending` until the app answers, then `asked`. A failure is the card's
 * `errors` entry, never a stopped job. */
export function useJobs(conversationId: number | null, follow: boolean, refresh: () => void) {
  const [tagged, setTagged] = useState<Tagged<JobsState> | null>(null)
  const [asked, setAsked] = useState<Tagged<Record<string, 'sending' | 'asked'>> | null>(null)
  const [failed, setFailed] = useState<Tagged<Record<string, string>> | null>(null)
  const latest = useRef<Tagged<JobsState> | null>(null)
  const refreshRef = useRef(refresh)
  useEffect(() => {
    refreshRef.current = refresh
  })

  useEffect(() => {
    if (conversationId === null || !follow || typeof EventSource === 'undefined') return
    // What the page knew before, so a job that ended unseen is noticed.
    const known = latest.current?.id === conversationId ? latest.current.value : EMPTY_JOBS
    let state: JobsState = { ...known, lastId: 0 }
    const es = new EventSource(turnEventsUrl(conversationId))
    let closed = false
    let errors = 0
    const onEvent = (name: (typeof JOB_EVENTS)[number]) => (event: MessageEvent<string>) => {
      if (closed) return
      errors = 0
      let data: unknown
      try {
        data = JSON.parse(event.data)
      } catch {
        return
      }
      state = applyJobEvent(state, name, data, Number(event.lastEventId) || state.lastId + 1)
      latest.current = { id: conversationId, value: state }
      setTagged(latest.current)
      const ended = name === 'job' && (data as { ended?: boolean }).ended === true
      if (ended || state.dropped.length > 0) refreshRef.current()
      if (!following(state)) es.close()
    }
    for (const name of JOB_EVENTS) es.addEventListener(name, onEvent(name) as EventListener)
    es.onerror = () => {
      errors += 1
      if (!following(state) || es.readyState === EventSource.CLOSED || errors >= JOB_STREAM_RETRIES) es.close()
    }
    return () => {
      closed = true
      es.close()
    }
  }, [conversationId, follow])

  const cancel = useCallback(
    async (app: string, ref: string) => {
      if (conversationId === null) return
      const key = jobKey({ app, ref })
      const put = (value: 'sending' | 'asked' | null) =>
        setAsked((prev) => {
          const rest = { ...(prev?.id === conversationId ? prev.value : {}) }
          if (value === null) delete rest[key]
          else rest[key] = value
          return { id: conversationId, value: rest }
        })
      const note = (text: string | null) =>
        setFailed((prev) => {
          const rest = { ...(prev?.id === conversationId ? prev.value : {}) }
          if (text === null) delete rest[key]
          else rest[key] = text
          return { id: conversationId, value: rest }
        })
      note(null)
      put('sending')
      try {
        await cancelJob(conversationId, app, ref)
        put('asked')
        refreshRef.current()
      } catch (e: unknown) {
        put(null)
        note(sendErrorMessage(e))
      }
    },
    [conversationId],
  )

  return {
    state: tagged?.id === conversationId ? tagged.value : EMPTY_JOBS,
    asks: asked?.id === conversationId ? asked.value : NONE,
    errors: failed?.id === conversationId ? failed.value : NONE,
    cancel,
  }
}
