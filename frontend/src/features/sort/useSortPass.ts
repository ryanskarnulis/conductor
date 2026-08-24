import { useCallback, useEffect, useRef, useState } from 'react'
import { fileSongs, getWorklist, openGroup } from '../../api/sort'
import { ApiError } from '../../api/client'
import type { OpenedGroup, SortStatus, UnsortedGroup } from '../../types/sort'
import { filingNote } from './sortTurn'

/**
 * The sorting pass as the panel drives it: read the worklist, answer, read it back.
 *
 * **The state is fetched, never derived from a reply.** What the panel shows is
 * the library as it is right now, so an answer given by voice, by the terminal
 * pass, or by dragging a file in a file manager moves the buttons too. That is
 * what filing directly buys beyond speed — a snapshot of what a model last said
 * would go stale the moment anything else touched the folder.
 *
 * Skipping is client-side on purpose: not answering is not an act, and it must
 * not write anything. A skipped artist is still waiting, and is asked about
 * again next time the panel opens.
 */
interface UseSortPass {
  status: SortStatus | null
  /** The question on screen: the biggest group not skipped this session. */
  current: UnsortedGroup | null
  /** Set when the current group has been opened up, song by song. */
  opened: OpenedGroup | null
  loading: boolean
  busy: boolean
  error: string | null
  file: (genre: string, tracks?: string[]) => Promise<void>
  open: () => Promise<void>
  collapse: () => void
  skip: () => void
  retry: () => void
}

function failure(e: unknown): string {
  if (e instanceof ApiError) {
    const detail = (e.body as { detail?: unknown } | null)?.detail
    if (typeof detail === 'string') return detail
    if (e.status === 404) return 'Music has nothing waiting under that name any more.'
    if (e.status === 502) return "Music didn't answer — is it running?"
  }
  return e instanceof Error ? e.message : 'That did not go through'
}

export function useSortPass(onFiled: (note: string) => void): UseSortPass {
  const [status, setStatus] = useState<SortStatus | null>(null)
  const [opened, setOpened] = useState<OpenedGroup | null>(null)
  const [skipped, setSkipped] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)
  const live = useRef(true)

  // Loading is derived rather than stored, the way `useConversation` does it:
  // nothing has answered yet and nothing has failed. That keeps the effect free
  // of a synchronous setState, which React now warns about.
  const loading = status === null && error === null

  useEffect(() => {
    live.current = true
    getWorklist()
      .then((fresh) => {
        if (live.current) setStatus(fresh)
      })
      .catch((e: unknown) => {
        if (live.current) setError(failure(e))
      })
    return () => {
      live.current = false
    }
  }, [attempt])

  const current =
    status?.next_up.find((group) => !skipped.includes(group.artist)) ?? null

  const file = useCallback(
    async (genre: string, tracks?: string[]) => {
      if (current === null || busy) return
      setBusy(true)
      setError(null)
      try {
        const answered = await fileSongs({
          artist: current.artist,
          genre,
          tracks: tracks ?? [],
        })
        setStatus(answered)
        setOpened(null)
        onFiled(
          filingNote({
            artist: answered.filed_artist ?? current.artist,
            genre: answered.filed_into ?? genre,
            tracks: tracks ?? [],
            filed: answered.filed_tracks,
            notFound: answered.not_found,
          }),
        )
      } catch (e: unknown) {
        setError(failure(e))
      } finally {
        setBusy(false)
      }
    },
    [busy, current, onFiled],
  )

  const open = useCallback(async () => {
    if (current === null || busy) return
    setBusy(true)
    setError(null)
    try {
      const detail = await openGroup(current.artist)
      setStatus(detail)
      setOpened(detail.opened)
    } catch (e: unknown) {
      setError(failure(e))
    } finally {
      setBusy(false)
    }
  }, [busy, current])

  const collapse = useCallback(() => setOpened(null), [])

  const skip = useCallback(() => {
    if (current === null) return
    setOpened(null)
    setSkipped((names) => [...names, current.artist])
  }, [current])

  const retry = useCallback(() => {
    setStatus(null)
    setError(null)
    setAttempt((n) => n + 1)
  }, [])

  return { status, current, opened, loading, busy, error, file, open, collapse, skip, retry }
}
