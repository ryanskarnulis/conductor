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
 * would go stale the moment anything else touched the folder. `syncKey` is how
 * that gets in: it changes when a turn lands, and the pass re-reads itself.
 *
 * **An opened group stays open until it is empty.** Going song by song means
 * several answers about one artist — two into Dubstep, three into House — so
 * filing part of a group returns to the rest of it rather than to the next
 * artist. Collapsing back to the whole group is `collapse`, and it is a choice
 * somebody makes, not something that happens to them mid-answer.
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
  open: () => void
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

export function useSortPass(
  onFiled: (note: string) => void,
  syncKey: number = 0,
): UseSortPass {
  const [status, setStatus] = useState<SortStatus | null>(null)
  const [opened, setOpened] = useState<OpenedGroup | null>(null)
  // The artist whose group is open, held separately from the group's contents:
  // it is the *question* being asked, and it survives each answer while the
  // contents are re-read after every one of them.
  const [openedArtist, setOpenedArtist] = useState<string | null>(null)
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
    const reading =
      openedArtist === null ? getWorklist() : openGroup(openedArtist)
    reading
      .then((fresh) => {
        if (!live.current) return
        setStatus(fresh)
        setOpened(fresh.opened)
      })
      .catch((e: unknown) => {
        if (!live.current) return
        // The opened artist having nothing left is the ordinary end of going
        // song by song, not a failure: fall back to the worklist, which the
        // effect re-reads on the next pass.
        if (openedArtist !== null && e instanceof ApiError && e.status === 404) {
          setOpenedArtist(null)
          return
        }
        setError(failure(e))
      })
    return () => {
      live.current = false
    }
  }, [attempt, openedArtist, syncKey])

  const current =
    status?.next_up.find((group) => !skipped.includes(group.artist)) ?? null

  const file = useCallback(
    async (genre: string, tracks?: string[]) => {
      if (current === null || busy) return
      setBusy(true)
      setError(null)
      const someOfThem = (tracks ?? []).length > 0
      try {
        const answered = await fileSongs({
          artist: current.artist,
          genre,
          tracks: tracks ?? [],
        })
        setStatus(answered)
        if (someOfThem) {
          // Still going song by song through this artist — re-read the group so
          // what is left is what is on screen. If nothing is left, that read
          // 404s and the pass moves on by itself.
          setAttempt((n) => n + 1)
        } else {
          setOpenedArtist(null)
          setOpened(null)
        }
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

  // Opening is a change of question, not a fetch: the effect owns every read,
  // so the group's contents come from the same place after an answer as before
  // one.
  const open = useCallback(() => {
    if (current === null || busy) return
    setError(null)
    setOpenedArtist(current.artist)
  }, [busy, current])

  const collapse = useCallback(() => {
    setOpenedArtist(null)
    setOpened(null)
  }, [])

  const skip = useCallback(() => {
    if (current === null) return
    setOpenedArtist(null)
    setOpened(null)
    setSkipped((names) => [...names, current.artist])
  }, [current])

  const retry = useCallback(() => {
    setStatus(null)
    setError(null)
    setAttempt((n) => n + 1)
  }, [])

  // The question on screen is the opened artist while there is one, so an
  // answer about part of a group never jumps to somebody else mid-decision.
  const asking =
    openedArtist !== null
      ? (status?.next_up.find((group) => group.artist === openedArtist) ??
        (opened !== null
          ? { artist: opened.artist, tracks: opened.tracks.length, titles: opened.tracks, tags_say: opened.tags_say }
          : null))
      : current

  return {
    status,
    current: asking,
    opened,
    loading,
    busy,
    error,
    file,
    open,
    collapse,
    skip,
    retry,
  }
}
