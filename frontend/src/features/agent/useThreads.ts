import { useCallback, useEffect, useState } from 'react'
import { closeThread, getThread, listThreads, startThread } from '../../api/agent'
import type { CarryData } from '../../design/components'
import type { Surface, ThreadInfo } from '../../types/agent'
import { sendErrorMessage } from './useConversation'

/** Side threads (#139): the open ones, which is on the page, and the
 * page's controls. A thread is a conversation of its own, so the page swaps
 * which conversation it shows; the timeline is `current === null`. Several
 * may be open; one is on the page. A closed thread can be opened again from
 * its result note to read, never to write. Nothing here says a thread is
 * closed before the server's answer does. */
export interface UseThreads {
  open: ThreadInfo[]
  /** The thread on the page: an open one, or a closed one being read. */
  current: ThreadInfo | null
  starting: boolean
  closing: boolean
  error: string | null
  start: (title: string | null, incognito: boolean) => Promise<boolean>
  /** Show a thread (an open one's id), or the timeline (`null`). */
  select: (id: number | null) => void
  /** Open a thread from its result note: read-only once closed. */
  view: (id: number) => Promise<void>
  /** Close the current thread; resolves once its result is in the timeline. */
  close: (surface?: Surface) => Promise<boolean>
}

export function useThreads(ready: boolean): UseThreads {
  const [open, setOpen] = useState<ThreadInfo[]>([])
  const [activeId, setActiveId] = useState<number | null>(null)
  const [viewing, setViewing] = useState<ThreadInfo | null>(null)
  const [starting, setStarting] = useState(false)
  const [closing, setClosing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!ready) return
    let active = true
    listThreads().then(
      (found) => {
        if (active) setOpen(found)
      },
      () => {
        // No list: the timeline still works, and a new thread can start.
      },
    )
    return () => {
      active = false
    }
  }, [ready])

  const start = useCallback(async (title: string | null, incognito: boolean): Promise<boolean> => {
    setStarting(true)
    setError(null)
    try {
      const thread = await startThread(title, incognito)
      setOpen((prev) => [...prev, thread])
      setViewing(null)
      setActiveId(thread.id)
      return true
    } catch (e: unknown) {
      setError(`The side thread didn’t start: ${sendErrorMessage(e)}`)
      return false
    } finally {
      setStarting(false)
    }
  }, [])

  const select = useCallback((id: number | null) => {
    setError(null)
    setViewing(null)
    setActiveId(id)
  }, [])

  const view = useCallback(
    async (id: number): Promise<void> => {
      setError(null)
      const known = open.find((thread) => thread.id === id)
      if (known) {
        setViewing(null)
        setActiveId(id)
        return
      }
      try {
        const thread = await getThread(id)
        setActiveId(null)
        setViewing(thread)
      } catch (e: unknown) {
        setError(`That side thread can’t be opened: ${sendErrorMessage(e)}`)
      }
    },
    [open],
  )

  const close = useCallback(
    async (surface?: Surface): Promise<boolean> => {
      if (activeId === null) return false
      setClosing(true)
      setError(null)
      try {
        await closeThread(activeId, surface)
        setOpen((prev) => prev.filter((thread) => thread.id !== activeId))
        setActiveId(null)
        return true
      } catch (e: unknown) {
        setError(`The side thread is still open: ${sendErrorMessage(e)}`)
        return false
      } finally {
        setClosing(false)
      }
    },
    [activeId],
  )

  const current = viewing ?? open.find((thread) => thread.id === activeId) ?? null
  return { open, current, starting, closing, error, start, select, view, close }
}

/** What a thread carries, as the header shows it: titles and saved facts. */
export function carryChips(thread: ThreadInfo): CarryData[] {
  return thread.carries.map((carry) =>
    carry.type === 'fact'
      ? { label: carry.text ?? '' }
      : { label: carry.title ?? `${carry.kind ?? 'item'}`, outside: carry.untrusted === true },
  )
}
