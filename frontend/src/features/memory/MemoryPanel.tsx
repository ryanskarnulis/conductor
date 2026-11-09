import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { X } from 'lucide-react'
import { ApiError } from '../../api/client'
import { sourceLine } from './factSource'
import { editFact, forgetFact, listFacts, type MemoryFact } from './memoryApi'
import '../../styles/memory.css'

/**
 * Memory you can see (#81): what conductor remembers about the person, each
 * fact with where it came from, to edit or forget. A lazy chunk: the entry
 * never carries it.
 *
 * Nothing here decides anything: the list is the server's, an edit is the
 * person's own words, and each edit or forget made with a conversation open
 * is recorded there as their note turn (`onChanged` refreshes the thread).
 * A forgotten fact is gone from recall and the state block, so the panel says
 * so, and never shows it as kept.
 */
export default function MemoryPanel({
  conversationId,
  onClose,
  onChanged,
}: {
  conversationId: number | null
  onClose: () => void
  onChanged: () => void
}) {
  const [facts, setFacts] = useState<MemoryFact[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [editing, setEditing] = useState<number | null>(null)
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [gone, setGone] = useState<string[]>([])
  const [announce, setAnnounce] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    let live = true
    listFacts().then(
      (found) => {
        if (live) setFacts(found)
      },
      () => {
        if (live) setError('Couldn’t read the memory list.')
      },
    )
    return () => {
      live = false
    }
  }, [])

  useEffect(() => {
    if (editing !== null) inputRef.current?.focus()
  }, [editing])

  const startEdit = (fact: MemoryFact) => {
    setEditing(fact.id)
    setDraft(fact.text)
  }

  const save = async (fact: MemoryFact) => {
    const text = draft.trim()
    if (text === '' || text === fact.text) {
      setEditing(null)
      return
    }
    setBusy(true)
    try {
      const change = await editFact(fact.id, text, conversationId)
      setFacts((now) => (now ?? []).map((f) => (f.id === fact.id && change.fact ? change.fact : f)))
      setEditing(null)
      setAnnounce(`Changed to “${text}”.`)
      setError(null)
      onChanged()
    } catch (e: unknown) {
      setError(e instanceof ApiError && e.status === 422 ? 'That’s too long to remember.' : 'Couldn’t change it.')
    } finally {
      setBusy(false)
    }
  }

  const forget = async (fact: MemoryFact) => {
    setBusy(true)
    try {
      await forgetFact(fact.id, conversationId)
      setFacts((now) => (now ?? []).filter((f) => f.id !== fact.id))
      setGone((was) => [...was, fact.text])
      setAnnounce(`Forgot “${fact.text}”.`)
      setError(null)
      onChanged()
    } catch {
      setError('Couldn’t forget it.')
    } finally {
      setBusy(false)
    }
  }

  const onKey = (event: KeyboardEvent<HTMLInputElement>, fact: MemoryFact) => {
    if (event.key === 'Enter') {
      event.preventDefault()
      void save(fact)
    } else if (event.key === 'Escape') {
      setEditing(null)
    }
  }

  return (
    <section id="shell-memory" className="memory" aria-labelledby="memory-title">
      <header className="memory-head">
        <h2 id="memory-title">Memory</h2>
        <button type="button" className="shell-iconbtn" aria-label="Close memory" onClick={onClose}>
          <X size={17} aria-hidden="true" />
        </button>
      </header>
      <div className="memory-body">
        <p className="memory-intro">
          What conductor remembers about you, and where each one came from. Nothing gets here silently: only what you
          asked it to remember.
        </p>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        {facts === null && !error && <div className="page-loading">Loading memory…</div>}
        {facts !== null && facts.length > 0 && (
          <ul className="memory-list" aria-label="What conductor remembers">
            {facts.map((fact) => (
              <li key={fact.id} className="cd-fact memory-fact">
                <div className="body">
                  <span className="src">{sourceLine(fact)}</span>
                  {editing === fact.id ? (
                    <input
                      ref={inputRef}
                      className="memory-input"
                      value={draft}
                      maxLength={300}
                      aria-label="Change this memory"
                      onChange={(e) => setDraft(e.target.value)}
                      onKeyDown={(e) => onKey(e, fact)}
                      disabled={busy}
                    />
                  ) : (
                    <q>{fact.text}</q>
                  )}
                </div>
                <div className="memory-actions">
                  {editing === fact.id ? (
                    <>
                      <button type="button" className="cd-btn cd-btn--small" disabled={busy} onClick={() => void save(fact)}>
                        Save
                      </button>
                      <button type="button" className="cd-btn cd-btn--small" disabled={busy} onClick={() => setEditing(null)}>
                        Cancel
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        type="button"
                        className="cd-btn cd-btn--small"
                        aria-label={`Edit: ${fact.text}`}
                        disabled={busy}
                        onClick={() => startEdit(fact)}
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                        className="cd-btn cd-btn--small"
                        aria-label={`Forget: ${fact.text}`}
                        disabled={busy}
                        onClick={() => void forget(fact)}
                      >
                        Forget
                      </button>
                    </>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
        {facts !== null && facts.length === 0 && (
          <p className="memory-empty">Nothing remembered. Say “remember …” to add something.</p>
        )}
        {gone.length > 0 && (
          <p className="memory-gone">
            Forgotten just now: {gone.map((text) => `“${text}”`).join(', ')}. Gone from recall and the state block.
          </p>
        )}
        <div className="cd-sr" aria-live="polite">
          {announce}
        </div>
      </div>
    </section>
  )
}
