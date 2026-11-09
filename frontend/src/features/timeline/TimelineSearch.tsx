import { useEffect, useRef, useState } from 'react'
import { Search, X } from 'lucide-react'
import type { TimelineHit } from '../../types/agent'
import { dayLabel, surfaceLabel } from '../agent/timeline'
import { searchTimeline } from './timelineApi'
import '../../styles/timeline.css'

// Typing settles this long before a search goes out.
const DEBOUNCE_MS = 250
const MIN_CHARS = 2

const MATCHED: Record<TimelineHit['matched'][number], string> = {
  words: 'what you said',
  reply: 'the reply',
  receipt: 'what an app did',
}

/**
 * Search the timeline (#82): what was said, conductor's replies and the titles
 * the apps' receipts touched. A lazy chunk: the entry never carries it.
 * Incognito and forgotten turns are never hits (the server leaves them out).
 * Picking a hit closes the panel and brings that turn into view when it is on
 * the page.
 */
export default function TimelineSearch({ onClose, onOpen }: { onClose: () => void; onOpen: (messageId: number | null) => void }) {
  const [query, setQuery] = useState('')
  const [hits, setHits] = useState<TimelineHit[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const words = query.trim()
  const searching = words.length >= MIN_CHARS

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  useEffect(() => {
    if (!searching) return
    const abort = new AbortController()
    const timer = setTimeout(() => {
      searchTimeline(words, abort.signal).then(
        (found) => {
          setHits(found)
          setError(null)
        },
        (e: unknown) => {
          if (!abort.signal.aborted) setError(e instanceof Error && e.name === 'AbortError' ? null : 'Couldn’t search the timeline.')
        },
      )
    }, DEBOUNCE_MS)
    return () => {
      clearTimeout(timer)
      abort.abort()
    }
  }, [words, searching])

  const shown = searching ? hits : null

  return (
    <section id="shell-search" className="tl-search" aria-labelledby="tl-search-title">
      <header className="tl-search-head">
        <h2 id="tl-search-title">Search the timeline</h2>
        <button type="button" className="shell-iconbtn" aria-label="Close search" onClick={onClose}>
          <X size={17} aria-hidden="true" />
        </button>
      </header>
      <div className="tl-search-body">
        <label className="tl-search-field">
          <Search size={15} aria-hidden="true" />
          <input
            ref={inputRef}
            type="search"
            value={query}
            placeholder="Words, replies, what the apps did…"
            aria-label="Search the timeline"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') onClose()
            }}
          />
        </label>
        <p className="tl-search-note">Incognito turns aren’t searched: their words were never kept.</p>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        {shown !== null && shown.length === 0 && <p className="tl-search-empty">Nothing on the timeline matches “{words}”.</p>}
        {shown !== null && shown.length > 0 && (
          <ul className="tl-hits" aria-label="Matches">
            {shown.map((hit) => (
              <li key={hit.turn_id}>
                <button type="button" className="tl-hit" onClick={() => onOpen(hit.message_id)}>
                  <span className="tl-hit-meta">
                    {dayLabel(hit.at)} · {new Date(hit.at).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}
                    <span className="tl-tag">{hit.kind === 'job' ? 'job' : surfaceLabel(hit.surface)}</span>
                  </span>
                  {hit.said && <q className="tl-hit-said">{hit.said}</q>}
                  {hit.reply && <span className="tl-hit-reply">{hit.reply}</span>}
                  {hit.titles.length > 0 && (
                    <span className="tl-hit-titles">{hit.titles.map((title) => `“${title}”`).join(', ')}</span>
                  )}
                  <span className="cd-sr">Matched {hit.matched.map((m) => MATCHED[m]).join(', ')}.</span>
                </button>
              </li>
            ))}
          </ul>
        )}
        <div className="cd-sr" aria-live="polite">
          {shown !== null ? `${shown.length} ${shown.length === 1 ? 'match' : 'matches'}` : ''}
        </div>
      </div>
    </section>
  )
}
