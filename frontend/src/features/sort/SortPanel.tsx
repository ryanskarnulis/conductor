import { useState, type FormEvent } from 'react'
import { FolderPlus, ListMusic, SkipForward, X } from 'lucide-react'
import type { UnsortedGroup } from '../../types/sort'
import { useSortPass } from './useSortPass'

/**
 * Answering music's sorting pass by clicking, inside the chat.
 *
 * The pass is ~150 questions and every one of them has the same shape: which
 * folder do these songs go in? Typing or speaking each answer works and keeps
 * working — this is the same question with the answers already on screen.
 *
 * **The panel never decides a genre.** It offers the folders that exist, the
 * same ones the terminal pass offers, and files what is clicked. The leftover
 * `tags say:` hint is shown for the same reason it is shown there — it is the
 * only signal the file carries, and half of them are junk.
 */
export function SortPanel({
  onFiled,
  onDismiss,
  syncKey = 0,
}: {
  onFiled: (note: string) => void
  onDismiss: () => void
  /** Changes when a turn lands, so the pass re-reads what is waiting: an answer
   * given by voice or by typing files songs too. */
  syncKey?: number
}) {
  const pass = useSortPass(onFiled, syncKey)
  const [selected, setSelected] = useState<string[]>([])
  const [naming, setNaming] = useState(false)
  const [newFolder, setNewFolder] = useState('')

  const { status, current, opened, busy } = pass
  // Opened up, the answer is about the ticked songs; collapsed, it is about the
  // whole group. An artist is an ordering of the questions, never a rule.
  const tracks = opened !== null ? selected : undefined
  const canAnswer = !busy && current !== null && (opened === null || selected.length > 0)

  const answer = (genre: string) => {
    void pass.file(genre, tracks)
    setSelected([])
    setNaming(false)
    setNewFolder('')
  }

  const submitFolder = (event: FormEvent) => {
    event.preventDefault()
    const name = newFolder.trim()
    if (name !== '') answer(name)
  }

  const toggle = (title: string) =>
    setSelected((chosen) =>
      chosen.includes(title) ? chosen.filter((name) => name !== title) : [...chosen, title],
    )

  return (
    <section className="sort-panel" aria-label="Sort music">
      <header className="sort-panel-head">
        <ListMusic size={15} aria-hidden="true" />
        <span className="sort-panel-title">
          {status === null
            ? 'Sorting'
            : `${status.unsorted_tracks} song${status.unsorted_tracks === 1 ? '' : 's'} to sort · ${status.unsorted_artists} artist${status.unsorted_artists === 1 ? '' : 's'}`}
        </span>
        <button
          type="button"
          className="sort-panel-close"
          onClick={onDismiss}
          aria-label="Hide the sorting panel"
        >
          <X size={15} aria-hidden="true" />
        </button>
      </header>

      {pass.loading && <p className="sort-panel-note">Reading the library…</p>}

      {pass.error !== null && (
        <p className="sort-panel-error" role="alert">
          {pass.error}{' '}
          <button type="button" className="sort-linkish" onClick={pass.retry}>
            try again
          </button>
        </p>
      )}

      {!pass.loading && current === null && pass.error === null && (
        <p className="sort-panel-note">
          Nothing left waiting — everything has a folder.
        </p>
      )}

      {current !== null && (
        <>
          <Question group={current} opened={opened !== null} />

          {opened !== null && (
            <ul className="sort-tracks">
              {opened.tracks.map((title) => (
                <li key={title}>
                  <label className="sort-track">
                    <input
                      type="checkbox"
                      checked={selected.includes(title)}
                      onChange={() => toggle(title)}
                    />
                    <span>{title}</span>
                  </label>
                </li>
              ))}
            </ul>
          )}

          <div className="sort-answers">
            {(status?.folders ?? []).map((folder) => (
              <button
                key={folder}
                type="button"
                className="sort-folder"
                disabled={!canAnswer}
                onClick={() => answer(folder)}
              >
                {folder}
              </button>
            ))}

            {naming ? (
              <form className="sort-new-folder" onSubmit={submitFolder}>
                <input
                  autoFocus
                  value={newFolder}
                  onChange={(event) => setNewFolder(event.target.value)}
                  placeholder="New folder name"
                  aria-label="New folder name"
                  maxLength={60}
                />
                <button type="submit" className="sort-folder" disabled={!canAnswer}>
                  Create &amp; file
                </button>
              </form>
            ) : (
              <button
                type="button"
                className="sort-action"
                onClick={() => setNaming(true)}
                disabled={busy}
              >
                <FolderPlus size={14} aria-hidden="true" />
                New folder
              </button>
            )}

            {opened === null ? (
              current.tracks > 1 && (
                <button
                  type="button"
                  className="sort-action"
                  onClick={() => void pass.open()}
                  disabled={busy}
                >
                  <ListMusic size={14} aria-hidden="true" />
                  One at a time
                </button>
              )
            ) : (
              <button
                type="button"
                className="sort-action"
                onClick={() => {
                  pass.collapse()
                  setSelected([])
                }}
                disabled={busy}
              >
                All of them
              </button>
            )}

            <button
              type="button"
              className="sort-action"
              onClick={() => {
                pass.skip()
                setSelected([])
              }}
              disabled={busy}
            >
              <SkipForward size={14} aria-hidden="true" />
              Skip
            </button>
          </div>

          {opened !== null && (
            <p className="sort-panel-note">
              {selected.length === 0
                ? 'Tick the songs that go together, then pick a folder.'
                : `${selected.length} selected`}
            </p>
          )}
        </>
      )}
    </section>
  )
}

function Question({ group, opened }: { group: UnsortedGroup; opened: boolean }) {
  return (
    <div className="sort-question">
      <p className="sort-artist">
        {group.artist}
        <span className="sort-count">
          {group.tracks} song{group.tracks === 1 ? '' : 's'}
        </span>
      </p>
      {!opened && group.titles.length > 0 && (
        <p className="sort-titles">
          {group.titles.join(' · ')}
          {group.tracks > group.titles.length && ` (+${group.tracks - group.titles.length} more)`}
        </p>
      )}
      {group.tags_say.length > 0 && (
        // Shown, never believed — half the library's genre tags say "Music" or
        // the name of whoever ripped it.
        <p className="sort-hint">tags say: {group.tags_say.join(' · ')}</p>
      )}
    </div>
  )
}
