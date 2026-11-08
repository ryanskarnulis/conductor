import type { ReactNode } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import type { JobCardData } from './jobStream'
import type { ShellTurn } from './turnView'

/** Live job cards and finished notes (#90). Presentational: the page hands
 * over the cards the stream's events made (`jobCards`) and what to do on
 * Cancel. Every word is a job's own status, and progress is always numbers
 * as well as a bar. An outside app's title is a quoted label, never markup. */

const SHOWN = 3

function Title({ card }: { card: JobCardData }) {
  return card.outside ? (
    <q className="cd-job-title" title="This name comes from outside the house">
      {card.title}
    </q>
  ) : (
    <span className="cd-job-title">{card.title}</span>
  )
}

function Bar({ card }: { card: JobCardData }) {
  if (card.ratio === null) return null
  return <progress max={card.total ?? 1} value={card.done} aria-label={`${card.title} progress`} />
}

function Stop({ card, error, onCancel }: { card: JobCardData; error?: string; onCancel: (card: JobCardData) => void }) {
  if (!card.cancelable) return null
  const busy = card.asking !== 'idle'
  return (
    <>
      <button
        type="button"
        className="cd-btn cd-btn--small"
        disabled={busy}
        aria-label={`Stop ${card.title}`}
        onClick={() => onCancel(card)}
      >
        {card.asking === 'sending' ? 'Asking…' : card.asking === 'asked' ? 'Asked to stop' : 'Stop'}
      </button>
      {error && (
        <span className="cd-job-error" role="alert">
          Couldn’t stop it: {error}
        </span>
      )}
    </>
  )
}

/** One job in full: its title, status in words, `n of m`, the bar and Stop. */
function Card({ card, error, onCancel }: { card: JobCardData; error?: string; onCancel: (card: JobCardData) => void }) {
  return (
    <div className="cd-job" role="group" aria-label={`Job: ${card.title}`} data-state={card.state}>
      <div className="r1">
        <span>
          <b>Job · {card.appName}</b> · <Title card={card} />
        </span>
        <span className="big">
          {card.word} · {card.progress}
        </span>
      </div>
      <Bar card={card} />
      <span className="hint">
        {card.ended ? 'Its note is on the way.' : 'A note arrives when it ends. Say “stop”, or tap Stop.'}
      </span>
      <div className="cd-row">
        <Stop card={card} error={error} onCancel={onCancel} />
      </div>
    </div>
  )
}

/** One job as a row of a stack: the same words, no chrome. */
function Row({ card, error, onCancel }: { card: JobCardData; error?: string; onCancel: (card: JobCardData) => void }) {
  return (
    <li className="cd-jobrow" data-state={card.state}>
      <span className="who">
        <b>{card.appName}</b> <Title card={card} />
      </span>
      <span className="big">
        {card.word} · {card.progress}
      </span>
      <Bar card={card} />
      <span className="cd-row">
        <Stop card={card} error={error} onCancel={onCancel} />
      </span>
    </li>
  )
}

/** The conversation's running jobs: one job is a card; several are a calm
 * stack (three rows, the rest folded). `announce` is read once, politely,
 * when a job appears, changes status or ends, never on every tick. */
export function JobCards({
  cards,
  announce,
  errors,
  onCancel,
}: {
  cards: JobCardData[]
  announce: string | null
  errors: Readonly<Record<string, string>>
  onCancel: (card: JobCardData) => void
}) {
  const rows = (list: JobCardData[]) =>
    list.map((card) => <Row key={card.key} card={card} error={errors[card.key]} onCancel={onCancel} />)
  return (
    <>
      <div className="cd-sr" role="status" aria-live="polite">
        {announce}
      </div>
      {cards.length === 1 && <Card card={cards[0]} error={errors[cards[0].key]} onCancel={onCancel} />}
      {cards.length > 1 && (
        <section className="cd-jobs" aria-label="Background jobs">
          <b className="cd-jobs-head">{cards.length} jobs in the background</b>
          <ul>{rows(cards.slice(0, SHOWN))}</ul>
          {cards.length > SHOWN && (
            <details>
              <summary>{cards.length - SHOWN} more</summary>
              <ul>{rows(cards.slice(SHOWN))}</ul>
            </details>
          )}
        </section>
      )}
    </>
  )
}

/** A background job's one note (#90), where it landed in the thread. Neutral
 * on purpose: the note says in words whether the job finished, failed or was
 * stopped, and what it missed. */
function Note({ turn, children }: { turn: ShellTurn; children: ReactNode }) {
  const when = turn.at ? new Date(turn.at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : null
  return (
    <div className="cd-jobnote">
      <b>Background job{when ? ` · ${when}` : ''}</b>
      {children}
    </div>
  )
}

// Links in a note open in a new tab, as in a reply (an outside title can't navigate the SPA away).
const LINKS: Components = {
  a: ({ node, ...props }) => {
    void node
    return <a {...props} target="_blank" rel="noopener noreferrer" />
  },
}
const markdown = (text: string) => <ReactMarkdown components={LINKS}>{text}</ReactMarkdown>
const firstLine = (text: string) => text.split('\n')[0].replace(/[*_`#>[\]]/g, '')

/** Job notes that arrived together: one reads in full; several stack, each a
 * line that opens to its note, so ten jobs ending at once are one quiet block. */
export function JobNotes({ turns }: { turns: ShellTurn[] }) {
  if (turns.length === 1) {
    const turn = turns[0]
    return (
      <aside className="shell-jobnotes" aria-label="Job note">
        <Note turn={turn}>{markdown(turn.reply ?? '')}</Note>
      </aside>
    )
  }
  return (
    <aside className="shell-jobnotes" aria-label="Job notes">
      <b className="cd-jobs-head">{turns.length} jobs finished</b>
      {turns.map((turn) => (
        <details key={turn.key} className="cd-jobnote">
          <summary>{firstLine(turn.reply ?? '')}</summary>
          {markdown(turn.reply ?? '')}
        </details>
      ))}
    </aside>
  )
}
