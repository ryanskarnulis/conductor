import { useId, type ReactNode } from 'react'
import { AlertTriangle, MessageSquareShare } from 'lucide-react'
import { STEP_WORD, type PresenceState, type StepState } from './status'

/* The design system's components (#86) as React. Presentational only: each
 * takes data and handlers, holds no fetch or state of its own, and writes every
 * status as a word. Styles: src/styles/components.css (`cd-` classes). */

/** What Glitch is doing now, in one word or phrase. */
export function Presence({ state, label, live = false }: { state: PresenceState; label: string; live?: boolean }) {
  return (
    <span className="cd-presence" data-state={state} role={live ? 'status' : undefined}>
      <i aria-hidden="true" />
      {label}
    </span>
  )
}

/** A step's state as a word; `detail` follows it ("failed · chess didn't answer"). */
export function StatusPill({ state, detail }: { state: StepState; detail?: string }) {
  return (
    <span className="cd-pill" data-state={state}>
      {STEP_WORD[state]}
      {detail ? ` · ${detail}` : ''}
    </span>
  )
}

/** A step's words, split where it names an earlier step's result (`{s1}`). */
export type StepPart = { text: string } | { ref: string; filled: string | null }

export interface PlanStepData {
  kind: string
  app: string
  text: string
  state: StepState
  detail?: string
  /** `text` split into plain words and refs; absent when it names none. */
  parts?: StepPart[]
}

/** A ref to an earlier step's result: dashed until that step's receipt fills
 * it with the real title, then solid. Written out in words for a reader. */
function StepRef({ id, filled }: { id: string; filled: string | null }) {
  const n = id.slice(1)
  if (filled !== null) {
    return (
      <span className="cd-ref" data-filled="true" title={`filled in from step ${n}`}>
        {filled}
      </span>
    )
  }
  return (
    <span className="cd-ref" data-filled="false" title={`fills in from step ${n}’s receipt`}>
      <span aria-hidden="true">{`{${id}}`}</span>
      <span className="cd-sr">{`what step ${n} finds`}</span>
    </span>
  )
}

/** The compiled plan: one row per step, its kind, app, text and status. */
export function PlanSteps({ steps }: { steps: PlanStepData[] }) {
  return (
    <ol className="cd-steps" aria-label="Plan steps">
      {steps.map((step, index) => (
        <li key={index} className="cd-step" data-state={step.state}>
          <span className="k">{step.kind}</span>
          <span className="app">{step.app}</span>
          <span className="t" title={step.text}>
            {step.parts
              ? step.parts.map((part, i) =>
                  'ref' in part ? <StepRef key={i} id={part.ref} filled={part.filled} /> : <span key={i}>{part.text}</span>,
                )
              : step.text}
          </span>
          <StatusPill state={step.state} detail={step.detail} />
        </li>
      ))}
    </ol>
  )
}

export interface ReceiptData {
  verb: string
  app: string
  /** Entity text from the app's record. Untrusted: rendered as text only. */
  entity: string
  /** Set when the receipt carries an undo handle: its one-line "how". */
  undoHow?: string
  /** Set when the receipt says there is no undo, and why. */
  noUndo?: string
}

/** Proof that a step happened: verb · app · entity, from the app's receipt. */
export function ReceiptChip({ receipt, onUndo }: { receipt: ReceiptData; onUndo?: () => void }) {
  return (
    <li className="cd-receipt">
      <i aria-hidden="true" />
      <span className="ent">
        <b>{receipt.verb}</b> · {receipt.app}
        {receipt.entity ? ` · ${receipt.entity}` : ''}
        {receipt.noUndo ? <span className="cd-none"> · can’t undo: {receipt.noUndo}</span> : null}
      </span>
      {receipt.undoHow && onUndo ? (
        <button type="button" className="cd-btn cd-btn--small" title={receipt.undoHow} onClick={onUndo}>
          Undo
        </button>
      ) : null}
    </li>
  )
}

export function Receipts({ receipts, onUndo }: { receipts: ReceiptData[]; onUndo?: (receipt: ReceiptData) => void }) {
  return (
    <ul className="cd-receipts" aria-label="Receipts">
      {receipts.map((receipt, index) => (
        <ReceiptChip key={index} receipt={receipt} onUndo={onUndo ? () => onUndo(receipt) : undefined} />
      ))}
    </ul>
  )
}

/** A write waiting on you: the owning app's question, passed through verbatim. */
export function ConfirmationCard({
  app,
  question,
  why,
  actionLabel,
  onConfirm,
  onDecline,
  disabled = false,
}: {
  app: string
  question: string
  why?: string
  actionLabel: string
  onConfirm: () => void
  onDecline: () => void
  disabled?: boolean
}) {
  return (
    <div className="cd-confirm" role="group" aria-label="Confirmation">
      <b>Waiting on you · {app}</b>
      <p>{question}</p>
      {why ? <p className="why">{why}</p> : null}
      <div className="cd-row">
        <button type="button" className="cd-btn cd-btn--confirm primary" onClick={onConfirm} disabled={disabled}>
          {actionLabel}
        </button>
        <button type="button" className="cd-btn cd-btn--confirm" onClick={onDecline} disabled={disabled}>
          Not now
        </button>
      </div>
    </div>
  )
}

/** Conductor needs a choice to continue. Never a yes/no about a write. */
export function QuestionCard({
  question,
  choices,
  onChoose,
  spoken = 'Or say it.',
  disabled = false,
}: {
  question: string
  choices: { label: string; detail?: string }[]
  onChoose: (label: string) => void
  spoken?: string
  disabled?: boolean
}) {
  return (
    <div className="cd-question" role="group" aria-label="Question">
      <b>Conductor asks</b>
      <p>{question}</p>
      <div className="cd-choices">
        {choices.map((choice) => (
          <button key={choice.label} type="button" className="cd-choice" onClick={() => onChoose(choice.label)} disabled={disabled}>
            {choice.label}
            {choice.detail ? ` · ${choice.detail}` : ''}
          </button>
        ))}
      </div>
      <p className="why">{spoken}</p>
    </div>
  )
}

/** Something Conductor could do that you did not ask for. */
export function ProposalCard({ text, onAccept, onDismiss }: { text: string; onAccept: () => void; onDismiss: () => void }) {
  return (
    <div className="cd-proposal" role="group" aria-label="Proposal">
      <b>Conductor could</b>
      <p>{text}</p>
      <div className="cd-row">
        <button type="button" className="cd-btn cd-btn--propose primary" onClick={onAccept}>
          Do it
        </button>
        <button type="button" className="cd-btn cd-btn--propose" onClick={onDismiss}>
          Not now
        </button>
      </div>
    </div>
  )
}

/** A step that moved to the background. Progress is written as numbers too. */
export function JobCard({ app, title, done, total, stopHint }: { app: string; title: string; done: number; total: number; stopHint?: string }) {
  const fmt = (n: number) => n.toLocaleString('en-US')
  return (
    <div className="cd-job">
      <div className="r1">
        <span>
          <b>Job · {app}</b> · {title}
        </span>
        <span className="big">
          running · {fmt(done)} of {fmt(total)}
        </span>
      </div>
      <progress max={total} value={done} aria-label={`${title} progress`} />
      <span className="hint">{stopHint ?? 'A note arrives when it ends. Stop it any time: “stop”.'}</span>
    </div>
  )
}

/** A background job that ended, from its receipt. */
export function FinishedNote({
  app,
  at,
  text,
  onUndo,
  onOpen,
}: {
  app: string
  at: string
  text: string
  onUndo?: () => void
  onOpen?: () => void
}) {
  return (
    <div className="cd-note">
      <b>
        Finished · {app} · {at}
      </b>
      <span>{text}</span>
      {onUndo || onOpen ? (
        <div className="cd-row">
          {onUndo ? (
            <button type="button" className="cd-btn cd-btn--small" onClick={onUndo}>
              Undo
            </button>
          ) : null}
          {onOpen ? (
            <button type="button" className="cd-btn cd-btn--small" onClick={onOpen}>
              Open in {app}
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

/** What you said, quoted exactly. `live` while the words are still arriving. */
export function Utterance({ text, followUp, label = 'You', live = false }: { text: string; followUp?: string; label?: ReactNode; live?: boolean }) {
  return (
    <div className="cd-utterance">
      <span className="cd-label">{label}</span>
      <p className={`cd-said${live ? ' live' : ''}`}>{text}</p>
      {followUp ? <p className="cd-said-again">{followUp}</p> : null}
    </div>
  )
}

/** This turn picks up a plan that was waiting on you; says how. */
export function ResumedNote({ text }: { text: string }) {
  return <p className="cd-resumed">{text}</p>
}

/** Conductor's one-line acknowledgement of a plan. */
export function Ack({ text }: { text: string }) {
  return <p className="cd-ack">{text}</p>
}

/** Conductor's reply, built from receipts. `children` lets a caller render markdown. */
export function Narration({ children, label = 'Conductor' }: { children: ReactNode; label?: string }) {
  return (
    <div className="cd-narration">
      <span className="cd-label">{label}</span>
      <div className="cd-narration-text">{children}</div>
    </div>
  )
}

/** The ledger lines the screen is drawn from; the newest is current. */
export function Ledger({ lines, defaultOpen = false }: { lines: { time: string; event: string }[]; defaultOpen?: boolean }) {
  return (
    <details className="cd-ledger" open={defaultOpen}>
      <summary>Ledger · what this screen is drawn from</summary>
      <ol>
        {lines.map((line, index) => (
          <li key={index} aria-current={index === lines.length - 1 ? 'true' : undefined}>
            <time>{line.time}</time>
            {line.event}
          </li>
        ))}
      </ol>
    </details>
  )
}

/** What a delegate app did, folded into one line you can open. */
export function SideThread({
  app,
  summary,
  calls,
  failed = false,
  defaultOpen = false,
}: {
  app: string
  summary: string
  calls: { name: string; outcome: string }[]
  failed?: boolean
  defaultOpen?: boolean
}) {
  const id = useId()
  const Icon = failed ? AlertTriangle : MessageSquareShare
  return (
    <details className="cd-thread" data-state={failed ? 'failed' : undefined} open={defaultOpen}>
      <summary aria-describedby={id}>
        <Icon size={15} aria-hidden="true" />
        <span className="app">{app}</span>
        <span id={id}>{summary}</span>
      </summary>
      {calls.length > 0 ? (
        <ol>
          {calls.map((call, index) => (
            <li key={index}>
              <b>{call.name}</b> · {call.outcome}
            </li>
          ))}
        </ol>
      ) : null}
    </details>
  )
}

/** One thing Conductor recalled from your own record, quoted, with its source and date. */
export function MemoryFact({ source, date, fact, onForget }: { source: string; date: string; fact: string; onForget?: () => void }) {
  return (
    <div className="cd-fact">
      <div className="body">
        <span className="src">
          {source} · {date}
        </span>
        <q>{fact}</q>
      </div>
      {onForget ? (
        <button type="button" className="cd-btn cd-btn--small" onClick={onForget}>
          Forget
        </button>
      ) : null}
    </div>
  )
}

/** One thing a side thread carries in: an entity by its title (outside text
 * quoted), or a saved fact's words. */
export interface CarryData {
  label: string
  outside?: boolean
}

/** A side thread's header (#139): its name, what it carries, and how it ends.
 * Open, "Done" closes it and its result flows back to the timeline (nothing
 * says closed before the server does); closed, it shows what flowed back. */
export function ThreadHead({
  title,
  carries,
  result = null,
  closing = false,
  onDone,
  onBack,
}: {
  title: string
  carries: CarryData[]
  /** Set once the thread closed: the note it posted to the timeline. */
  result?: string | null
  closing?: boolean
  onDone?: () => void
  onBack: () => void
}) {
  const closed = result !== null
  return (
    <section className="cd-thhead" aria-label={`Side thread: ${title}`}>
      <div className="cd-row cd-thhead-r">
        <b>Side thread · {title}</b>
        {closed || !onDone ? (
          <button type="button" className="cd-btn cd-btn--small" onClick={onBack}>
            Back to the timeline
          </button>
        ) : (
          <button type="button" className="cd-btn cd-btn--small" onClick={onDone} disabled={closing}>
            {closing ? 'Closing…' : 'Done · back to the timeline'}
          </button>
        )}
      </div>
      <div className="cd-carry">
        <span>Carries</span>
        {carries.length === 0 ? (
          <span className="cd-carry-none">nothing: it started empty</span>
        ) : (
          carries.map((carry, index) => (
            <span key={index} className="c" data-outside={carry.outside ? 'true' : undefined}>
              {carry.outside ? `“${carry.label}”` : carry.label}
            </span>
          ))
        )}
      </div>
      <p className="cd-why">
        {closed
          ? `Closed. What flowed back: ${result}`
          : closing
            ? 'Closing: its result is being written from what it did.'
            : 'Its own small context: only what it carries, not the whole timeline. Close it and what it did flows back.'}
      </p>
    </section>
  )
}

/** A side thread's result in the timeline (#139): what flowed back when it closed. */
export function ThreadNote({ title, text, onOpen }: { title: string; text: string; onOpen?: () => void }) {
  return (
    <div className="cd-flow" role="group" aria-label={`Side thread result: ${title}`}>
      <span className="cd-flow-tag">side thread</span>
      <span className="cd-flow-text">
        <b>{title}</b> · {text}
      </span>
      {onOpen ? (
        <button type="button" className="cd-btn cd-btn--small" onClick={onOpen}>
          Open
        </button>
      ) : null}
    </div>
  )
}

/** The timeline and each open side thread; one is on the page at a time. */
export function ThreadTabs({
  threads,
  active,
  onSelect,
  disabled = false,
}: {
  threads: { id: number; title: string }[]
  active: number | null
  onSelect: (id: number | null) => void
  disabled?: boolean
}) {
  return (
    <nav className="cd-threadtabs" aria-label="Timeline and side threads">
      <button type="button" aria-current={active === null ? 'page' : undefined} onClick={() => onSelect(null)} disabled={disabled}>
        Timeline
      </button>
      {threads.map((thread) => (
        <button
          key={thread.id}
          type="button"
          aria-current={active === thread.id ? 'page' : undefined}
          onClick={() => onSelect(thread.id)}
          disabled={disabled}
        >
          {thread.title}
        </button>
      ))}
    </nav>
  )
}

/** Starting a side thread: an optional name (none while incognito: it would be kept). */
export function ThreadStart({
  name,
  incognito,
  busy,
  error,
  onName,
  onStart,
  onCancel,
}: {
  name: string
  incognito: boolean
  busy: boolean
  error?: string | null
  onName: (name: string) => void
  onStart: () => void
  onCancel: () => void
}) {
  const id = useId()
  return (
    <form
      className="cd-thstart"
      aria-label="New side thread"
      onSubmit={(event) => {
        event.preventDefault()
        onStart()
      }}
    >
      <p className="cd-why">It carries what the timeline touched and saved since the last quiet gap: names and facts, not what was said.</p>
      {incognito ? (
        <p className="cd-why">Incognito is on, so its name isn’t kept.</p>
      ) : (
        <>
          <label htmlFor={id} className="cd-label">
            Name (optional)
          </label>
          <input id={id} value={name} maxLength={60} onChange={(event) => onName(event.target.value)} placeholder="What’s it about?" />
        </>
      )}
      {error ? (
        <p role="alert" className="cd-why cd-why--bad">
          {error}
        </p>
      ) : null}
      <div className="cd-row">
        <button type="submit" className="cd-btn cd-btn--propose primary" disabled={busy}>
          {busy ? 'Starting…' : 'Start'}
        </button>
        <button type="button" className="cd-btn" onClick={onCancel} disabled={busy}>
          Cancel
        </button>
      </div>
    </form>
  )
}
