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

export interface PlanStepData {
  kind: string
  app: string
  text: string
  state: StepState
  detail?: string
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
            {step.text}
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
}: {
  question: string
  choices: { label: string; detail?: string }[]
  onChoose: (label: string) => void
  spoken?: string
}) {
  return (
    <div className="cd-question" role="group" aria-label="Question">
      <b>Conductor asks</b>
      <p>{question}</p>
      <div className="cd-choices">
        {choices.map((choice) => (
          <button key={choice.label} type="button" className="cd-choice" onClick={() => onChoose(choice.label)}>
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
export function Utterance({ text, followUp, label = 'You', live = false }: { text: string; followUp?: string; label?: string; live?: boolean }) {
  return (
    <div className="cd-utterance">
      <span className="cd-label">{label}</span>
      <p className={`cd-said${live ? ' live' : ''}`}>{text}</p>
      {followUp ? <p className="cd-said-again">{followUp}</p> : null}
    </div>
  )
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
