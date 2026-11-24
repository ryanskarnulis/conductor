import { useEffect, useState } from 'react'
import type { CardOutcome, WaitingCard } from '../../types/agent'
import { appLabel } from './turnView'
import { askedAgo, cardLabel, groupCards } from './waitingCards'

/** How often a card's age is read again. */
const AGE_TICK_MS = 30_000

interface Props {
  cards: WaitingCard[]
  /** What the last tap came to, for a card that didn't go through. */
  results: CardOutcome[]
  busy: boolean
  /** Approve (`yes`) or Not now (`no`) on these cards; `option` on a pick-one. */
  onAnswer: (cards: WaitingCard[], answer: 'yes' | 'no', option?: number) => void
}

/**
 * Waiting on you (#251): every open question as a card, with how long it has
 * waited. A question never blocks the chat, and only a tap here approves it:
 * no words do. One request's cards are a group with one "Approve all".
 */
export function WaitingOnYou({ cards, results, busy, onAnswer }: Props) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (cards.length === 0) return
    const timer = setInterval(() => setNow(Date.now()), AGE_TICK_MS)
    return () => clearInterval(timer)
  }, [cards.length])
  // Open when the turn right before asked something; older cards fold to
  // the head so they never crowd the chat. The person's own toggle wins
  // until a card it hasn't seen appears.
  const [toggled, setToggled] = useState<{ seen: string[]; open: boolean } | null>(null)
  const open = toggled !== null && cards.every((card) => toggled.seen.includes(card.id)) ? toggled.open : cards.some((card) => card.latest)
  const said = results.filter((result) => result.outcome === 'failed' || result.outcome === 'refused')
  if (cards.length === 0 && said.length === 0) return null
  const groups = groupCards(cards)
  const oldest = cards.length > 0 ? askedAgo(cards[0].asked_at, now) : ''
  return (
    <section className="wait" aria-label="Waiting on you" data-open={open ? '' : undefined}>
      <h2 className="wait-head">
        <button type="button" className="wait-toggle" aria-expanded={open} aria-controls="wait-cards" onClick={() => setToggled({ seen: cards.map((card) => card.id), open: !open })} disabled={cards.length === 0}>
          <span className="wait-title">Waiting on you</span>
          {cards.length > 0 && <span className="wait-count">{cards.length}</span>}
          {cards.length > 0 && !open && <span className="wait-oldest">oldest {oldest}</span>}
          {cards.length > 0 && <span className="wait-chevron" aria-hidden="true">{open ? '▾' : '▸'}</span>}
        </button>
      </h2>
      {said.map((result) => (
        <p key={result.id} className="wait-result" role="status">
          {result.detail ?? 'That one wasn’t done.'}
        </p>
      ))}
      <div className="wait-groups" id="wait-cards" hidden={!open}>
        {groups.map((group) => (
          <div key={group.key} className="wait-group" role="group" aria-label={group.cards.length > 1 ? `${group.cards.length} questions from one request` : 'One question'}>
            {group.approvable.length > 1 && (
              <div className="wait-all">
                <span>{group.cards.length} from one request</span>
                <button type="button" className="cd-btn cd-btn--confirm primary" onClick={() => onAnswer(group.approvable, 'yes')} disabled={busy}>
                  Approve all {group.approvable.length}
                </button>
              </div>
            )}
            {group.cards.map((card) => (
              <div key={card.id} className="wait-card" role="group" aria-label={`Waiting: ${card.question}`}>
                <div className="wait-meta">
                  <b>{cardLabel(card, appLabel)}</b>
                  <time dateTime={card.asked_at}>asked {askedAgo(card.asked_at, now)}</time>
                </div>
                <p className="wait-question">{card.question}</p>
                {card.kind === 'pick' ? (
                  <div className="cd-choices">
                    {card.options.map((option, index) => (
                      <button key={`${index}-${option}`} type="button" className="cd-choice" onClick={() => onAnswer([card], 'yes', index)} disabled={busy}>
                        {option}
                      </button>
                    ))}
                    <button type="button" className="cd-btn cd-btn--confirm" onClick={() => onAnswer([card], 'no')} disabled={busy}>
                      Not now
                    </button>
                  </div>
                ) : (
                  <div className="cd-row">
                    <button type="button" className="cd-btn cd-btn--confirm primary" onClick={() => onAnswer([card], 'yes')} disabled={busy}>
                      Approve
                    </button>
                    <button type="button" className="cd-btn cd-btn--confirm" onClick={() => onAnswer([card], 'no')} disabled={busy}>
                      Not now
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        ))}
      </div>
      {cards.length > 0 && open && <p className="wait-how">Only a tap approves. Saying “yes” doesn’t.</p>}
    </section>
  )
}
