import type { WaitingCard } from '../../types/agent'

/** One request's cards (#251): asked by the same turn, answered together by
 * "Approve all" or one by one. */
export interface CardGroup {
  key: string
  cards: WaitingCard[]
  /** The cards "Approve all" answers: every one but a pick-one, which needs
   * its own option. The button shows when there are two or more. */
  approvable: WaitingCard[]
}

/** Cards in the order asked, grouped by the turn that asked them. */
export function groupCards(cards: WaitingCard[]): CardGroup[] {
  const groups = new Map<string, WaitingCard[]>()
  for (const card of cards) {
    const key = card.group === null ? card.id : `m${card.group}`
    groups.set(key, [...(groups.get(key) ?? []), card])
  }
  return [...groups.entries()].map(([key, grouped]) => ({ key, cards: grouped, approvable: grouped.filter((card) => card.kind !== 'pick') }))
}

/** How long a card has waited, in words: "just now", "5 min ago", "2 h 37 min ago". */
export function askedAgo(askedAt: string, now: number): string {
  const at = Date.parse(askedAt)
  if (Number.isNaN(at)) return ''
  const minutes = Math.max(0, Math.floor((now - at) / 60_000))
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes} min ago`
  const hours = Math.floor(minutes / 60)
  const rest = minutes % 60
  return rest === 0 ? `${hours} h ago` : `${hours} h ${rest} min ago`
}

/** The tap as the thread records it (backend `cards.tap_note`): shown while
 * the plan it picked up runs. */
export function tapNote(cards: WaitingCard[], option?: string): string {
  const asked = cards
    .slice(0, 3)
    .map((card) => card.question)
    .join(' ')
  if (option !== undefined) return `Tapped “${option}”: ${asked}`
  return `Tapped Approve: ${asked}${cards.length > 3 ? ` (and ${cards.length - 3} more)` : ''}`
}

/** Whether a yes on these cards picks a waiting plan up, so the tap runs a
 * turn: always for code's gate, a proposal and a pick-one. An app's own
 * question may too (steps that waited on it); the server says. */
export function resumesPlan(cards: WaitingCard[]): boolean {
  return cards.some((card) => card.kind !== 'app_confirm')
}

/** What each kind of card is, in a word, beside the app that asked. */
export function cardLabel(card: WaitingCard, appName: (key: string) => string): string {
  if (card.kind === 'proposal') return 'A plan I proposed'
  if (card.kind === 'pick') return card.app ? `${appName(card.app)} asks` : 'Pick one'
  return card.app ? appName(card.app) : 'Conductor'
}
