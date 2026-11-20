import { INCOGNITO, type AgentMessage, type Surface } from '../../types/agent'

/** The one timeline (#82): pure helpers the page and the search panel share.
 * Day separators, each entry's surface tag, incognito's own clock, and the
 * live words of an incognito turn the person is reading right now. */

const SURFACE_LABELS: Record<Surface, string> = {
  desktop: 'web',
  phone: 'phone',
  voice: 'voice',
  room_speaker: 'room speaker',
}

/** Where a turn came from, as its tag says it. A turn the page typed and
 * sent with no surface is "web". */
export function surfaceLabel(surface: Surface | string | null | undefined): string {
  if (!surface) return 'web'
  return SURFACE_LABELS[surface as Surface] ?? surface.replace(/_/g, ' ')
}

/** The calendar day of a timestamp, in this browser's zone: what a separator
 * groups by. */
export function dayKey(iso: string): string {
  const at = new Date(iso)
  return `${at.getFullYear()}-${at.getMonth() + 1}-${at.getDate()}`
}

/** A day separator's words: "Today", "Yesterday", else "Tue, Nov 3". */
export function dayLabel(iso: string, now: Date = new Date()): string {
  const key = dayKey(iso)
  if (key === dayKey(now.toISOString())) return 'Today'
  const yesterday = new Date(now)
  yesterday.setDate(now.getDate() - 1)
  if (key === dayKey(yesterday.toISOString())) return 'Yesterday'
  const at = new Date(iso)
  const sameYear = at.getFullYear() === now.getFullYear()
  return at.toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    ...(sameYear ? {} : { year: 'numeric' }),
  })
}

/** The day each item starts, where it differs from the item before: the
 * separators to draw, by index. Items with no time join the day before. */
export function daySeparators(times: (string | null | undefined)[]): Map<number, string> {
  const found = new Map<number, string>()
  let last: string | null = null
  times.forEach((iso, index) => {
    if (!iso) return
    const key = dayKey(iso)
    if (key !== last) found.set(index, iso)
    last = key
  })
  return found
}

/** What the fold reads of a turn: when it was said, and whether it is a note
 * (a job's or a side thread's), which never breaks a quiet gap. */
export interface FoldTurn {
  when?: string | null
  live: boolean
  jobNote?: boolean
  threadNote?: unknown
}

/** Where the current window starts (#226): the index of the first turn since
 * the timeline's last quiet gap, counted back from `now` the way the planner
 * counts it (`window_since`), so a page opened a gap after the last turn
 * starts empty. Turns before it fold away behind "Show earlier". A note that
 * landed within the gap just before the window stays with it; the turn in
 * flight is always in it. `gapMs` of 0 folds nothing. */
export function windowStart(turns: readonly FoldTurn[], now: number, gapMs: number): number {
  if (gapMs <= 0) return 0
  let start = turns.length
  let later = now
  for (let i = turns.length - 1; i >= 0; i--) {
    const turn = turns[i]
    const at = turn.live || !turn.when ? NaN : Date.parse(turn.when)
    if (Number.isNaN(at)) {
      start = i
      continue
    }
    if (turn.jobNote || turn.threadNote) {
      if (start === i + 1 && now - at < gapMs) start = i
      continue
    }
    if (later - at >= gapMs) break
    start = i
    later = at
  }
  return start
}

/** How many turns or note groups one "Show earlier" brings back (#233):
 * about twenty messages, enough to read back without a press per turn. */
export const REVEAL_BATCH = 10

/** How far above the fold to show so the item at `index` is on the page:
 * never fewer than are shown already. A search hit unfolds this much. */
export function revealedFor(index: number, foldAt: number, revealed: number): number {
  return index < 0 ? revealed : Math.max(revealed, foldAt - index)
}

/** What a scroll area's place is read from. */
export interface ScrollBox {
  scrollHeight: number
  scrollTop: number
  clientHeight: number
}

/** How far the bottom edge of the view is from the end of the content. */
export function fromEnd(box: ScrollBox): number {
  return box.scrollHeight - box.scrollTop - box.clientHeight
}

/** The scrollTop that keeps the place after content arrived above (#233):
 * what was on screen stays on screen. The distance from the end is what is
 * kept, so it also holds when the area itself grew (a phone's bottom sheet
 * grows upward). */
export function keptScrollTop(before: ScrollBox, now: Pick<ScrollBox, 'scrollHeight' | 'clientHeight'>): number {
  return Math.max(0, now.scrollHeight - now.clientHeight - fromEnd(before))
}

/** Incognito turns itself off after the same quiet gap the planner uses (#82):
 * true once `gapMs` has passed since it was last switched on or used. */
export function incognitoLapsed(lastUsedAt: number | null, now: number, gapMs: number): boolean {
  return lastUsedAt !== null && gapMs > 0 && now - lastUsedAt >= gapMs
}

/** The timeline's messages with the words of this session's incognito turns
 * put back: what the person reads now, never what was written. */
export function withLiveWords(messages: AgentMessage[], live: ReadonlyMap<number, string | null>): AgentMessage[] {
  if (live.size === 0) return messages
  return messages.map((m) => (live.has(m.id) ? { ...m, content: live.get(m.id) ?? null } : m))
}

/** True for a turn whose words were never written: shown as "(incognito)". */
export function isIncognitoPlaceholder(message: Pick<AgentMessage, 'incognito' | 'content'>): boolean {
  return message.incognito === true && message.content === INCOGNITO
}
