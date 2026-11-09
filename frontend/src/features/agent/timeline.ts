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
