import type { MemoryFact } from './memoryApi'

const WORDS_MAX = 80

const SURFACES: Record<string, string> = {
  voice: 'by voice',
  room_speaker: 'on the room speaker',
  phone: 'on your phone',
}

function clip(text: string, max: number): string {
  const flat = text.split(/\s+/).join(' ').trim()
  return flat.length <= max ? flat : `${flat.slice(0, max - 1).trimEnd()}…`
}

/** The day a fact was said, short: "Tue, Nov 3". */
export function factDay(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
}

/** Where a fact came from, in words: what the person said, when, where. */
export function sourceLine(fact: MemoryFact): string {
  const source = fact.source
  if (source === null) return `you changed it in the memory list · ${factDay(fact.updated_at)}`
  const parts: string[] = []
  if (source.kind === 'note') parts.push('you changed it in the memory list')
  else if (source.words !== null) parts.push(`you said “${clip(source.words, WORDS_MAX)}”`)
  else parts.push('you said it (those words were forgotten)')
  parts.push(factDay(source.said_at))
  const surface = source.surface !== null ? SURFACES[source.surface] : undefined
  if (surface) parts.push(surface)
  if (source.thread) parts.push(`in “${clip(source.thread.title ?? 'a conversation', 40)}”`)
  return parts.join(' · ')
}
