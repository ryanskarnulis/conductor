import type { StepPart } from '../../design/components'
import type { AppRecordEntry } from '../../types/agent'

/** `{sN}` in a plan step's words: a later step naming an earlier one's result
 * (#72). The frontend twin of backend `dataflow` (`from_record`, `fill`): it
 * only ever reads what the earlier step's receipt says it touched, never an
 * app's reply text, and it never guesses. A value it can't be sure the
 * backend will splice stays a dashed `{sN}`. */

const REF = /\{(s(?:[1-9]|1[0-2]))(?:\.[a-z_]+)?\}/g
const MAX_NAMED = 3
const MAX_TITLE_CHARS = 80
const CONTROL = /[\p{Cc}\u00a0\u2028\u2029]/gu
const STRIP = new Set([...'"\'“”‘’`{}'])

export interface RefValue {
  /** Text ready to splice in: quoted titles, “Honey”. */
  text: string
  /** Outside text (a web title, a file's tags): shown only once the step runs. */
  untrusted: boolean
}

/** A title made safe to splice into a request, as the backend cleans it. */
export function cleanTitle(title: string): string {
  const flat = [...title.replace(CONTROL, ' ')].map((char) => (STRIP.has(char) ? ' ' : char)).join('')
  const words = flat.split(/\s+/).filter(Boolean).join(' ')
  return words.length > MAX_TITLE_CHARS ? `${words.slice(0, MAX_TITLE_CHARS - 1).trimEnd()}…` : words
}

/** What step N gives later steps, from its app's own record: the titles of
 * the done and looked-up entries' touched refs (no jobs, no repeats), up to
 * three. No title, or more than three, is nothing to pass on. */
export function valueFromRecord(entries: readonly AppRecordEntry[]): RefValue | null {
  const titles: string[] = []
  const seen = new Set<string>()
  let untrusted = false
  for (const entry of entries) {
    if (entry.outcome !== 'done' && entry.outcome !== 'looked_up') continue
    for (const ref of entry.touched ?? []) {
      const title = cleanTitle(ref.title ?? '')
      if (!title || ref.kind.endsWith('job')) continue
      untrusted = untrusted || Boolean(ref.untrusted || entry.untrusted)
      const key = title.toLowerCase()
      if (!seen.has(key)) {
        seen.add(key)
        titles.push(title)
      }
    }
  }
  if (titles.length === 0 || titles.length > MAX_NAMED) return null
  const quoted = titles.map((title) => `“${title}”`)
  const text = quoted.length === 1 ? quoted[0] : `${quoted.slice(0, -1).join(', ')} and ${quoted[quoted.length - 1]}`
  return { text, untrusted }
}

/** An `ask_user` answer as the backend splices it: the person's own words
 * (trusted), or the picked option's entity title with its trust (#120). */
export function answerValue(words: string, untrusted = false): RefValue {
  return { text: words.trim().replace(/\{/g, '(').replace(/\}/g, ')'), untrusted }
}

/** A step's words as parts: plain text and refs, each ref filled when its
 * step's value is known and trusted. `text` is the same words, flat. */
export function fillParts(words: string, values: Readonly<Record<string, RefValue>>): { text: string; parts: StepPart[] | undefined } {
  const parts: StepPart[] = []
  let flat = ''
  let at = 0
  let any = false
  for (const match of words.matchAll(REF)) {
    any = true
    if (match.index > at) parts.push({ text: words.slice(at, match.index) })
    const value = values[match[1]]
    const filled = value && !value.untrusted ? value.text : null
    parts.push({ ref: match[1], filled })
    flat += words.slice(at, match.index) + (filled ?? match[0])
    at = match.index + match[0].length
  }
  if (!any) return { text: words, parts: undefined }
  if (at < words.length) parts.push({ text: words.slice(at) })
  flat += words.slice(at)
  return { text: flat, parts }
}
