import type { AgentMessage } from '../../types/agent'

/** Whether a finished turn was about sorting music — the panel's one trigger.
 *
 * Read off the persisted trajectory's `app_tools` (which tools the app itself
 * ran), never off the reply text. A panel that looks for "sort_music" inside a
 * 12B's paraphrase of its own answer is a panel that breaks the first time the
 * model rephrases. */

/** The app whose sorting pass this panel drives, and the tool that is that pass.
 * The coupling is real and lives here rather than being spread through the UI. */
export const SORT_APP = 'music'
export const SORT_TOOL = 'sort_music'

const ASK_TOOL = `ask_${SORT_APP}`

export function isSortTurn(message: AgentMessage | undefined): boolean {
  if (message === undefined || message.role !== 'assistant') return false
  return (message.tool_calls ?? []).some(
    (record) =>
      record.tool === ASK_TOOL &&
      record.error === null &&
      (record.app_tools ?? []).includes(SORT_TOOL),
  )
}

/** The id of the most recent sorting turn in a thread, or null if there is none.
 *
 * The panel keys off *this*, not off the last message, because a sorting pass
 * outlives the turn that started it. Answering "one at a time" out loud is a
 * turn of its own, and it does not necessarily run the sorting tool again —
 * looking only at the newest message made the panel vanish mid-pass and stay
 * gone, which is exactly what a person doing this by voice will do first.
 */
export function latestSortTurnId(messages: readonly AgentMessage[]): number | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (isSortTurn(messages[index])) return messages[index].id
  }
  return null
}

/** The note a filing leaves in the thread, so a click reads as the answer it was.
 *
 * Written as what the person did, because they are who acted — conductor never
 * said this, and putting words in its mouth is the one thing a truthful
 * transcript cannot do. */
export function filingNote(input: {
  artist: string
  genre: string
  tracks: string[]
  filed: number
  notFound: string[]
}): string {
  const what =
    input.tracks.length > 0
      ? `${input.artist}: ${input.tracks.join(', ')}`
      : `${input.artist} (${input.filed} ${input.filed === 1 ? 'song' : 'songs'})`
  const missed =
    input.notFound.length > 0 ? ` · couldn't find ${input.notFound.join(', ')}` : ''
  return `${what} → ${input.genre}${missed}`
}
