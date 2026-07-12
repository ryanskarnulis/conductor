import type { TurnActivity } from '../../types/agent'

/**
 * The spoken acknowledgment for a delegate beat on a voice-initiated turn,
 * or null when this beat warrants none. Fleet UX rule
 * (`agent-standard/voice.md`): a delegated conductor turn blocks 12–22 s, so
 * the first activity beat naming each `ask_<app>` tool is spoken ("Asking
 * chess…") rather than leaving dead air.
 *
 * `acked` is the per-turn memory of apps already announced — the caller
 * resets it when a new turn starts; this function records into it, so a
 * given app is acknowledged at most once per turn no matter how many beats
 * (or repeat calls to the same app) the poll surfaces.
 */
export function delegateAckText(activity: TurnActivity | null, acked: Set<string>): string | null {
  if (activity === null || !activity.active || activity.kind !== 'tool') return null
  const tool = activity.tool
  if (tool === null || !tool.startsWith('ask_')) return null
  const app = tool.slice('ask_'.length)
  if (acked.has(app)) return null
  acked.add(app)
  return `Asking ${app}…`
}
