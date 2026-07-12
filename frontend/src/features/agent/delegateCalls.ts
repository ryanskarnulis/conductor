import type { ToolCallRecord, TurnActivity } from '../../types/agent'

/** Pure helpers turning conductor's tool names (`ask_<app>`, `list_agents`)
 * into human labels — shared by the trajectory list and the live progress
 * line so both always agree on what "ask_chess" is called. */

/** The app name behind an `ask_<app>` tool, or null for non-delegate tools.
 * Tool names flatten slug hyphens to underscores (backend `ask_tool_name`),
 * so the display name reverses to spaces: `ask_home_media` → "home media". */
export function appFromTool(tool: string): string | null {
  if (!tool.startsWith('ask_') || tool.length <= 4) return null
  return tool.slice(4).replace(/_/g, ' ')
}

/** One-line summary of a persisted trajectory entry: "Asked chess",
 * "Checked the fleet", or a generic fallback for tools we don't know. */
export function describeToolCall(record: ToolCallRecord): string {
  const app = appFromTool(record.tool)
  if (app !== null) return `Asked ${app}`
  if (record.tool === 'list_agents') return 'Checked the fleet'
  return `Ran ${record.tool}`
}

/** The live progress line while a turn is in flight. `null` activity (poll
 * hasn't landed yet) reads as thinking — the run always starts on the model. */
export function activityLabel(activity: TurnActivity | null): string {
  if (activity?.kind === 'tool' && activity.tool !== null) {
    const app = appFromTool(activity.tool)
    if (app !== null) return `Asking ${app}…`
    if (activity.tool === 'list_agents') return 'Checking the fleet…'
    return `Running ${activity.tool}…`
  }
  return 'Thinking…'
}

/** Coarse elapsed suffix for the progress line; blank for the first seconds
 * so quick turns never flash a counter. */
export function elapsedSuffix(activity: TurnActivity | null): string {
  const seconds = activity?.elapsed_seconds
  if (seconds == null || seconds < 5) return ''
  return ` · ${Math.round(seconds)}s`
}
