import type { ToolCallRecord, TurnActivity } from '../../types/agent'

/** Pure helpers turning conductor's tool names (`ask_<app>`, `open_<app>`,
 * `list_agents`) into human labels — shared by the trajectory list and the
 * live progress line so both always agree on what "ask_chess" is called. */

/** The app name behind a prefixed tool (`ask_<app>`, `open_<app>`), or null.
 * Tool names flatten slug hyphens to underscores (backend `ask_tool_name`),
 * so the display name reverses to spaces: `ask_home_media` → "home media". */
function appFrom(tool: string, prefix: string): string | null {
  if (!tool.startsWith(prefix) || tool.length <= prefix.length) return null
  return tool.slice(prefix.length).replace(/_/g, ' ')
}

/** The app behind an `ask_<app>` tool, or null for anything else. */
export function appFromTool(tool: string): string | null {
  return appFrom(tool, 'ask_')
}

/** The app behind an `open_<app>` handoff tool, or null for anything else. */
export function openedAppFromTool(tool: string): string | null {
  return appFrom(tool, 'open_')
}

/** One-line summary of a persisted trajectory entry: "Asked chess",
 * "Opened chess", "Checked the fleet", or a generic fallback. */
export function describeToolCall(record: ToolCallRecord): string {
  const asked = appFromTool(record.tool)
  if (asked !== null) return `Asked ${asked}`
  const opened = openedAppFromTool(record.tool)
  if (opened !== null) return `Opened ${opened}`
  if (record.tool === 'list_agents') return 'Checked the fleet'
  return `Ran ${record.tool}`
}

/** The live progress line while a turn is in flight. `null` activity (poll
 * hasn't landed yet) reads as thinking — the run always starts on the model. */
export function activityLabel(activity: TurnActivity | null): string {
  if (activity?.kind === 'tool' && activity.tool !== null) {
    const asked = appFromTool(activity.tool)
    if (asked !== null) return `Asking ${asked}…`
    const opened = openedAppFromTool(activity.tool)
    if (opened !== null) return `Opening ${opened}…`
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
