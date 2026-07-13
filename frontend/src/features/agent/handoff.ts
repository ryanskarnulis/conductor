import type { ToolCallRecord } from '../../types/agent'

/** Reading a handoff off a finished turn: some apps are places the user goes,
 * not services conductor calls. When the agent runs an `open_<app>` tool the
 * backend returns a payload naming where to send them; this turns that into
 * the URL the page navigates to. */

/** The backend's `open_<app>` payload (app/fleet/tools.py::_make_open_tool). */
interface HandoffPayload {
  handoff: string
  title: string
  path: string
  upstream: string
  intent_param?: string
  intent?: string
}

export interface Handoff {
  /** The app to send the user to — the manifest name, which is also its
   * subdomain (`chess` → chess.home.lan). */
  app: string
  /** Human title, for the "Opening Chess…" line. */
  title: string
  /** Absolute URL to navigate to, intent already encoded. */
  url: string
}

function isPayload(value: unknown): value is HandoffPayload {
  if (typeof value !== 'object' || value === null) return false
  const p = value as Record<string, unknown>
  return (
    typeof p.handoff === 'string' &&
    typeof p.title === 'string' &&
    typeof p.path === 'string' &&
    typeof p.upstream === 'string'
  )
}

const IPV4 = /^\d{1,3}(\.\d{1,3}){3}$/

/**
 * The app's front door, as seen from wherever this page is being served.
 *
 * The gateway gives every app a vhost at `<name>.<domain>`, so we swap our own
 * leading label for theirs: conductor.home.lan → chess.home.lan. That keeps the
 * domain out of conductor's config — whatever host served this page names the
 * one the user can actually reach.
 *
 * A host with no domain to swap into (dev's `localhost:5174`, or a raw LAN IP)
 * isn't behind the gateway at all, so we dial the app's upstream `host:port`
 * direct.
 */
function frontDoor(payload: HandoffPayload): URL {
  const { hostname, port, protocol } = window.location
  const labels = hostname.split('.')
  const behindGateway = labels.length > 1 && !IPV4.test(hostname)
  if (!behindGateway) return new URL(payload.path, `http://${payload.upstream}`)

  const domain = labels.slice(1).join('.')
  const authority = `${payload.handoff}.${domain}${port ? `:${port}` : ''}`
  return new URL(payload.path, `${protocol}//${authority}`)
}

/** The handoff an assistant turn asked for, or null if it didn't ask for one.
 * Only a *successful* open call counts — a failed tool call navigates nowhere. */
export function handoffFrom(toolCalls: ToolCallRecord[] | null): Handoff | null {
  const call = (toolCalls ?? []).find(
    (record) => record.tool.startsWith('open_') && record.error === null && record.result !== null,
  )
  if (!call) return null

  let payload: unknown
  try {
    payload = JSON.parse(call.result as string)
  } catch {
    return null
  }
  if (!isPayload(payload)) return null

  const url = frontDoor(payload)
  // The user's own words ride along so the app's agent picks up where
  // conductor left off; `searchParams` handles the encoding.
  if (payload.intent_param && payload.intent) {
    url.searchParams.set(payload.intent_param, payload.intent)
  }
  return { app: payload.handoff, title: payload.title, url: url.toString() }
}
