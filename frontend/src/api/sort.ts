import { apiClient } from './client'
import type { Filing, SortStatus } from '../types/sort'

// Music's sorting routes, reached through conductor's fleet action proxy
// (`/api/fleet/{app}/actions/…`, backend `app/api/routes_fleet.py`). The page
// never dials music directly: that would need CORS on music and would make it
// the first app in the fleet a foreign page can write to.
const ACTIONS = '/api/fleet/music/actions'

// These calls are a local file move, not a model turn. If one has not answered
// in a few seconds something is wrong, and a person waiting on a button should
// be told so rather than watching it spin.
const ACTION_TIMEOUT_MS = 10_000

export async function getWorklist(signal?: AbortSignal): Promise<SortStatus> {
  const res = await apiClient(`${ACTIONS}/`, { signal, timeoutMs: ACTION_TIMEOUT_MS })
  return (await res.json()) as SortStatus
}

export async function openGroup(artist: string, signal?: AbortSignal): Promise<SortStatus> {
  const res = await apiClient(`${ACTIONS}/artists/${encodeURIComponent(artist)}`, {
    signal,
    timeoutMs: ACTION_TIMEOUT_MS,
  })
  return (await res.json()) as SortStatus
}

export async function fileSongs(filing: Filing, signal?: AbortSignal): Promise<SortStatus> {
  const res = await apiClient(`${ACTIONS}/filings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(filing),
    signal,
    timeoutMs: ACTION_TIMEOUT_MS,
  })
  return (await res.json()) as SortStatus
}
