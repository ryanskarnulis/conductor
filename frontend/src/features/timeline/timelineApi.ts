import { apiClient } from '../../api/client'
import type { TimelineHit } from '../../types/agent'

/** Search the timeline (#82) over what was said, the replies and the titles
 * the apps' receipts touched. No model runs; incognito and forgotten turns
 * are never hits. Lives in the search panel's lazy chunk. */
export async function searchTimeline(query: string, signal?: AbortSignal): Promise<TimelineHit[]> {
  const res = await apiClient(`/api/agent/timeline/search?q=${encodeURIComponent(query)}`, { signal })
  return (await res.json()) as TimelineHit[]
}
