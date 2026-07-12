// Trimmed from PCC's utils/dates.ts: conductor only needs the relative label.

const RELATIVE_UNITS: [seconds: number, name: string][] = [
  [31_536_000, 'year'],
  [2_592_000, 'month'],
  [604_800, 'week'],
  [86_400, 'day'],
  [3_600, 'hour'],
  [60, 'minute'],
]

/**
 * A coarse "3 days ago" style label for a past ISO timestamp. Returns "just now"
 * under a minute. `now` is injectable for deterministic tests.
 */
export function formatRelative(iso: string, now: number = Date.now()): string {
  const diffSec = Math.floor((now - new Date(iso).getTime()) / 1000)
  if (diffSec < 60) return 'just now'
  for (const [seconds, name] of RELATIVE_UNITS) {
    const value = Math.floor(diffSec / seconds)
    if (value >= 1) return `${value} ${name}${value === 1 ? '' : 's'} ago`
  }
  return 'just now'
}
