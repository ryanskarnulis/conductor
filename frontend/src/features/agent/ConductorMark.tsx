import { Waypoints } from 'lucide-react'

/** Conductor's little avatar mark — the routing glyph the assistant bubbles
 * and empty state share (PCC's SpiderMark slot). */
export function ConductorMark({ size = 16, className }: { size?: number; className?: string }) {
  return <Waypoints size={size} className={className} aria-hidden="true" />
}
