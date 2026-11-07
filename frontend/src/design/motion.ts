/** Motion timings (design system #86), mirrored from `--dur-*` in
 * styles/conductor.css for code that animates outside CSS (the 3D scene, timers). */
export const DURATION = {
  receipt: 180,
  receiptTravel: 1000,
  queueStagger: 300,
  panelFade: 500,
  sceneFade: 600,
  waitPulse: 1600,
  hover: 250,
  ring: 1100,
} as const

/** Glitch's error flicker never exceeds this many flashes a second. */
export const GLITCH_MAX_RATE = 3

export function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && typeof window.matchMedia === 'function'
    ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
    : false
}
