import { useEffect, useRef, useState, type RefObject } from 'react'
import type { SceneState } from './sceneState'

/** The Living Web behind the panel. three.js and the scene load after first
 * paint (a separate chunk), and only where WebGL 2 exists; until then, or
 * without it, a still drawing of the web stands in. Decoration only. */
export function ShellScene({
  getState,
  phone,
  reducedMotion,
  panelRef,
  deviceRef,
}: {
  getState: () => SceneState
  phone: boolean
  reducedMotion: boolean
  panelRef: RefObject<HTMLElement | null>
  deviceRef: RefObject<HTMLElement | null>
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const viewRef = useRef(phone)
  const rmRef = useRef(reducedMotion)
  const stateRef = useRef(getState)
  const handleRef = useRef<{ relayout(): void; dispose(): void } | null>(null)
  const [perf, setPerf] = useState<string | null>(null)
  const showPerf = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('perf')

  useEffect(() => {
    viewRef.current = phone
    rmRef.current = reducedMotion
    stateRef.current = getState
    handleRef.current?.relayout()
  })

  useEffect(() => {
    const canvas = canvasRef.current
    const panel = panelRef.current
    const device = deviceRef.current
    if (!canvas || !panel || !device || typeof window.WebGL2RenderingContext === 'undefined') return
    let cancelled = false
    // Let the text paint first; the scene is never on the critical path.
    const start = window.setTimeout(() => {
      void import('../../scene/livingWeb.js').then(({ mountLivingWeb }) => {
        if (cancelled) return
        handleRef.current = mountLivingWeb({
          canvas,
          panel,
          device,
          getState: () => stateRef.current(),
          view: () => (viewRef.current ? 'phone' : 'desktop'),
          reducedMotion: () => rmRef.current,
          onPerf: showPerf ? (p) => setPerf(p.label) : undefined,
          onFail: () => canvas.setAttribute('hidden', ''),
        })
      })
    }, 30)
    return () => {
      cancelled = true
      window.clearTimeout(start)
      handleRef.current?.dispose()
      handleRef.current = null
    }
  }, [panelRef, deviceRef, showPerf])

  return (
    <>
      <svg className="shell-poster" viewBox="0 0 1000 625" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
        <g fill="none" stroke="rgba(155,200,255,.13)" strokeWidth="1">
          <circle cx="330" cy="300" r="70" />
          <circle cx="330" cy="300" r="150" />
          <circle cx="330" cy="300" r="240" />
          <path d="M330 300L620 300M330 300L581 445M330 300L475 551M330 300L330 590M330 300L185 551M330 300L79 445M330 300L40 300M330 300L79 155M330 300L185 49M330 300L330 10M330 300L475 49M330 300L581 155" />
        </g>
        <circle cx="330" cy="300" r="9" fill="#00c3ff" opacity=".7" />
      </svg>
      <canvas ref={canvasRef} className="shell-canvas" aria-hidden="true" />
      {showPerf && perf && (
        <span className="shell-perf" aria-hidden="true">
          {perf}
        </span>
      )}
    </>
  )
}
