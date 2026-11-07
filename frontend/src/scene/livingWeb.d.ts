import type { SceneState } from '../features/agent/sceneState'

export interface ScenePerf {
  fps: number
  cost: number
  tris: number
  calls: number
  W: number
  H: number
  state: string
  label: string
}

export interface LivingWebHandle {
  /** Re-measure the panel the camera frames around (after a layout change). */
  relayout(): void
  /** Stop the loop and free GPU resources. */
  dispose(): void
}

export function mountLivingWeb(options: {
  canvas: HTMLCanvasElement
  panel: HTMLElement
  device: HTMLElement
  getState: () => SceneState
  view: () => 'desktop' | 'phone'
  reducedMotion: () => boolean
  onReady?: (ms: number) => void
  onPerf?: (perf: ScenePerf) => void
  onFail?: (error: unknown) => void
}): LivingWebHandle | null
