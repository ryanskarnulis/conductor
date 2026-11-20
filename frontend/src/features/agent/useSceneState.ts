import { useCallback, useEffect, useRef } from 'react'
import type { PresenceState } from '../../design/status'
import type { JobCardData } from './jobStream'
import { nextSceneState, nextSceneWake, type SceneState } from './sceneState'
import type { ShellTurn } from './turnView'

/** The scene's state, kept in a ref the Living Web reads each frame. It is
 * re-derived when the page's facts change and, with no render, when the clock
 * alone changes it: a finished turn's strands letting go (#232). */
export function useSceneState(
  latest: ShellTurn | null,
  presence: PresenceState,
  confirmApp: string | null,
  jobs: readonly JobCardData[],
): () => SceneState {
  const ref = useRef<SceneState>(nextSceneState(null, null, 'idle', null, 0))
  useEffect(() => {
    let timer: number | undefined
    const derive = () => {
      const now = performance.now()
      ref.current = nextSceneState(ref.current, latest, presence, confirmApp, now, jobs)
      const wake = nextSceneWake(ref.current, now)
      timer = wake === null ? undefined : window.setTimeout(derive, Math.max(0, Math.ceil(wake - now)))
    }
    derive()
    return () => window.clearTimeout(timer)
  }, [latest, presence, confirmApp, jobs])
  return useCallback(() => ref.current, [])
}
