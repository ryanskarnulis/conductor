import type { TurnActivity } from '../../types/agent'
import { activityLabel, elapsedSuffix } from './delegateCalls'
import { ConductorMark } from './ConductorMark'

/** The optimistic tail while a run is in flight: the user's bubble plus a
 * live progress line. Conductor turns are slow by construction (one master
 * iteration can wrap a subagent's full loop; analysis asks run ~12–22 s), so
 * unlike PCC's static "Working…" this renders the polled activity beat —
 * "Asking chess… · 15s" — and updates as the loop moves. */
export function PendingExchange({
  text,
  activity,
}: {
  text: string
  activity: TurnActivity | null
}) {
  return (
    <>
      <li className="agent-message agent-message--user">
        <div className="agent-bubble">{text}</div>
      </li>
      <li className="agent-message agent-message--assistant">
        <span className="agent-avatar" aria-hidden="true">
          <ConductorMark size={16} className="agent-mark-working" />
        </span>
        <div className="agent-message-body">
          <div className="agent-bubble agent-bubble--working" role="status">
            <span className="agent-working-dots" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
            {activityLabel(activity)}
            {elapsedSuffix(activity)}
          </div>
        </div>
      </li>
    </>
  )
}
