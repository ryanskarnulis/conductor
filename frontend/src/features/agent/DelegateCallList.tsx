import { AlertTriangle, ListTree, MessageSquareShare } from 'lucide-react'
import type { ToolCallRecord } from '../../types/agent'
import { describeToolCall } from './delegateCalls'

/** The persisted trajectory above an assistant reply: which apps conductor
 * asked (and whether a call failed). Leaner than PCC's ToolCallList on
 * purpose — conductor mutates nothing locally, so there is no undo and no
 * entity link; the delegate call's outcome lives in the reply itself. */
export function DelegateCallList({
  messageId,
  records,
}: {
  messageId: number
  records: ToolCallRecord[]
}) {
  return (
    <ul className="agent-tool-calls" aria-label="Delegate calls">
      {records.map((record, index) => {
        const failed = record.error !== null
        const Icon = failed
          ? AlertTriangle
          : record.tool === 'list_agents'
            ? ListTree
            : MessageSquareShare
        return (
          <li
            key={`${messageId}-${index}`}
            className={`agent-tool-call${failed ? ' agent-tool-call--failed' : ''}`}
          >
            <Icon size={14} aria-hidden="true" />
            <span className="agent-tool-call-summary">
              {describeToolCall(record)}
              {failed && ' (failed)'}
            </span>
            {failed && <span className="agent-tool-call-error">{record.error}</span>}
          </li>
        )
      })}
    </ul>
  )
}
