import ReactMarkdown, { type Components } from 'react-markdown'
import type { AgentMessage, AgentStopReason } from '../../types/agent'
import { formatRelative } from '../../utils/dates'
import { ConductorMark } from './ConductorMark'
import { DelegateCallList } from './DelegateCallList'

// react-markdown is safe by default (raw HTML never rendered). Links open in
// a new tab so a stray absolute URL in a reply can't navigate the SPA away.
const MARKDOWN_COMPONENTS: Components = {
  a: ({ node, ...props }) => {
    void node // hast node isn't a DOM prop; strip it before spreading.
    return <a {...props} target="_blank" rel="noopener noreferrer" />
  },
}

const STOP_FALLBACK: Record<Exclude<AgentStopReason, 'completed'>, string> = {
  max_iterations:
    'Conductor hit its step limit before finishing — the delegate calls above still ran.',
  correction_limit:
    'Conductor kept producing invalid tool calls and gave up on this request.',
}

export function MessageBubble({ message }: { message: AgentMessage }) {
  if (message.role === 'user') {
    return (
      <li className="agent-message agent-message--user">
        <div className="agent-bubble">{message.content}</div>
        <span className="agent-message-time">{formatRelative(message.created_at)}</span>
      </li>
    )
  }

  const stopReason = message.stop_reason
  return (
    <li className="agent-message agent-message--assistant">
      <span className="agent-avatar" aria-hidden="true">
        <ConductorMark size={16} />
      </span>
      <div className="agent-message-body">
        {message.tool_calls !== null && message.tool_calls.length > 0 && (
          <DelegateCallList messageId={message.id} records={message.tool_calls} />
        )}
        {message.content !== null && (
          <div className="agent-bubble">
            <ReactMarkdown components={MARKDOWN_COMPONENTS}>{message.content}</ReactMarkdown>
          </div>
        )}
        {stopReason !== null && stopReason !== 'completed' && (
          <p className="agent-stop-note" role="status">
            {STOP_FALLBACK[stopReason]}
          </p>
        )}
        <span className="agent-message-time">{formatRelative(message.created_at)}</span>
      </div>
    </li>
  )
}
