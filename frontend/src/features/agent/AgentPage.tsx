import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { MessageSquarePlus, SendHorizontal, Trash2, Volume2, VolumeX } from 'lucide-react'
import { formatRelative } from '../../utils/dates'
import { MicButton } from '../../voice/MicButton'
import { playText } from '../../voice/tts'
import { ConductorMark } from './ConductorMark'
import { MessageBubble } from './MessageBubble'
import { PendingExchange } from './PendingExchange'
import { useConversation } from './useConversation'
import { useConversations } from './useConversations'
import { useTurnActivity } from './useTurnActivity'
import { delegateAckText } from './voiceAck'

export function AgentPage() {
  const params = useParams<{ conversationId?: string }>()
  const navigate = useNavigate()
  const activeId = params.conversationId !== undefined ? Number(params.conversationId) : null

  const {
    conversations,
    loading: listLoading,
    error: listError,
    refresh,
    create,
    remove,
  } = useConversations()
  const onExchange = useCallback(() => void refresh(), [refresh])
  const { detail, loading, error, pendingText, send } = useConversation(
    activeId !== null && Number.isFinite(activeId) ? activeId : null,
    onExchange,
  )

  const [draft, setDraft] = useState('')
  const threadEndRef = useRef<HTMLLIElement>(null)
  const sending = pendingText !== null
  // The live progress beat ("asking chess…") polled while the send blocks.
  const activity = useTurnActivity(activeId, sending)

  // Voice-output toggle (fleet UX rule: voice in → voice out, typed in →
  // silent — this mutes even the voiced path). Client-owned: conductor has
  // no server-side user settings, and the flag only governs local playback.
  const [voiceOutput, setVoiceOutput] = useState(
    () => localStorage.getItem('agent-voice-output') !== 'off',
  )
  const toggleVoiceOutput = () => {
    setVoiceOutput((on) => {
      localStorage.setItem('agent-voice-output', on ? 'off' : 'on')
      return !on
    })
  }

  // Voice-initiated turns speak the reply (typed turns never do). Playback
  // is started, not awaited — MicButton's hands-free loop waits on
  // audioIdle() so the mic still reopens only after the reply finishes.
  // localStorage (not the state) is read at speak time: hands-free VAD
  // callbacks close over this function from an earlier render, and the
  // stored flag is the always-current source of truth.
  const voiceTurnRef = useRef(false)
  const ackedAppsRef = useRef(new Set<string>())
  const onTranscript = async (text: string) => {
    voiceTurnRef.current = true
    ackedAppsRef.current = new Set()
    try {
      const reply = await send(text)
      if (reply && localStorage.getItem('agent-voice-output') !== 'off') {
        void playText(reply)
      }
    } finally {
      voiceTurnRef.current = false
    }
  }

  // Slow-turn acknowledgment: a delegated conductor turn can block 12–22 s,
  // so on voice-initiated turns the first activity beat naming each
  // ask_<app> delegate is spoken ("Asking chess…") — typed turns keep the
  // silent progress line only. The reply's playText interrupts a still-
  // playing ack (shared audio element), so they never overlap or queue up.
  useEffect(() => {
    if (!voiceTurnRef.current) return
    const ack = delegateAckText(activity, ackedAppsRef.current)
    if (ack && localStorage.getItem('agent-voice-output') !== 'off') {
      void playText(ack)
    }
  }, [activity])

  // Keep the newest turn in view as messages/pending state arrive.
  // (Guarded call: jsdom has no scrollIntoView.)
  useEffect(() => {
    threadEndRef.current?.scrollIntoView?.({ block: 'end' })
  }, [detail?.messages.length, pendingText, activity])

  const openConversation = (id: number) => navigate(`/c/${id}`)

  const startConversation = async () => {
    const conversation = await create()
    openConversation(conversation.id)
  }

  const removeConversation = async (id: number, title: string | null) => {
    if (!window.confirm(`Delete “${title ?? 'this conversation'}”?`)) return
    await remove(id)
    if (id === activeId) navigate('/')
  }

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    const content = draft.trim()
    if (content === '' || sending || activeId === null) return
    setDraft('')
    await send(content)
  }

  // Enter sends, Shift+Enter makes a newline (the form's onSubmit handles
  // the actual send so both paths stay identical).
  const onComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  return (
    <main className="agent-page">
      <aside className="agent-sidebar" aria-label="Conversations">
        <div className="agent-brand">
          <ConductorMark size={18} />
          <span>Conductor</span>
        </div>

        <button type="button" className="agent-new-chat" onClick={() => void startConversation()}>
          <MessageSquarePlus size={17} aria-hidden="true" />
          New conversation
        </button>

        {listError && (
          <p role="alert" className="error">
            {listError}
          </p>
        )}
        {listLoading && <div className="page-loading">Loading…</div>}
        {!listLoading && conversations.length === 0 && (
          <p className="agent-sidebar-empty">No conversations yet.</p>
        )}

        <ul className="agent-conversation-list">
          {conversations.map((conversation) => (
            <li key={conversation.id}>
              <div
                className={`agent-conversation${conversation.id === activeId ? ' active' : ''}`}
              >
                <button
                  type="button"
                  className="agent-conversation-open"
                  onClick={() => openConversation(conversation.id)}
                >
                  <span className="agent-conversation-title">
                    {conversation.title ?? 'New conversation'}
                  </span>
                  <span className="agent-conversation-time">
                    {formatRelative(conversation.updated_at)}
                  </span>
                </button>
                <button
                  type="button"
                  className="agent-conversation-delete"
                  aria-label={`Delete conversation ${conversation.title ?? conversation.id}`}
                  onClick={() => void removeConversation(conversation.id, conversation.title)}
                >
                  <Trash2 size={15} aria-hidden="true" />
                </button>
              </div>
            </li>
          ))}
        </ul>
      </aside>

      <section className="agent-thread" aria-label="Conductor chat">
        {activeId === null ? (
          <div className="agent-thread-empty">
            <ConductorMark size={22} />
            <h1>Conductor</h1>
            <p>
              One place to talk to the house. Ask for anything an app can do —
              “what’s due today?”, “play e4” — and conductor routes it to the
              right app’s agent and relays the answer.
            </p>
            <button
              type="button"
              className="agent-new-chat"
              onClick={() => void startConversation()}
            >
              <MessageSquarePlus size={17} aria-hidden="true" />
              Start a conversation
            </button>
          </div>
        ) : (
          <>
            {error && (
              <p role="alert" className="error">
                {error}
              </p>
            )}
            {loading && <div className="page-loading">Loading conversation…</div>}

            <ul className="agent-messages">
              {detail?.messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              {pendingText !== null && (
                <PendingExchange text={pendingText} activity={activity} />
              )}
              {!loading && detail?.messages.length === 0 && pendingText === null && (
                <li className="agent-thread-hint">
                  What should the house do? Conductor will route it to the right app.
                </li>
              )}
              {/* Autoscroll sentinel — must live INSIDE the scroll container
                  (the ul), or scrollIntoView can't scroll the thread. */}
              <li ref={threadEndRef} className="agent-thread-sentinel" aria-hidden="true" />
            </ul>

            <form className="agent-composer" onSubmit={(e) => void submit(e)}>
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={onComposerKeyDown}
                placeholder={
                  sending ? 'Conductor is working…' : 'Message conductor (Enter to send)'
                }
                aria-label="Message conductor"
                rows={2}
                disabled={sending}
              />
              <MicButton onTranscript={onTranscript} disabled={sending} />
              <button
                type="button"
                className="agent-voice-toggle"
                onClick={toggleVoiceOutput}
                aria-pressed={voiceOutput}
                aria-label={voiceOutput ? 'Mute spoken replies' : 'Unmute spoken replies'}
                title={voiceOutput ? 'Spoken replies on' : 'Spoken replies off'}
              >
                {voiceOutput ? (
                  <Volume2 size={17} aria-hidden="true" />
                ) : (
                  <VolumeX size={17} aria-hidden="true" />
                )}
              </button>
              <button
                type="submit"
                className="btn--primary agent-send"
                disabled={sending || draft.trim() === ''}
                aria-label="Send message"
              >
                <SendHorizontal size={17} aria-hidden="true" />
              </button>
            </form>
          </>
        )}
      </section>
    </main>
  )
}
