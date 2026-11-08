import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import ReactMarkdown, { type Components } from 'react-markdown'
import {
  ListMusic,
  MessageSquarePlus,
  PanelLeft,
  SendHorizontal,
  Square,
  Trash2,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react'
import { cancelTurn } from '../../api/agent'
import {
  ConfirmationCard,
  Narration,
  PlanSteps,
  Presence,
  Receipts,
  Utterance,
} from '../../design/components'
import { earcons, setSound, soundOn } from '../../design/earcons'
import { prefersReducedMotion } from '../../design/motion'
import { formatRelative } from '../../utils/dates'
import { MicButton } from '../../voice/MicButton'
import { playText } from '../../voice/tts'
import type { AgentStopReason } from '../../types/agent'
import { ConductorMark } from './ConductorMark'
import { activityLabel, elapsedSuffix } from './delegateCalls'
import { nextSceneState, type SceneState } from './sceneState'
import { ShellScene } from './ShellScene'
import { SortPanel } from '../sort/SortPanel'
import { latestSortTurnId } from '../sort/sortTurn'
import { buildTurns, pendingConfirm, presenceFor, type ShellTurn } from './turnView'
import { useConversation } from './useConversation'
import { useConversations } from './useConversations'
import { useTurnActivity } from './useTurnActivity'
import { delegateAckText } from './voiceAck'

// Long enough for the send-off line to be read, short enough that it still
// feels like one motion rather than a page that forgot to redirect.
const HANDOFF_DELAY_MS = 1_200

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
  truncated: 'Conductor’s reply ran past its length limit, so it was dropped.',
  no_progress: 'Conductor kept repeating the same calls and stopped.',
  tool_call_limit: 'Conductor reached its limit of tool calls for one request.',
  cancelled: 'Stopped.',
  provider_error: 'Conductor’s model stopped answering partway through.',
  timed_out: 'Conductor ran out of time partway through.',
  internal_error: 'Something broke inside conductor partway through.',
}

function usePhone(): boolean {
  const query = '(max-width: 700px)'
  const [phone, setPhone] = useState(() => typeof window.matchMedia === 'function' && window.matchMedia(query).matches)
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return
    const mql = window.matchMedia(query)
    const on = () => setPhone(mql.matches)
    mql.addEventListener('change', on)
    return () => mql.removeEventListener('change', on)
  }, [])
  return phone
}

/** One turn of the conversation surface. The latest turn shows everything
 * (plan, confirmation, receipts, reply); earlier turns fold to what was said,
 * their receipts and the reply. */
function TurnView({
  turn,
  latest,
  confirm,
  busy,
  onAnswer,
}: {
  turn: ShellTurn
  latest: boolean
  confirm: { app: string; question: string } | null
  busy: boolean
  onAnswer: (words: string) => void
}) {
  const stop = turn.stopReason !== null && turn.stopReason !== 'completed' ? STOP_FALLBACK[turn.stopReason] : null
  return (
    <article className={`shell-turn${latest ? '' : ' shell-turn--old'}`} aria-label={latest ? 'Latest turn' : 'Earlier turn'}>
      {turn.said !== '' && <Utterance text={turn.said} label={turn.note ? 'You · from the panel' : 'You'} />}
      {latest && turn.steps.length > 0 && <PlanSteps steps={turn.steps} />}
      {latest && confirm && (
        <ConfirmationCard
          app={confirm.app}
          question={confirm.question}
          why={`${confirm.app} asks before this write. Say “yes”, or tap.`}
          actionLabel="Yes, do it"
          onConfirm={() => onAnswer('yes')}
          onDecline={() => onAnswer('not now')}
          disabled={busy}
        />
      )}
      {turn.receipts.length > 0 && <Receipts receipts={latest ? turn.receipts : turn.receipts.slice(-3)} />}
      {turn.reply !== null && turn.reply !== '' && (
        <Narration>
          <ReactMarkdown components={MARKDOWN_COMPONENTS}>{turn.reply}</ReactMarkdown>
        </Narration>
      )}
      {stop && (
        <p className="agent-stop-note" role="status">
          {stop}
        </p>
      )}
    </article>
  )
}

export function AgentPage() {
  const params = useParams<{ conversationId?: string }>()
  const navigate = useNavigate()
  const activeId = params.conversationId !== undefined ? Number(params.conversationId) : null

  const { conversations, loading: listLoading, error: listError, refresh, create, remove } = useConversations()
  const onExchange = useCallback(() => void refresh(), [refresh])
  const { detail, loading, error, pendingText, handoff, send, note } = useConversation(
    activeId !== null && Number.isFinite(activeId) ? activeId : null,
    onExchange,
  )

  // The sort panel opens on a turn where music actually ran its sorting tool
  // and then stays open (see sortTurn.ts) until dismissed.
  const messages = useMemo(() => detail?.messages ?? [], [detail])
  const lastMessage = messages.length > 0 ? messages[messages.length - 1] : undefined
  const sortTurnId = latestSortTurnId(messages)
  const [panelHiddenFor, setPanelHiddenFor] = useState<number | null>(null)

  const [draft, setDraft] = useState('')
  const [threadsOpen, setThreadsOpen] = useState(false)
  const [sound, setSoundState] = useState(soundOn)
  const convoEndRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLElement>(null)
  const deviceRef = useRef<HTMLDivElement>(null)
  const sending = pendingText !== null
  const live = useTurnActivity(activeId, sending)
  const activity = live?.activity ?? null
  const stream = live?.stream ?? null
  const phone = usePhone()
  const [reducedMotion, setReducedMotion] = useState(prefersReducedMotion)
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)')
    const on = () => setReducedMotion(mql.matches)
    mql.addEventListener('change', on)
    return () => mql.removeEventListener('change', on)
  }, [])

  const turns = useMemo(() => buildTurns(messages, pendingText, activity, stream), [messages, pendingText, activity, stream])
  const confirm = sending ? null : pendingConfirm(messages)
  const presence = presenceFor(sending, activity, confirm !== null)
  const latest = turns.length > 0 ? turns[turns.length - 1] : null

  // The scene reads the same facts as the panel, each frame, from this ref.
  const sceneRef = useRef<SceneState>(nextSceneState(null, null, 'idle', null, 0))
  const confirmApp = confirm?.raw.app ?? null
  useEffect(() => {
    sceneRef.current = nextSceneState(sceneRef.current, latest, presence.state, confirmApp, performance.now())
  }, [latest, presence.state, confirmApp])
  const getSceneState = useCallback(() => sceneRef.current, [])

  // Voice-output toggle (fleet UX rule: voice in → voice out, typed in → silent).
  const [voiceOutput, setVoiceOutput] = useState(() => localStorage.getItem('agent-voice-output') !== 'off')
  const toggleVoiceOutput = () => {
    setVoiceOutput((on) => {
      localStorage.setItem('agent-voice-output', on ? 'off' : 'on')
      return !on
    })
  }
  const toggleSound = () => {
    setSoundState((on) => {
      setSound(!on)
      return !on
    })
  }

  // Earcons: a finished turn plucks each app string that returned a receipt;
  // a confirmation sounds once. Off unless Sound is on.
  const heardRef = useRef<number | null>(null)
  useEffect(() => {
    if (!lastMessage || lastMessage.role !== 'assistant' || heardRef.current === lastMessage.id) return
    const first = heardRef.current === null
    heardRef.current = lastMessage.id
    if (first) return
    const apps = new Set<string>()
    for (const call of lastMessage.tool_calls ?? []) {
      if (call.tool.startsWith('ask_') && (call.app_record ?? []).some((e) => e.outcome === 'done' || e.outcome === 'looked_up')) apps.add(call.tool.slice(4))
    }
    apps.forEach((app) => earcons.receipt(app))
    if (pendingConfirm(messages)) earcons.confirmation()
  }, [lastMessage, messages])

  // Voice-initiated turns speak the reply (typed turns never do).
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

  // Slow-turn acknowledgment on voice turns: "Asking chess…" is spoken once per app.
  useEffect(() => {
    if (!voiceTurnRef.current) return
    const ack = delegateAckText(activity, ackedAppsRef.current)
    if (ack && localStorage.getItem('agent-voice-output') !== 'off') {
      void playText(ack)
    }
  }, [activity])

  // The handoff: when a turn ends in `open_<app>`, let the send-off land, then go.
  useEffect(() => {
    if (!handoff) return
    const timer = setTimeout(() => window.location.assign(handoff.url), HANDOFF_DELAY_MS)
    return () => clearTimeout(timer)
  }, [handoff])

  // Keep the newest turn in view (guarded: jsdom has no scrollIntoView).
  useEffect(() => {
    convoEndRef.current?.scrollIntoView?.({ block: 'end' })
  }, [messages.length, pendingText, activity])

  // Escape closes the conversations drawer.
  useEffect(() => {
    if (!threadsOpen) return
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') setThreadsOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [threadsOpen])

  const openConversation = (id: number) => {
    setThreadsOpen(false)
    navigate(`/c/${id}`)
  }
  const startConversation = async () => {
    const conversation = await create()
    openConversation(conversation.id)
  }
  // No confirm prompt: deletes are soft (recoverable server-side).
  const removeConversation = async (id: number) => {
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
  const answer = (words: string) => {
    if (sending || activeId === null) return
    void send(words)
  }
  const stopTurn = () => {
    if (activeId !== null) void cancelTurn(activeId)
  }
  const onComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  return (
    <main className="shell" data-view={phone ? 'phone' : 'desktop'}>
      <header className="shell-bar">
        <button
          type="button"
          className="shell-iconbtn"
          aria-label="Conversations"
          aria-expanded={threadsOpen}
          aria-controls="shell-threads"
          onClick={() => setThreadsOpen((open) => !open)}
        >
          <PanelLeft size={17} aria-hidden="true" />
        </button>
        <span className="shell-brand">
          <ConductorMark size={18} />
          <span className="shell-wordmark">Conductor</span>
        </span>
        <Presence state={presence.state} label={presence.label} live />
        <span className="shell-bar-spacer" />
        <button type="button" className="shell-iconbtn shell-sound" aria-pressed={sound} onClick={toggleSound}>
          {sound ? 'Sound: on' : 'Sound'}
        </button>
      </header>

      <aside id="shell-threads" className="shell-threads" aria-label="Conversations" hidden={!threadsOpen}>
        <div className="shell-threads-head">
          <button type="button" className="agent-new-chat" onClick={() => void startConversation()}>
            <MessageSquarePlus size={17} aria-hidden="true" />
            New conversation
          </button>
          <button type="button" className="shell-iconbtn" aria-label="Close conversations" onClick={() => setThreadsOpen(false)}>
            <X size={17} aria-hidden="true" />
          </button>
        </div>
        {listError && (
          <p role="alert" className="error">
            {listError}
          </p>
        )}
        {listLoading && <div className="page-loading">Loading…</div>}
        {!listLoading && conversations.length === 0 && <p className="agent-sidebar-empty">No conversations yet.</p>}
        <ul className="agent-conversation-list">
          {conversations.map((conversation) => (
            <li key={conversation.id}>
              <div className={`agent-conversation${conversation.id === activeId ? ' active' : ''}`}>
                <button type="button" className="agent-conversation-open" onClick={() => openConversation(conversation.id)}>
                  <span className="agent-conversation-title">{conversation.title ?? 'New conversation'}</span>
                  <span className="agent-conversation-time">{formatRelative(conversation.updated_at)}</span>
                </button>
                <button
                  type="button"
                  className="agent-conversation-delete"
                  aria-label={`Delete conversation ${conversation.title ?? conversation.id}`}
                  onClick={() => void removeConversation(conversation.id)}
                >
                  <Trash2 size={15} aria-hidden="true" />
                </button>
              </div>
            </li>
          ))}
        </ul>
      </aside>

      <div className="shell-stage" ref={deviceRef}>
        <ShellScene getState={getSceneState} phone={phone} reducedMotion={reducedMotion} panelRef={panelRef} deviceRef={deviceRef} />

        <section className="shell-panel" ref={panelRef} aria-label="Conductor chat">
          {activeId === null ? (
            <div className="shell-welcome">
              <h1>Conductor</h1>
              <p>
                One place to talk to the house. Ask for anything an app can do — “what’s due today?”, “play e4” — and
                conductor routes it to the right app’s agent and relays the answer.
              </p>
              <button type="button" className="agent-new-chat" onClick={() => void startConversation()}>
                <MessageSquarePlus size={17} aria-hidden="true" />
                Start a conversation
              </button>
            </div>
          ) : (
            <>
              <div className="shell-convo" aria-label="Conversation">
                {error && (
                  <p role="alert" className="error">
                    {error}
                  </p>
                )}
                {loading && <div className="page-loading">Loading conversation…</div>}
                {turns.map((turn, index) => (
                  <TurnView
                    key={turn.key}
                    turn={turn}
                    latest={index === turns.length - 1}
                    confirm={index === turns.length - 1 && confirm ? confirm : null}
                    busy={sending}
                    onAnswer={answer}
                  />
                ))}
                {sending && (
                  <p className="shell-working" role="status">
                    <span className="agent-working-dots" aria-hidden="true">
                      <span />
                      <span />
                      <span />
                    </span>
                    {activityLabel(activity)}
                    {elapsedSuffix(activity)}
                  </p>
                )}
                {!loading && messages.length === 0 && !sending && (
                  <p className="agent-thread-hint">What should the house do? Conductor will route it to the right app.</p>
                )}
                {sortTurnId !== null && panelHiddenFor === sortTurnId && (
                  <button type="button" className="sort-reopen" onClick={() => setPanelHiddenFor(null)}>
                    <ListMusic size={14} aria-hidden="true" />
                    Back to sorting
                  </button>
                )}
                {sortTurnId !== null && panelHiddenFor !== sortTurnId && (
                  <SortPanel
                    syncKey={lastMessage?.id ?? 0}
                    onFiled={(text) => void note(text)}
                    onDismiss={() => setPanelHiddenFor(sortTurnId)}
                  />
                )}
                <div ref={convoEndRef} className="agent-thread-sentinel" aria-hidden="true" />
              </div>

              <form className="agent-composer" onSubmit={(e) => void submit(e)}>
                <textarea
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={onComposerKeyDown}
                  placeholder={sending ? 'Conductor is working…' : 'Ask conductor…'}
                  aria-label="Message conductor"
                  rows={1}
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
                  {voiceOutput ? <Volume2 size={17} aria-hidden="true" /> : <VolumeX size={17} aria-hidden="true" />}
                </button>
                {sending ? (
                  <button type="button" className="agent-stop" onClick={stopTurn} aria-label="Stop">
                    <Square size={15} aria-hidden="true" />
                  </button>
                ) : (
                  <button type="submit" className="btn--primary agent-send" disabled={draft.trim() === ''} aria-label="Send message">
                    <SendHorizontal size={17} aria-hidden="true" />
                  </button>
                )}
              </form>
            </>
          )}
        </section>
      </div>
    </main>
  )
}
