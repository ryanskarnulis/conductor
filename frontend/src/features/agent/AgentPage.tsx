import { Fragment, lazy, Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import { Ear, EarOff, EyeOff, ListMusic, SendHorizontal, Square, Volume2, VolumeX } from 'lucide-react'
import { cancelTurn, getTimeline, type SendOptions } from '../../api/agent'
import {
  Ack,
  ConfirmationCard,
  Narration,
  PlanSteps,
  Presence,
  QuestionCard,
  Receipts,
  ResumedNote,
  ThreadHead,
  ThreadNote,
  ThreadsClosed,
  ThreadStart,
  ThreadTabs,
  ToggleMenu,
  Utterance,
} from '../../design/components'
import { earcons, setSound, soundOn } from '../../design/earcons'
import { prefersReducedMotion } from '../../design/motion'
import { MicButton } from '../../voice/MicButton'
import { playText, type Heard } from '../../voice/tts'
import type { AgentStopReason, Surface, Timeline } from '../../types/agent'
import { ConductorMark } from './ConductorMark'
import { activityLabel, elapsedSuffix } from './delegateCalls'
import { JobCards, JobNotes } from './JobCards'
import { groupJobNotes, jobCards, type TurnItem } from './jobStream'
import { useJobs } from './useJobs'
import { nextSceneState, type SceneState } from './sceneState'
import { ShellScene } from './ShellScene'
import { SortPanel } from '../sort/SortPanel'
import { latestSortTurnId } from '../sort/sortTurn'
import {
  REVEAL_BATCH,
  dayLabel,
  daySeparators,
  fromEnd,
  incognitoLapsed,
  keptScrollTop,
  revealedFor,
  surfaceLabel,
  windowStart,
  withLiveWords,
  type ScrollBox,
} from './timeline'
import type { TurnStream } from './turnStream'
import { buildTurns, pendingConfirm, presenceFor, type ShellTurn } from './turnView'
import { useConversation } from './useConversation'
import { carryChips, useThreads } from './useThreads'
import { useTurnActivity } from './useTurnActivity'
import { delegateAckText } from './voiceAck'

// The memory list (#81) and timeline search (#82) are their own lazy chunks:
// the entry never carries them.
const MemoryPanel = lazy(() => import('../memory/MemoryPanel'))
const TimelineSearch = lazy(() => import('../timeline/TimelineSearch'))

// Long enough for the send-off line to be read, short enough that it still
// feels like one motion rather than a page that forgot to redirect.
const HANDOFF_DELAY_MS = 1_200
// How often the incognito switch checks whether the quiet gap has passed.
const INCOGNITO_CHECK_MS = 60_000

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

/** Who said it, tagged with where it came from (#82) and whether it was
 * incognito. The live, in-flight turn has no tag yet. */
function SaidLabel({ turn }: { turn: ShellTurn }) {
  const who = turn.note ? 'You · from the panel' : 'You'
  if (turn.live) return <>{who}</>
  return (
    <>
      {who}
      <span className="tl-tag">{surfaceLabel(turn.surface)}</span>
      {turn.incognito && <span className="tl-tag tl-tag--incognito">incognito</span>}
    </>
  )
}

/** One turn of the timeline. The latest turn shows everything (plan,
 * confirmation, receipts, reply); earlier turns fold to what was said, their
 * receipts and the reply. */
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
  const classes = `shell-turn${latest ? '' : ' shell-turn--old'}${turn.incognito ? ' shell-turn--incognito' : ''}`
  return (
    <article id={`turn-${turn.key}`} className={classes} aria-label={latest ? 'Latest turn' : 'Earlier turn'}>
      {turn.said !== '' && <Utterance text={turn.said} label={<SaidLabel turn={turn} />} />}
      {latest && turn.ack && <Ack text={turn.ack} />}
      {latest && turn.resumedNote && <ResumedNote text={turn.resumedNote} />}
      {latest && turn.steps.length > 0 && <PlanSteps steps={turn.steps} />}
      {latest && turn.announce && (
        <div className="cd-sr" aria-live="polite">
          {turn.announce}
        </div>
      )}
      {latest && turn.pause?.kind === 'question' && (
        <QuestionCard
          question={turn.pause.question}
          choices={turn.pause.options.map((label) => ({ label }))}
          onChoose={onAnswer}
          disabled={busy}
          spoken={turn.pause.options.length > 0 ? 'Or say it.' : 'Say or type your answer.'}
        />
      )}
      {latest && turn.pause?.kind === 'confirm' && (
        <ConfirmationCard
          app={turn.pause.app}
          question={turn.pause.question}
          why={turn.pause.why}
          actionLabel="Yes, do it"
          onConfirm={() => onAnswer('yes')}
          onDecline={() => onAnswer('not now')}
          disabled={busy}
        />
      )}
      {latest && !turn.pause && confirm && (
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

/** How long after a touch or wheel a scroll counts as the person's own. */
const GESTURE_MS = 1500

function withEnds(items: TurnItem[]): { item: TurnItem; end: number }[] {
  let end = -1
  return items.map((item) => {
    end += item.kind === 'notes' ? item.turns.length : 1
    return { item, end }
  })
}

/** The one timeline (#82), created on first use; a failed read says so. */
function useTimeline(): { timeline: Timeline | null; error: string | null } {
  const [timeline, setTimeline] = useState<Timeline | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let active = true
    getTimeline().then(
      (found) => {
        if (active) setTimeline(found)
      },
      (e: unknown) => {
        if (active) setError(e instanceof Error ? e.message : 'Failed to load the timeline')
      },
    )
    return () => {
      active = false
    }
  }, [])
  return { timeline, error }
}

export function AgentPage() {
  const { timeline, error: timelineError } = useTimeline()
  // Side threads (#139): the page shows one conversation at a time, the
  // timeline or a thread; each keeps its own turns, stream and jobs.
  const threads = useThreads(timeline !== null)
  const thread = threads.current
  const threadClosed = thread !== null && thread.closed_at !== null
  const activeId = thread?.id ?? timeline?.id ?? null
  const [startOpen, setStartOpen] = useState(false)
  const [threadName, setThreadName] = useState('')

  const { detail, loading, error, pendingText, handoff, send, liveWords, note, refresh: refreshThread } = useConversation(activeId)

  // The sort panel opens on a turn where music actually ran its sorting tool
  // and then stays open (see sortTurn.ts) until dismissed. This session's
  // incognito turns show the words the person read live (#82).
  const messages = useMemo(() => withLiveWords(detail?.messages ?? [], liveWords), [detail, liveWords])
  const lastMessage = messages.length > 0 ? messages[messages.length - 1] : undefined
  const sortTurnId = latestSortTurnId(messages)
  const [panelHiddenFor, setPanelHiddenFor] = useState<number | null>(null)

  const [draft, setDraft] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const [memoryOpen, setMemoryOpen] = useState(false)
  const [sound, setSoundState] = useState(soundOn)
  // Incognito (#82): per request, off when flipped back or after the quiet gap.
  const [incognito, setIncognito] = useState(false)
  const incognitoUsedAt = useRef<number | null>(null)
  const convoEndRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLElement>(null)
  const deviceRef = useRef<HTMLDivElement>(null)
  const sending = pendingText !== null
  // The turn that ended on a question keeps its stream, so its plan and card
  // stay on screen until the next turn: the answer resumes that same plan.
  const [held, setHeld] = useState<TurnStream | null>(null)
  const onTurnEnded = useCallback((ended: TurnStream) => setHeld(ended.paused !== null ? ended : null), [])
  const live = useTurnActivity(activeId, sending, onTurnEnded)
  // Jobs the timeline started (#90): followed while no send is in flight,
  // each a card until its note lands in the thread.
  const jobs = useJobs(activeId, !sending, refreshThread)
  const cards = useMemo(() => jobCards(jobs.state, jobs.asks, new Set(messages.map((m) => m.id))), [jobs.state, jobs.asks, messages])
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

  const turns = useMemo(() => buildTurns(messages, pendingText, activity, stream, held), [messages, pendingText, activity, stream, held])
  // The page opens on the current window (#226): turns and notes from before
  // the timeline's last quiet gap fold away behind "Show earlier". A side
  // thread is read whole. The clock is read when the page opens and again
  // when it comes back into view, so a phone resumed a day later folds too.
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const on = () => {
      if (document.visibilityState === 'visible') setNow(Date.now())
    }
    document.addEventListener('visibilitychange', on)
    return () => document.removeEventListener('visibilitychange', on)
  }, [])
  // How many items above the fold are shown (#233): "Show earlier" adds a batch.
  const [revealed, setRevealed] = useState(0)
  const windowAt = useMemo(
    () => (thread === null ? windowStart(turns, now, (timeline?.quiet_gap_seconds ?? 0) * 1000) : 0),
    [thread, turns, now, timeline],
  )
  // Each item with the index of its last turn: a group of job notes shows
  // whole when any of it is in the window.
  const items = useMemo(() => withEnds(groupJobNotes(turns)), [turns])
  const folded = items.findIndex(({ end }) => end >= windowAt)
  const foldAt = windowAt === 0 ? 0 : folded < 0 ? items.length : folded
  const shownFrom = Math.max(0, foldAt - revealed)
  const shown = useMemo(() => items.slice(shownFrom), [items, shownFrom])
  const separators = useMemo(
    () => daySeparators(shown.map(({ item }) => (item.kind === 'notes' ? item.turns[0]?.at : item.turn.when))),
    [shown],
  )
  const confirm = sending ? null : pendingConfirm(messages)
  // A job's note or a side thread's result lands in the conversation but is
  // never the turn the page is about.
  const latestAt = turns.findLastIndex((turn) => !turn.jobNote && !turn.threadNote)
  const latest = latestAt >= 0 ? turns[latestAt] : null
  const pause = !sending && latest ? latest.pause : null
  const presence = presenceFor(sending, activity, confirm !== null || pause !== null)
  // What came before the latest turn (#233). On the timeline it is one block
  // above it: a phone's sheet fits the latest turn and the block is reached
  // by scrolling up. A side thread keeps its one list.
  const aboveCount = thread === null ? Math.max(0, shown.findIndex(({ end }) => end >= latestAt)) : 0
  const above = shown.slice(0, aboveCount)
  const convoRef = useRef<HTMLDivElement>(null)
  const aboveRef = useRef<HTMLDivElement>(null)
  // Reading back: the person scrolled up, asked for earlier turns or opened
  // a search hit. It lasts until the next turn is sent or arrives.
  const turnMark = `${messages.length}:${pendingText === null ? '' : 'sending'}`
  const [readingAt, setReadingAt] = useState<string | null>(null)
  const reading = readingAt === turnMark
  const gestureAt = useRef(0)
  const onGesture = () => {
    gestureAt.current = Date.now()
  }
  const onConvoScroll = () => {
    const el = convoRef.current
    if (!el || !phone || reading || Date.now() - gestureAt.current > GESTURE_MS) return
    if (fromEnd(el) > 8) setReadingAt(turnMark)
  }
  // A phone's sheet is as tall as the latest turn, so the web keeps its room
  // above it (the reason the phone once hid earlier turns). They are in the
  // scroll area now, just above the top edge; reading back gives the sheet
  // its full height. Measured on every render: a streaming turn grows.
  const fitRef = useRef<number | null>(null)
  useLayoutEffect(() => {
    const el = convoRef.current
    if (!el) return
    const block = phone && !reading ? aboveRef.current : null
    const fit = block ? Math.round(el.scrollHeight - (block.getBoundingClientRect().bottom - el.getBoundingClientRect().top + el.scrollTop)) : null
    if (fit === fitRef.current) return
    fitRef.current = fit
    el.style.maxHeight = fit === null ? '' : `${fit}px`
    // At the end, where the latest turn is; the thread form opens at its top.
    if (fit !== null) el.scrollTop = el.scrollHeight - (startOpen ? fit : 0)
  })
  // "Show earlier" brings a batch back above and keeps the place.
  const placeRef = useRef<ScrollBox | null>(null)
  const showEarlier = () => {
    const el = convoRef.current
    if (el) placeRef.current = { scrollHeight: el.scrollHeight, scrollTop: el.scrollTop, clientHeight: el.clientHeight }
    setRevealed((n) => n + REVEAL_BATCH)
    setReadingAt(turnMark)
  }
  useLayoutEffect(() => {
    const el = convoRef.current
    const was = placeRef.current
    placeRef.current = null
    if (el && was) el.scrollTop = keptScrollTop(was, el)
  }, [revealed])

  // The scene reads the same facts as the panel, each frame, from this ref.
  const sceneRef = useRef<SceneState>(nextSceneState(null, null, 'idle', null, 0))
  const confirmApp = confirm?.raw.app ?? (pause?.kind === 'confirm' ? pause.appKey : null)
  useEffect(() => {
    sceneRef.current = nextSceneState(sceneRef.current, latest, presence.state, confirmApp, performance.now(), cards)
  }, [latest, presence.state, confirmApp, cards])
  const getSceneState = useCallback(() => sceneRef.current, [])

  // Voice-output toggle (fleet UX rule: voice in → voice out, typed in → silent).
  const [voiceOutput, setVoiceOutput] = useState(() => localStorage.getItem('agent-voice-output') !== 'off')
  const toggleVoiceOutput = () => {
    setVoiceOutput((on) => {
      localStorage.setItem('agent-voice-output', on ? 'off' : 'on')
      return !on
    })
  }
  // Barge-in (#77): talking over a spoken reply stops it, and the next turn
  // says how much of it was heard. On unless the person turned it off.
  const [bargeIn, setBargeIn] = useState(() => localStorage.getItem('agent-barge-in') !== 'off')
  const toggleBargeIn = () => {
    setBargeIn((on) => {
      localStorage.setItem('agent-barge-in', on ? 'off' : 'on')
      return !on
    })
  }
  const toggleSound = () => {
    setSoundState((on) => {
      setSound(!on)
      return !on
    })
  }

  // Incognito turns itself off after the same quiet gap the planner uses.
  const gapMs = (timeline?.quiet_gap_seconds ?? 0) * 1000
  const toggleIncognito = () => {
    incognitoUsedAt.current = incognito ? null : Date.now()
    setIncognito(!incognito)
  }
  useEffect(() => {
    if (!incognito) return
    const timer = setInterval(() => {
      if (incognitoLapsed(incognitoUsedAt.current, Date.now(), gapMs)) {
        incognitoUsedAt.current = null
        setIncognito(false)
      }
    }, INCOGNITO_CHECK_MS)
    return () => clearInterval(timer)
  }, [incognito, gapMs])

  /** How this turn goes out: its surface, incognito while it is on (and the
   * quiet gap hasn't lapsed it), and what was heard of a reply the person
   * spoke over (never with an incognito turn). Nothing extra for a plain
   * typed turn. */
  const sendOptions = (surface?: Surface, heard?: Heard): SendOptions | undefined => {
    let hidden = incognito
    if (hidden && incognitoLapsed(incognitoUsedAt.current, Date.now(), gapMs)) {
      hidden = false
      incognitoUsedAt.current = null
      setIncognito(false)
    }
    if (hidden) incognitoUsedAt.current = Date.now()
    if (!surface && !hidden) return undefined
    return {
      ...(surface ? { surface } : {}),
      ...(hidden ? { incognito: true } : {}),
      ...(heard && !hidden ? { heard } : {}),
    }
  }
  const say = (words: string, surface?: Surface, heard?: Heard) => send(words, sendOptions(surface, heard))

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

  // Voice-initiated turns speak the reply (typed turns never do), and are
  // tagged as voice on the timeline (#82).
  const voiceTurnRef = useRef(false)
  const ackedAppsRef = useRef(new Set<string>())
  const transcribed = async (text: string, heard?: Heard) => {
    voiceTurnRef.current = true
    ackedAppsRef.current = new Set()
    try {
      const reply = await say(text, 'voice', heard)
      if (reply && localStorage.getItem('agent-voice-output') !== 'off') {
        void playText(reply)
      }
    } finally {
      voiceTurnRef.current = false
    }
  }
  // The mic holds the callback it was given when the conversation started,
  // so it calls through a ref: an incognito switch flipped mid-conversation
  // applies to the next utterance.
  const transcribedRef = useRef(transcribed)
  useEffect(() => {
    transcribedRef.current = transcribed
  })
  const onTranscript = useCallback(
    (text: string, _interactionId?: string, heard?: Heard) => transcribedRef.current(text, heard),
    [],
  )

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

  // Keep the newest turn in view (guarded: jsdom has no scrollIntoView). A
  // fitted phone sheet sits at its very end, so no earlier turn peeks in.
  useEffect(() => {
    const el = convoRef.current
    if (el && fitRef.current !== null) el.scrollTop = el.scrollHeight
    else convoEndRef.current?.scrollIntoView?.({ block: 'end' })
  }, [messages.length, pendingText, activity])

  // A search hit brings its turn into view, when it's on the page.
  const [openTo, setOpenTo] = useState<number | null>(null)
  useEffect(() => {
    if (openTo === null || searchOpen) return
    document.getElementById(`turn-m${openTo}`)?.scrollIntoView?.({ block: 'center' })
  }, [openTo, searchOpen])
  const openHit = (messageId: number | null) => {
    setSearchOpen(false)
    // Unfold as far as the hit, and no further.
    const key = `m${messageId}`
    const at = items.findIndex(({ item }) => (item.kind === 'notes' ? item.turns.some((turn) => turn.key === key) : item.turn.key === key))
    setRevealed((n) => revealedFor(at, foldAt, n))
    setReadingAt(turnMark)
    setOpenTo(messageId)
  }

  const earlierButton = !thread && shownFrom > 0 && (
    <button type="button" className="cd-btn cd-btn--small tl-earlier" onClick={showEarlier}>
      Show earlier
    </button>
  )
  const day = (index: number) => {
    const at = separators.get(index)
    return (
      at && (
        <p className="tl-day" role="separator">
          {dayLabel(at)}
        </p>
      )
    )
  }
  // The first turn below the block has its day drawn at the block's end.
  const renderItem = (item: TurnItem, index: number) => (
    <Fragment key={item.kind === 'notes' ? item.key : item.turn.key}>
      {(above.length === 0 || index !== above.length) && day(index)}
      {item.kind === 'notes' ? (
        <JobNotes turns={item.turns} />
      ) : item.turn.threadNote ? (
        <ThreadNote
          title={item.turn.threadNote.title ?? 'Side thread'}
          text={item.turn.reply ?? ''}
          onOpen={item.turn.threadNote.id > 0 ? () => void threads.view(item.turn.threadNote?.id ?? 0) : undefined}
        />
      ) : (
        <TurnView
          turn={item.turn}
          latest={item.index === latestAt}
          confirm={item.index === latestAt && confirm ? confirm : null}
          busy={sending}
          onAnswer={answer}
        />
      )}
    </Fragment>
  )

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    const content = draft.trim()
    if (content === '' || sending || activeId === null) return
    setDraft('')
    await say(content)
  }
  const answer = (words: string) => {
    if (sending || activeId === null) return
    void say(words)
  }
  const stopTurn = () => {
    if (activeId !== null) void cancelTurn(activeId)
  }
  const beginThread = async () => {
    if (await threads.start(incognito ? null : threadName.trim() || null, incognito)) {
      setStartOpen(false)
      setThreadName('')
    }
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
        <span className="shell-brand">
          <ConductorMark size={18} />
          <span className="shell-wordmark">Conductor</span>
        </span>
        <Presence state={presence.state} label={presence.label} live />
        <span className="shell-bar-spacer" />
        <button
          type="button"
          className="shell-iconbtn"
          aria-expanded={startOpen}
          aria-controls="shell-thread-start"
          disabled={timeline === null || sending}
          onClick={() => {
            setSearchOpen(false)
            setMemoryOpen(false)
            if (!startOpen) threads.loadRecent()
            setStartOpen((open) => !open)
          }}
        >
          Side thread
        </button>
        <button
          type="button"
          className="shell-iconbtn"
          aria-label="Search the timeline"
          aria-expanded={searchOpen}
          aria-controls="shell-search"
          onClick={() => {
            setMemoryOpen(false)
            setSearchOpen((open) => !open)
          }}
        >
          Search
        </button>
        <button
          type="button"
          className="shell-iconbtn"
          aria-expanded={memoryOpen}
          aria-controls="shell-memory"
          onClick={() => {
            setSearchOpen(false)
            setMemoryOpen((open) => !open)
          }}
        >
          Memory
        </button>
        <button type="button" className="shell-iconbtn shell-sound" aria-pressed={sound} onClick={toggleSound}>
          {sound ? 'Sound: on' : 'Sound'}
        </button>
      </header>

      <div className="shell-stage" ref={deviceRef}>
        <ShellScene getState={getSceneState} phone={phone} reducedMotion={reducedMotion} panelRef={panelRef} deviceRef={deviceRef} />

        <section className="shell-panel" ref={panelRef} aria-label="Conductor chat">
          {memoryOpen ? (
            <Suspense fallback={null}>
              <MemoryPanel conversationId={activeId} onClose={() => setMemoryOpen(false)} onChanged={refreshThread} />
            </Suspense>
          ) : searchOpen ? (
            <Suspense fallback={null}>
              <TimelineSearch onClose={() => setSearchOpen(false)} onOpen={openHit} />
            </Suspense>
          ) : activeId === null ? (
            <div className="shell-welcome">
              <h1>Conductor</h1>
              {timelineError ? (
                <p role="alert" className="error">
                  {timelineError}
                </p>
              ) : (
                <div className="page-loading">Loading the timeline…</div>
              )}
            </div>
          ) : (
            <>
              {(threads.open.length > 0 || thread !== null) && (
                <ThreadTabs
                  threads={threads.open.map((t) => ({ id: t.id, title: t.title ?? 'Side thread' }))}
                  active={thread !== null && !threadClosed ? thread.id : null}
                  onSelect={threads.select}
                  disabled={sending || threads.closing}
                />
              )}
              <div
                className="shell-convo"
                aria-label={thread ? `Side thread: ${thread.title ?? 'Side thread'}` : 'Timeline'}
                data-thread={thread ? '' : undefined}
                ref={convoRef}
                onScroll={onConvoScroll}
                onWheel={onGesture}
                onTouchMove={onGesture}
              >
                {above.length > 0 && (
                  <div className="tl-above" ref={aboveRef}>
                    {earlierButton}
                    {above.map(({ item }, index) => renderItem(item, index))}
                    {day(above.length)}
                  </div>
                )}
                {startOpen && (
                  <div id="shell-thread-start">
                    <ThreadStart
                      name={threadName}
                      incognito={incognito}
                      busy={threads.starting}
                      error={threads.error}
                      onName={setThreadName}
                      onStart={() => void beginThread()}
                      onCancel={() => setStartOpen(false)}
                    />
                    <ThreadsClosed
                      threads={threads.recent.map((t) => ({
                        id: t.id,
                        title: t.title ?? 'Side thread',
                        closed: t.closed_at ? dayLabel(t.closed_at).replace(/^(Today|Yesterday)$/, (day) => day.toLowerCase()) : '',
                      }))}
                      onOpen={(id) => {
                        setStartOpen(false)
                        void threads.view(id)
                      }}
                    />
                  </div>
                )}
                {!startOpen && threads.error && (
                  <p role="alert" className="error">
                    {threads.error}
                  </p>
                )}
                {thread && (
                  <ThreadHead
                    title={thread.title ?? 'Side thread'}
                    carries={carryChips(thread)}
                    result={thread.closed_at !== null ? thread.result : null}
                    closing={threads.closing}
                    onDone={sending ? undefined : () => void threads.close()}
                    onBack={() => threads.select(null)}
                  />
                )}
                {error && (
                  <p role="alert" className="error">
                    {error}
                  </p>
                )}
                {loading && <div className="page-loading">{thread ? 'Loading the side thread…' : 'Loading the timeline…'}</div>}
                {above.length === 0 && earlierButton}
                {shown.slice(above.length).map(({ item }, index) => renderItem(item, index + above.length))}
                <JobCards
                  cards={cards}
                  announce={jobs.state.announce}
                  errors={jobs.errors}
                  onCancel={(card) => void jobs.cancel(card.app, card.ref)}
                />

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
                {!loading && shown.length === 0 && !sending && (
                  <p className="agent-thread-hint">
                    {thread
                      ? 'A side thread starts with only what it carries. Ask something below; it stays in here until you close it.'
                      : 'What should the house do? Conductor will route it to the right app.'}
                  </p>
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

              {incognito && !threadClosed && (
                <p className="tl-incognito-note" role="status">
                  {timeline?.incognito_notice}
                </p>
              )}
              {!threadClosed && (
                <form className="agent-composer" onSubmit={(e) => void submit(e)}>
                  <textarea
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={onComposerKeyDown}
                    placeholder={
                      sending
                        ? 'Conductor is working…'
                        : thread
                          ? incognito
                            ? 'Ask in this side thread, incognito…'
                            : 'Ask in this side thread…'
                          : incognito
                            ? 'Ask conductor, incognito…'
                            : 'Ask conductor…'
                    }
                    aria-label="Message conductor"
                    rows={1}
                    disabled={sending || threads.closing}
                  />
                  <MicButton onTranscript={onTranscript} disabled={sending || threads.closing} bargeIn={bargeIn} />
                  <ToggleMenu
                    label="Composer options"
                    marked={incognito}
                    items={[
                      {
                        id: 'incognito',
                        label: 'Incognito',
                        checked: incognito,
                        onToggle: toggleIncognito,
                        icon: <EyeOff size={16} aria-hidden="true" />,
                        hint: incognito ? 'Incognito on: this turn’s words won’t be kept' : 'Incognito: don’t keep this turn’s words',
                      },
                      {
                        id: 'voice-output',
                        label: 'Spoken replies',
                        checked: voiceOutput,
                        onToggle: toggleVoiceOutput,
                        icon: voiceOutput ? <Volume2 size={16} aria-hidden="true" /> : <VolumeX size={16} aria-hidden="true" />,
                        hint: voiceOutput ? 'Spoken replies on' : 'Spoken replies off',
                      },
                      {
                        id: 'barge-in',
                        label: 'Talk over replies',
                        checked: bargeIn,
                        onToggle: toggleBargeIn,
                        icon: bargeIn ? <Ear size={16} aria-hidden="true" /> : <EarOff size={16} aria-hidden="true" />,
                        hint: bargeIn ? 'Talking over a spoken reply stops it' : 'Spoken replies play to the end',
                      },
                    ]}
                  />
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
              )}
            </>
          )}
        </section>
      </div>
    </main>
  )
}
