// vendored voice module — canonical: chess/frontend/src/MicButton.tsx (agent-standard/voice.md); re-copy, never edit
import { useEffect, useRef, useState } from 'react'
import { transcribe } from './api'
import { audioIdle, unlockAudio } from './tts'
import { createVad, type Vad } from './vad'
import { ListeningIcon, MicIcon } from './icons'
import { encodeWav } from './wav'

export interface MicButtonProps {
  /** Receives the recognized text — the caller sends it down the same
   * pipeline as a typed command. Hands-free mode awaits the returned promise
   * (the agent turn) before listening again. `interactionId` is whatever
   * `beginUtterance` returned for this utterance, if anything. */
  onTranscript: (text: string, interactionId?: string) => void | Promise<void>
  /** Optional: called the moment an utterance ends (the VAD's end of speech,
   * or the push-to-talk stop) — the start of the player's wait. Whatever id it
   * returns rides the transcription request as `X-Interaction-Id` and comes
   * back through `onTranscript`, so a host can join its latency records. */
  beginUtterance?: () => string | undefined
  /** Starting a conversation is pointless while the agent is busy — lock the
   * button. An already-running conversation stays escapable. */
  disabled: boolean
}

type MicState =
  // Hands-free conversation mode (the normal path).
  | 'idle'
  | 'starting' // VAD model loading after the tap
  | 'listening' // waiting for speech
  | 'capturing' // user is talking
  | 'working' // transcribing / agent thinking / reply playing
  // Push-to-talk fallback when the VAD can't load.
  | 'recording'
  | 'transcribing'

const LABELS: Record<MicState, string> = {
  idle: 'Start voice conversation',
  starting: 'Stop listening',
  listening: 'Stop listening',
  capturing: 'Stop listening',
  working: 'Stop listening',
  recording: 'Stop recording',
  transcribing: 'Transcribing',
}

// Inline icons, not emoji: iOS renders an emoji-presentation character in
// Apple Color Emoji whatever the surrounding style says, which put a color
// mic in an otherwise monochrome row. The typographic states ('…', '■') are
// text glyphs already and stay as they are.
const ICONS: Record<MicState, React.ReactNode> = {
  idle: <MicIcon />,
  starting: '…',
  listening: <ListeningIcon />,
  capturing: <ListeningIcon />,
  working: '…',
  recording: '■',
  transcribing: '…',
}

/**
 * Hands-free voice conversation: one tap opens a session where the VAD
 * detects each utterance's end, the clip is transcribed and sent down the
 * same pipeline as a typed command, and — once the agent's spoken reply has
 * finished — the mic reopens for the next utterance. Half-duplex by
 * construction: the mic is never open while the agent is thinking or
 * talking, so the agent cannot hear itself. A second tap ends the session.
 *
 * When the VAD can't load, degrades to classic push-to-talk (tap to record,
 * tap to send). Renders nothing in browsers without MediaRecorder /
 * getUserMedia — voice is an enhancement, the text box always works.
 */
export function MicButton({ onTranscript, beginUtterance, disabled }: MicButtonProps) {
  const [micState, setMicState] = useState<MicState>('idle')
  const [error, setError] = useState<string | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  // Held so teardown can release the mic: onstop stops the tracks on the
  // normal record → send path, but teardown never lets onstop run.
  const streamRef = useRef<MediaStream | null>(null)
  const vadRef = useRef<Vad | null>(null)
  // Conversation sessions are numbered; async continuations belonging to an
  // exited session must not touch state or reopen the mic.
  const sessionRef = useRef(0)
  // An utterance is already being processed — drop any VAD event that slips
  // through before pause() takes effect.
  const busyRef = useRef(false)

  /**
   * Abandon any push-to-talk recording and release the microphone. Handlers
   * are cleared first: this is the teardown path, so the half-captured clip
   * must not reach transcription or setState on the way out.
   */
  function discardRecording() {
    const recorder = recorderRef.current
    if (recorder) {
      recorder.ondataavailable = null
      recorder.onstop = null
      if (recorder.state !== 'inactive') recorder.stop()
    }
    streamRef.current?.getTracks().forEach((t) => t.stop())
    recorderRef.current = null
    streamRef.current = null
  }

  useEffect(
    () => () => {
      sessionRef.current++
      vadRef.current?.destroy()
      discardRecording()
    },
    [],
  )

  const supported =
    typeof MediaRecorder !== 'undefined' &&
    typeof navigator.mediaDevices?.getUserMedia === 'function'
  if (!supported) return null

  function endConversation() {
    sessionRef.current++
    vadRef.current?.destroy()
    vadRef.current = null
    setMicState('idle')
  }

  async function handleUtterance(audio: Float32Array, session: number) {
    if (busyRef.current) return
    busyRef.current = true
    const interactionId = beginUtterance?.()
    try {
      vadRef.current?.pause()
      setMicState('working')
      const text = await transcribe(encodeWav(audio), 'clip.wav', interactionId)
      if (sessionRef.current !== session) return
      if (text === null) {
        // The speech service is down; auto-resuming would hammer it forever.
        endConversation()
        setError('Voice input is unavailable.')
        return
      }
      if (text.trim()) {
        // The agent turn, then its spoken reply — only when both are done is
        // it safe to listen again (half-duplex).
        await onTranscript(text.trim(), interactionId)
        await audioIdle()
        if (sessionRef.current !== session) return
      }
      // Empty transcript: a VAD misfire, not worth nagging about.
      vadRef.current?.resume()
      setMicState('listening')
    } catch {
      // Nothing above is meant to reject — the api client resolves its failures
      // to null and tts.ts never rejects — so reaching here means the turn died
      // somewhere unaccounted for. That is exactly the case that must not be
      // resumed: an escape past the two lines above leaves the mic showing
      // 'working' with the VAD paused and nothing left to reopen it, which on
      // a phone is a session that has silently stopped listening (#232). End
      // it like the unavailable branch — visibly, one tap from the way back.
      if (sessionRef.current !== session) return
      endConversation()
      setError('Voice conversation stopped — tap to start again.')
    } finally {
      busyRef.current = false
    }
  }

  async function startConversation() {
    const session = ++sessionRef.current
    setError(null)
    setMicState('starting')
    const vad = await createVad({
      onSpeechStart: () => {
        if (sessionRef.current === session && !busyRef.current) setMicState('capturing')
      },
      onSpeechEnd: (audio) => {
        if (sessionRef.current === session) void handleUtterance(audio, session)
      },
      onUnavailable: (reason) => {
        // Surfaced, not swallowed: on a phone this line is the only
        // diagnostic anyone will ever see.
        setError(`Hands-free unavailable (${reason}) — tap-to-talk instead.`)
      },
    })
    if (sessionRef.current !== session) {
      // The user tapped out (or the component unmounted) while loading.
      vad?.destroy()
      return
    }
    if (!vad) {
      // No worklet/WASM support or the mic was refused to the VAD — degrade
      // to classic push-to-talk.
      await startRecording(session)
      return
    }
    vadRef.current = vad
    setMicState('listening')
  }

  async function startRecording(session: number) {
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      if (sessionRef.current !== session) return
      setError('Microphone unavailable — check browser permissions.')
      setMicState('idle')
      return
    }
    if (sessionRef.current !== session) {
      // The user tapped out (or the component unmounted) while the permission
      // prompt was up. The grant still arrived — release it rather than open
      // a mic nobody asked for and nothing is left to close.
      stream.getTracks().forEach((t) => t.stop())
      return
    }
    const recorder = new MediaRecorder(stream)
    const chunks: Blob[] = []
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data)
    }
    recorder.onstop = async () => {
      const interactionId = beginUtterance?.()
      // Release the mic as soon as the clip is captured; transcription is
      // backend work.
      stream.getTracks().forEach((t) => t.stop())
      recorderRef.current = null
      streamRef.current = null
      setMicState('transcribing')
      const clip = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' })
      const text = await transcribe(clip, undefined, interactionId)
      if (sessionRef.current !== session) return
      setMicState('idle')
      if (text === null) {
        setError('Voice input is unavailable.')
      } else if (text.trim()) {
        void onTranscript(text.trim(), interactionId)
      } else {
        setError("Didn't catch that — try again.")
      }
    }
    recorderRef.current = recorder
    streamRef.current = stream
    recorder.start()
    setMicState('recording')
  }

  function toggle() {
    // Mobile browsers only allow audio primed inside a user gesture; a mic
    // tap is the last gesture before the agent's spoken reply.
    unlockAudio()
    if (micState === 'idle') void startConversation()
    else if (micState === 'recording') recorderRef.current?.stop()
    else if (micState !== 'transcribing') endConversation()
  }

  return (
    <>
      <button
        type="button"
        className={`mic-button mic-${micState}`}
        aria-label={LABELS[micState]}
        title={LABELS[micState]}
        onClick={toggle}
        disabled={micState === 'transcribing' || (disabled && micState === 'idle')}
      >
        {ICONS[micState]}
      </button>
      {error && (
        <p className="mic-error" role="alert">
          {error}
        </p>
      )}
    </>
  )
}
