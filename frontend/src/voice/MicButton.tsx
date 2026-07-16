// vendored voice module — canonical: chess/frontend/src/MicButton.tsx (agent-standard/voice.md); re-copy, never edit
import { useEffect, useRef, useState } from 'react'
import { transcribe } from './api'
import { audioIdle, unlockAudio } from './tts'
import { createVad, type Vad } from './vad'
import { encodeWav } from './wav'

export interface MicButtonProps {
  /** Receives the recognized text — the caller sends it down the same
   * pipeline as a typed command. Hands-free mode awaits the returned promise
   * (the agent turn) before listening again. */
  onTranscript: (text: string) => void | Promise<void>
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

const ICONS: Record<MicState, string> = {
  idle: '🎤',
  starting: '…',
  listening: '👂',
  capturing: '👂',
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
export function MicButton({ onTranscript, disabled }: MicButtonProps) {
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
    try {
      vadRef.current?.pause()
      setMicState('working')
      const text = await transcribe(encodeWav(audio), 'clip.wav')
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
        await onTranscript(text.trim())
        await audioIdle()
        if (sessionRef.current !== session) return
      }
      // Empty transcript: a VAD misfire, not worth nagging about.
      vadRef.current?.resume()
      setMicState('listening')
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
      // Release the mic as soon as the clip is captured; transcription is
      // backend work.
      stream.getTracks().forEach((t) => t.stop())
      recorderRef.current = null
      streamRef.current = null
      setMicState('transcribing')
      const clip = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' })
      const text = await transcribe(clip)
      if (sessionRef.current !== session) return
      setMicState('idle')
      if (text === null) {
        setError('Voice input is unavailable.')
      } else if (text.trim()) {
        void onTranscript(text.trim())
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
