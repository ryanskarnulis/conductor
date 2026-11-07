/** Earcons (design system #86): quiet plucked strings, off by default.
 * Each app is a string (PCC D, music A, chess E); a receipt plucks its app's
 * string once; recall ends in a soft low tone; the mic opening rises and
 * closing falls; a confirmation sounds two soft plucks once. Nothing loops. */

const NOTE: Record<string, number> = { tasks: 293.66, music: 220, chess: 329.63 }
const STORAGE_KEY = 'conductor-sound'

let context: AudioContext | null = null

export function soundOn(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'on'
  } catch {
    return false
  }
}

export function setSound(on: boolean): void {
  try {
    localStorage.setItem(STORAGE_KEY, on ? 'on' : 'off')
  } catch {
    // Storage may be blocked; sound just stays off next time.
  }
  if (on) void audio()?.resume()
}

function audio(): AudioContext | null {
  if (typeof window === 'undefined' || typeof window.AudioContext !== 'function') return null
  context = context ?? new window.AudioContext()
  return context
}

function pluck(freq: number, when = 0, volume = 0.16, duration = 1.6): void {
  if (!soundOn()) return
  const ac = audio()
  if (!ac) return
  const t = ac.currentTime + when
  const tone = ac.createOscillator()
  const octave = ac.createOscillator()
  const gain = ac.createGain()
  const filter = ac.createBiquadFilter()
  tone.type = 'triangle'
  tone.frequency.value = freq
  octave.type = 'sine'
  octave.frequency.value = freq * 2.003
  filter.type = 'lowpass'
  filter.frequency.setValueAtTime(freq * 9, t)
  filter.frequency.exponentialRampToValueAtTime(freq * 1.5, t + duration)
  gain.gain.setValueAtTime(0, t)
  gain.gain.linearRampToValueAtTime(volume, t + 0.006)
  gain.gain.exponentialRampToValueAtTime(0.0001, t + duration)
  tone.connect(filter)
  octave.connect(filter)
  filter.connect(gain)
  gain.connect(ac.destination)
  tone.start(t)
  octave.start(t)
  tone.stop(t + duration)
  octave.stop(t + duration)
}

export const earcons = {
  receipt(app: string): void {
    const note = NOTE[app]
    if (note) pluck(note, 0.18)
  },
  recall(): void {
    pluck(146.83, 0.1, 0.07, 2.6)
    pluck(220, 0.16, 0.04, 2.2)
  },
  micOpen(): void {
    pluck(440, 0, 0.08, 0.5)
    pluck(660, 0.09, 0.08, 0.6)
  },
  micClose(): void {
    pluck(660, 0, 0.07, 0.5)
    pluck(440, 0.09, 0.07, 0.6)
  },
  confirmation(): void {
    pluck(392, 0, 0.08, 1.2)
    pluck(392, 0.32, 0.06, 1.2)
  },
}
