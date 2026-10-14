// Conductor-owned icons for the vendored MicButton (which imports './icons').
// Not vendored itself: this is the app-side seam the standard allows —
// restyling and wiring live outside the verbatim copies. The mic draws in
// conductor's own lucide set, at the 17px the voice toggle and send button use.
import { Ear, Mic } from 'lucide-react'

export function MicIcon() {
  return <Mic size={17} aria-hidden="true" />
}

/** Listening (and capturing): the ear the button used to show as an emoji. */
export function ListeningIcon() {
  return <Ear size={17} aria-hidden="true" />
}
