# Composer

Composer is where you type or talk: a text field, the mic, the read-aloud toggle and Send, on the frosted strip.

## Use
At the bottom of the chat view, and of the panel when it takes input. The mic shows `mic` at rest and `ear` with a solid `glow` outline while listening. The read-aloud toggle shows `volume-2` on and `volume-x` off (55% opacity). Send is `send-horizontal` on an `info-soft` fill, the chat view's primary.

## The consumer provides
The send handler, the mic session state (`data-listening` while listening or capturing), and the voice-output toggle (`aria-pressed`).

## Rules
- Every action has a voice equivalent; the composer never offers something you cannot say.
- `radius-chat` on the field and buttons; 42px targets.
- The field's focus ring is solid `glow` (the older 35% ring misses 3:1).
