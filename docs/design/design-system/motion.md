# Motion

Motion shows cause and effect: a request goes out, a receipt comes back. It never decorates, and it never shows something the ledger has not said.

## Timings

Use the `duration` and `easing` tokens: `dur-receipt` (180ms) for a receipt's arrival, `dur-receipt-travel` (1000ms) for the receipt running home, `dur-queue-stagger` (300ms) between queued strands, `dur-wait-pulse` (1600ms) for one amber pulse, `dur-ring` (1100ms) for a ring from Glitch, `dur-panel-fade` (500ms) and `dur-scene-fade` (600ms) for voice only, `dur-hover` (250ms) for hover. Things arriving use `ease-out`; fades use `ease-in-out`.

## Choreography by state

- **Planned.** The plan compiles: every step appears `queued` in the panel at once; in the scene a dotted strand reaches out to each app, `dur-queue-stagger` apart.
- **Running.** Warden turns to the app. A bright pulse runs out along the strand in 900ms, then dashes keep flowing; the strand stops at 80%. The anchor ring pulses.
- **Waiting on you.** The strand's dashes turn amber and slow; the anchor pulses amber once every `dur-wait-pulse`; Warden turns amber. The ConfirmationCard appears under the plan. Two soft plucks sound, once.
- **Done.** The receipt arrives: the strand's last 20% draws in `dur-receipt`, a green packet runs home in `dur-receipt-travel`, a ring expands from Warden (`dur-ring`), the strand rings once. The ReceiptChip appears, the pill turns `done`.
- **Failed.** The strand frays (a gap near its end) and goes slack; Warden glitches for about 1.4 s at no more than `glitch-max-rate`; the pill says why. A retry starts a new running strand.
- **Cancelled.** The strand slackens and draws back to the hub; the anchor returns to dim; the pill says `cancelled`. Nothing is narrated. (Not in the prototype yet.)
- **Queued job.** The step detaches: the strand stays open at 80% and hums; the anchor shows a progress ring that fills with each progress event. When the job's receipt arrives it plays Done, and the FinishedNote appears.
- **Recall.** No strand. Warden holds still; light runs inward through the web to him; when the result is found, a green pulse lands on him.
- **Voice only.** The panel fades over `dur-panel-fade`, the scene dims over `dur-scene-fade`, and captions show what you hear.

## Reduced motion

Every transition is already finished: the scene renders a still frame and redraws only when the ledger changes. Pulses, tremble, dew glints, parallax, icon turns and the three dots stop. State still reads, by colour, shape and word.
