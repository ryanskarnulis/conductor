# PlanStep

PlanStep is one row of the compiled plan: its kind, the app, what it does, and its StatusPill.

## Use
The plan appears as a list (`.cd-steps`) the moment it compiles, every step `queued`, then each row follows the ledger. One row per step; a strand in the scene matches each row whose app is not `conductor`.

## The consumer provides
`kind` (`recall`, `ask`, `look`), the app (`conductor` for recall, which is Conductor's own work), the step text, and its state.

## Rules
- Kind in mono `violet`; app in `muted`; text truncates on one line.
- Phone: the app column hides. Room: only kind, app and the pill, at three times the size.
