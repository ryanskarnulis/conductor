# Panel

The Panel is the glass text layer beside or over the 3D scene, and the accessible truth: everything the scene shows is written here as words.

## Use
- Desktop: a column on the right, `space-7` from the scene edges, `clamp(330px, 33%, 480px)` wide, `space-8` padding, `radius-panel`.
- Phone: `.cd-panel--sheet`, a bottom sheet `10px` from the edges, at most half the height, 22px radius. It scrolls to the newest card.
- Room / wall: the panel shows only what you said, the step words and the narration, at about three times the size (`room-said`).
- Voice only: set `data-off` and it fades out over `dur-panel-fade`; the scene dims with it.

## The consumer provides
Its children, in this order: Utterance, the acknowledgement (`.cd-ack`), PlanSteps, a ConfirmationCard or QuestionCard, ReceiptChips, a JobCard, a FinishedNote, Narration, and the ledger (TimelineItems) at the bottom.

## Rules
- Background is always `glass` with a 14px backdrop blur. Never opaque, never lighter.
- The 3D canvas behind it is `aria-hidden`; one polite live region announces what was heard, the plan, receipts, confirmations, failures and the finished note.
