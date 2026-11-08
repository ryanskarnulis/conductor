# StatusPill

StatusPill writes a step's state as a word.

## States
`queued` (dashed, `muted`), `run` and `job` (`glow` on `glow-soft`), `wait` (`warn`), `done` (`on-status` on solid `ok`), `failed` (`danger`, followed by why), `cancelled` (`muted`, struck through).

## The consumer provides
`data-state` and the words: `running · retry 1 of 2`, `failed · chess didn’t answer`, `running · job`.

## Rules
- The word is required. Colour never carries a state alone.
- `done` appears only after a receipt. Nothing turns green on a guess.
