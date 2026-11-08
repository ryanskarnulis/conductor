# Narration

Narration is what Conductor says back: the one-line acknowledgement of a plan (`.cd-ack`, in `glow`) and the spoken reply built from receipts (`.cd-narration`).

## Use
The acknowledgement appears with the plan. The narration appears when Conductor speaks, below the receipts it is built from.

## The consumer provides
The narrator's text. Its facts come from the receipts (titles, counts, undo); the code guarantees them, the persona only phrases them.

## Rules
- At most two sentences on voice.
- Names what was done, by title. Quoted outside text stays quoted.
- Label it `Conductor`.
