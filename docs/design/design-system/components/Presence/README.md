# Presence

Presence says, in one uppercase word or phrase, what Glitch is doing now: the DOM twin of Warden's pose in the scene.

## Use
At the top of a compact panel, in the room view, and anywhere the scene is hidden. `data-state` sets the colour: `idle` (`muted`), `listen`, `think`, `work`, `speak` (`glow`), `wait` (`warn`), `failed` (`danger`), `done` (`ok`).

## The consumer provides
The state and its words: Ready, Listening, Planning, Working · step N of M, Waiting on you, Speaking, Trying again, Done.

## Rules
- The words carry the meaning; the dot only repeats it.
- It follows the ledger, never a guess: Working only after a step started, Done only after its receipt.
