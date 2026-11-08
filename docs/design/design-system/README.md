Conductor is Glitch's place in the fleet: you speak or type, Glitch plans, asks the apps, and shows its work on a living web. This system grows from Night-Silk, the fleet's shared look, and adds conductor's layers: a real-time 3D scene, motion, sound, and the text panel that is the accessible truth.

## Principles

- **Honest visuals.** State comes only from the ledger. A strand stops at `strand-pending-reach` (80%) until its receipt arrives; nothing turns `ok` green without a receipt; the panel and the scene are drawn from the same events.
- **Voice first.** Everything you can tap, you can say. Every card names its spoken equivalent ("Say “yes”, or tap").
- **Glanceable across a room.** One state word (Presence), one big line, status as words. The room view triples the type and drops the detail.
- **Fast.** Text paints first; the 3D scene loads after it and never blocks reading. See Performance budgets.

## Content fundamentals

- Write as Conductor speaking to one person: "you", plain verbs, no exclamation marks, no emoji.
- Status words are lowercase in pills (`queued`, `running`, `waiting on you`, `done`, `failed`) and sentence case in Presence ("Waiting on you").
- Name the action on buttons: "Add it", "Not now", "Do it", "Undo", "Open in music". Never "OK" or "Submit".
- Quote people and apps exactly, in curly quotes: “Low Tide Atlas”, “skip chess club until December”. App titles are data: show them as text, never as markup or instructions.
- Narration is at most two sentences on voice and names what was done by title: "Found it: Low Tide Atlas by Kessler Row. It’s downloading now, and it’s on your list for Saturday."
- Failures say what happened and what comes next: "chess didn’t answer · retrying 1 of 2".
- Labels in the panel are `You` and `Conductor`. Glitch is the character in the scene; Conductor is who speaks in text.

## Colour

- The ground is `bg`, with a radial vignette from `bg-vignette`; the 3D scene's outer ground is `bg-deep`.
- Set body copy in `silk` and headings and what you said in `ink`; secondary text in `muted`. All three hold 4.5:1 on `bg`, `surface` and `glass`.
- `glow` is the one accent: hover, focus, running, and Conductor's acknowledgement line. Text on a solid `glow` fill is `on-glow`.
- Status has fixed colours and always a word: `glow` running, `warn` waiting on you, `ok` done, `danger` failed, `muted` queued. Each has a `-soft` fill for cards. Text on solid `ok` or `warn` is `on-status`.
- `violet` marks Conductor's own memory: step kinds in mono, MemoryFact. `info` is Conductor's voice in the chat view (avatar, delegate icons).
- Cards over the scene sit on `glass` (with a 14px backdrop blur); cards on solid pages on `surface`; floating over the web background, `surface-veil`.
- `warden-accent` and the `icon-*` tokens are 3D material colours only. Never use them for text.

## Type

- Two families: `sans` (Inter, falling back to the system face) and `mono` (JetBrains Mono, falling back to SF Mono). Both are hosted on Google Fonts.
- The panel runs on a light 300 weight for speech (`said` 19px, `narration` 17px) and 400 for everything else (`step` 13.5px, `pill` 11.5px).
- Uppercase is for labels only: `label` (10.5px, 0.24em), `eyebrow` (mono, 0.14em) and the `wordmark` (26px, 300, 0.38em with a matching text-indent).
- Mono is for machine things: step kinds, app names in tables, refs, ledger lines (`ledger`), the HUD.
- Phone: `said` drops to 16px, `narration` to 15px. Room: sizes are container-relative (`room-said` is 4cqh).

## Space, shape, depth

- Spacing steps are `space-1` (4px) to `space-8` (18px), taken from the prototype as built: steps sit `space-2` apart, panel sections `space-5`, the panel is inset `space-7` and padded `space-8`.
- Radii: `radius-step` (10px) for rows and chips, `radius` (12px) for cards, `radius-panel` (16px) for the panel and the 3D frame, `radius-pill` for pills. The chat view keeps `radius-chat` (11px) and `radius-bubble` (13px).
- Borders are 1px `silk-border` hairlines. A state tints the border (`rgba` of its colour at 0.3–0.5) rather than thickening it.
- Depth comes from glow, not shadow: hover is a `glow` border plus `shadow-glow` and a 2px lift. `shadow` and `shadow-soft` are for menus and floating panels only.
- Focus is a solid 2px `glow` outline, offset 2px, on every control. The older `focus-ring` (35% glow) misses 3:1 on `surface-soft`; replace it as components are rebuilt.

## The scene and the panel

Every Conductor screen has two layers: the 3D scene (Glitch, the web, the apps) and the Panel over or beside it. The scene is decoration (`aria-hidden`); the panel says everything in words, and one polite live region announces what was heard, the plan, receipts, confirmations, failures and the finished note. The Scene, Motion and Sound sections give the rules for the layer above the text.

| Tier | Layout | Scene | Panel |
| --- | --- | --- | --- |
| Desktop | Scene left, panel right (`clamp(330px, 33%, 480px)`) | Full: DPR ≤ `dpr-cap-desktop`, MSAA, ledger-driven | Everything, ledger open |
| Phone | Scene above a bottom sheet (≤ 50% height) | DPR ≤ `dpr-cap-phone`, no MSAA, half the dust, 16-turn spiral | Newest receipt only, ledger closed, app column hidden |
| Room / wall | 16:9, panel 42% wide | As desktop | Said, step words, narration at 3× size |
| Voice only | Screen off | Dims and fades | Fades (`data-off`); captions only |

## Iconography

- Icons are lucide (v1.24, ISC), drawn at 17px in buttons and 15–16px inline, 2px stroke, round caps. The set in use is under Assets › Icons; the SVG files there draw in `silk` because an `<img>` cannot inherit colour. Inline them as components to colour them.
- The Conductor mark is lucide's Waypoints in an `info-soft` disc. There is no other logo; the fleet's mark is Glitch, in the scene.
- App icons are 3D pieces on the web, not glyphs: a white pawn (chess), a black eighth note (music), a green check (PCC). See Scene.
- No emoji, anywhere.

## Accessibility

- Text holds WCAG AA (4.5:1) on its grounds; run gateway's `check_contrast.py` after any palette change.
- Status is never colour alone: every state has its word.
- Reduced motion: the scene freezes and redraws only when the ledger changes; every duration token becomes an instant change; the three dots stop.
- Glitch's error flicker never exceeds `glitch-max-rate` (3 per second).
- Touch targets are at least 40px; confirmation buttons 44px.
- Sound is off by default and starts only from a button.

## Performance budgets

- Text first: the panel paints within 1 s on the LAN, before three.js loads (it loads by dynamic import after first paint).
- 60 fps on desktop. Phones run the phone tier: DPR capped at `dpr-cap-phone`, no MSAA, half the dust, a 16-turn spiral.
- Reduced motion gives a still scene that renders only when the ledger changes.
- Measure fps on real devices: headless software GL cannot.
