# Scene

The 3D layer: Glitch at the hub of a web, the apps moored at its frame. Built with three.js (r170, loaded as a module after first paint). It is decoration: `aria-hidden`, never the only place a fact appears.

## Glitch (Warden)

- Warden is Glitch's form: a procedural spider with a matte carbon-twill body, flush decal eyes and circuit traces in `warden-accent`. The model lives in gateway's theme (`createGlitch`), so the launcher and the apps share one Glitch.
- He stays at the hub. He turns to face the app he is working with and spins a strand to it; his feet grip the nearest spoke.
- His poses follow the ledger: idle at the hub, listening (rings pulse out with your voice), thinking (eyes half-lidded, motes orbit), spinning (a strand to an app), waiting (amber), humming (a job), speaking (the web trembles with the voice), error (a glitch, under `glitch-max-rate`). Clicking him makes him hop.
- Recall is his own work: he holds still at the hub and light runs inward through the web to him. No strand.

## The web

- A seeded orb web, identical on every load: a frame moored at the three app anchors (`web-radius` 4.3 from the hub), `web-spokes` (34) irregular spokes, a dense hub, a free zone, and a capture spiral of `web-spiral-turns` (24; 16 on phones) whose gaps widen outward. A few capture threads are broken; dew beads glint as you move.
- Mooring lines run off the frame and fade out before the edge. Nothing in the web is clipped by the frame of the screen.
- Silk is camera-facing ribbon strips with additive colour; strands are Verlet ropes that sag and ring.

## Strands: honest visuals

A strand means a call to another app. Only a receipt event draws its last 20%.

| Ledger | Strand | Anchor |
| --- | --- | --- |
| queued | dotted, dim, reaches 80% | dim ring |
| running | dashes flow out, stop at 80% | `glow`, pulsing |
| job | hums and trembles, open at 80% | progress ring |
| waiting on you | amber dashes | amber pulse; Warden turns amber |
| done | last 20% draws, the receipt runs home | ring turns `ok` |
| failed | frays: a gap near the end, slack | `danger`; Warden glitches |
| cancelled | slackens and draws back to the hub | back to dim (not in the prototype yet) |
| recall | none: Conductor's own record | light runs inward to Warden; green when found |

## App anchors

- Each app is moored at a frame corner: PCC at the top, chess low left, music low right.
- Each anchor holds a solid 3D icon at `icon-scale`, z 0.24 in front of the web so no thread shows through: a white pawn (`icon-pawn`, black outline) for chess, a black eighth note (`icon-note`, white outline) for music, a green check (`icon-check`, glowing `icon-check-glow`, black outline) for PCC. Outlines are inverted hulls, 0.0075 units.
- A ring around the icon carries the state colour. The icon turns slowly, tipped toward the camera; still under reduced motion.
- A new app gets a new frame corner and its own simple, solid object. Never an orb, never a logo.

## Light, camera, post

- HDR render, then bloom and an ACES tone curve. Lights write alpha 0.5 to skip the tone curve.
- Camera: `camera-fov` 32°, looking slightly up at the web, framed beside the panel (desktop, room) or above the sheet (phone) with a view offset. Fog density scales with camera distance.
- Parallax follows the cursor, or the phone's tilt. The camera eases in slightly while listening and speaking.
- The glitch effect is a post uniform: chromatic split and slice offset, under `glitch-max-rate`. Voice only fades the whole frame.
