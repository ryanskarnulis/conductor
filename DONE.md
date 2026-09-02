# DONE

Shipped work, newest first, one line each. Full narratives are in git history
(they lived inline in `TODO.md` before 2026-09-01).

- [x] 2026-08-24 — Sort panel first-use fixes: voice answers keep the panel open; opened groups stay open until empty
- [x] 2026-08-24 — The sort panel (#30): live worklist, folder buttons, one-at-a-time multi-select, note turn per filing
- [x] 2026-08-24 — Note turns (#29): `POST …/notes`, no model run; delegate records gained `app_tools`
- [x] 2026-08-24 — Fleet action proxy (#28): `/api/fleet/{app}/actions/{path}` to declared prefixes only
- [x] 2026-08-22 — Re-baselined routing evals 18/18 with music as the third fleet agent
- [x] 2026-07-13 — Handoff apps (#16): `open_<app>` tools carry the user + intent; chess goldens became handoffs (15/15 ×3)
- [x] 2026-07-12 — sync-check CI warning for vendored files (#15)
- [x] 2026-07-12 — Request-timeout chain aligned: nginx 330s above every 300s budget (#13)
- [x] 2026-07-12 — Deployed provider config fixed: `LLAMACPP_BASE_URL` via host-gateway (#10)
- [x] 2026-07-12 — Voice (#8–#10, VOICE-PLAN Phase 4): `/api/voice` proxies, vendored chess voice modules, spoken slow-turn ack
- [x] 2026-07-11/12 — Phase 3 complete (PRs #1–#6): engine (registry, provider, bounded loop, layered Glitch), `fleet/` (manifest discovery, delegate client, ask_<app> tools, MCP server), persistence + chat UI, routing evals GO 12/12 (the confirm rule rewritten to operational voice after the harness caught unconfirmed delegation on run one)
