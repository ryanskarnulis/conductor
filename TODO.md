# TODO

The backlog, in priority order. One task = one branch = one PR; finished work
moves to `DONE.md` with the date.

## Next

- [ ] **Decide the `refuse-playback` red** — broken by music #9's manifest
      widening, deterministic 0/3 (`docs/agent-evals.md`). Options: narrow the
      negative sentence in music's `agent.description` until both it and
      `music-save` hold; retire the golden (music's own agent answers
      truthfully one hop later); or leave red until music Phase 6 (Sonos)
      makes the route correct. Needs a few live iterations either way.
- [ ] **Multi-turn eval goldens** [M] — all goldens are single-turn. Two gaps:
      confirmation follow-through (after the user's "yes", conductor must
      actually delegate) and cross-turn subagent-thread reuse through the real
      HTTP seam (`delegate_threads` row reused, not recreated).
- [ ] **Sort the real library once** [S] — the sort panel chain is verified
      against a 60-file copy; do one real pass before calling music Phase 2.6
      done.

## Someday / blocked

- [ ] **SSE (or any multi-worker-safe) live-progress channel** [L] — v1's
      polling registry pins deployment to a single uvicorn worker; pick up if
      turn-latency UX hurts (master plan §10 defers it fleet-wide).
- [ ] **Subagent-thread cascade delete** — decided not-in-v1 (orphaned threads
      are inert; 404→recreate-once covers pruning). Either wire
      `DelegateClient.delete_conversation` as best-effort cleanup on master
      delete, or remove the dead code.
- [ ] **Hub-app control tools** — blocked on `../future-plans/hub-app.md`
      shipping; wraps the hub lifecycle API as local tools.

Standing constraint: the delegate contract must never assume a text-only
client (voice convergence is a master-plan placeholder).
