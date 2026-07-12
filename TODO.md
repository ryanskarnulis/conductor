# TODO

The living backlog, in priority order. One task = one branch = one PR,
squash-merged on green CI (see `CLAUDE.md` → Git workflow); re-plan freely
between slices. When an item ships, record the outcome inline — a bold note
with the date and PR number, the way `../agent-standard/AGENTS-MASTER-PLAN.md`
does. Leads carry a size (`[S]`/`[M]`/`[L]`) or a status tag (`[decision]`,
`[blocked]`).

## Next sprint (2026-07-12)

- [x] **[S] Fix the deployed provider config.** **Done 2026-07-12 (#10):
      `docker-compose.yml` now sets
      `LLAMACPP_BASE_URL: ${LLAMACPP_BASE_URL:-http://host.docker.internal:8200/v1}`
      (plus the speech-service URLs, same host-gateway mechanism), the deploy
      clone carries it, and a live chat turn through the deployed stack
      (`POST /api/agent/conversations/{id}/messages` via `127.0.0.1:8300`)
      round-tripped against llama-swap.** Original item: sets no
      `LLAMACPP_BASE_URL`, so the deployed backend falls back to the dev
      default `http://127.0.0.1:8200/v1` (`backend/app/config.py`) — loopback
      *inside the container*, where llama-swap isn't — while the compose file
      already plumbs `host.docker.internal` for the fleet upstreams. Mirror
      PCC's override
      (`LLAMACPP_BASE_URL: ${LLAMACPP_BASE_URL:-http://host.docker.internal:8200/v1}`,
      `../project-command-center/docker-compose.yml`), then verify a live chat
      turn from the deployed UI (`~/deploy/conductor`). This is
      `../agent-standard/NEW-APP-CHECKLIST.md` step 5, missed for the one
      service that isn't discovered from a manifest.
- [ ] **[S] Align the request-timeout chain.** `frontend/nginx.conf` caps
      `/api` at 200s (`proxy_read_timeout`/`proxy_send_timeout`) while the
      browser client waits 300s (`AGENT_RUN_TIMEOUT_MS`,
      `frontend/src/api/agent.ts`) and the backend's own budgets are 300s
      (the delegate read timeout in `fleet/delegate.py`,
      `llamacpp_timeout_seconds` in `config.py`) — so a long docker-deployed
      turn 504s at nginx while the backend keeps running and commits an
      assistant turn the browser never sees. The nginx comment claims the
      opposite ("give nginx headroom past the client's own timeout"). Pick one
      coherent story — e.g. nginx at 330s, above every 300s budget — and fix
      the wrong comment.
- [ ] **[M] Multi-turn eval goldens.** All 12 goldens in
      `backend/tests/test_agent_evals.py` are single-turn. Two gaps: (a)
      confirmation follow-through — `confirm-reset`/`confirm-resign` prove
      conductor withholds the tool call, but nothing proves it actually
      delegates after the user says yes, the second half of its
      one-safety-stop rule; (b) cross-turn subagent-thread reuse through the
      real HTTP seam — a follow-up to the same app must reuse its
      `delegate_threads` row, not open a fresh subagent thread. Extend the
      recorded baseline in `docs/agent-evals.md` to cover both (the baseline
      gates every prompt / manifest-examples / model / loop change).

## Backlog

- [ ] **[L] SSE — or any multi-worker-safe — live-progress channel.** v1's
      polling channel (`GET …/activity` over the in-memory `turn_activity`
      registry) is an explicit stopgap ("no SSE in v1") and pins the
      deployment to a single uvicorn worker (`backend/Dockerfile` CMD comment;
      the in-process rate limiter shares the constraint). The master plan
      (`../agent-standard/AGENTS-MASTER-PLAN.md` §10) defers SSE across the
      delegate contract and both chat UIs; pick it up if turn-latency UX
      still hurts.
- [ ] **[decision] Subagent-thread cascade delete — deliberately not
      implemented in v1** (2026-07-12). `DELETE /conversations/{id}`
      soft-deletes only the master row (`api/routes_agent.py`); the
      `delegate_threads` rows and the app-side subagent conversations stay.
      That's safe: orphaned threads are inert — apps soft-delete their own
      rows, and the 404→recreate-once rule covers a pruned thread. Keep the
      master plan's §5 lifecycle line ("deleting/resetting a conductor
      conversation deletes its subagent threads (best-effort)") in sync with
      this decision. `DelegateClient.delete_conversation` (`fleet/delegate.py`)
      is dead code today — either wire it as the optional best-effort cleanup
      on master-conversation delete, or remove it.
- [ ] **[blocked] Hub-app control tools** — wrap the hub's lifecycle API as
      local `start_app`/`stop_app`/`health` tools; the master plan Phase 3's
      one deliberately-open checkbox. Blocked on `../future-plans/hub-app.md`
      shipping (nothing checked out there yet). `app.yaml` already declares
      the compose `runtime:` block the hub will consume.

Standing constraint (not a task): the delegate contract must never assume a
text-only client. Voice convergence — chess's STT/TTS stack,
home-assistant-voice, and conductor meeting in one spoken house assistant — is
a master plan §10 placeholder, and contract changes here must not foreclose it.
