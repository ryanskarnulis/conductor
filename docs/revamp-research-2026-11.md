# Conductor revamp: product, research, options

2026-11-05. The research and options behind the revamp. **Decided the same
day: C and D combined** (compiled plans on a job ledger). The plan, phases,
spike results and child issues live in #56; this file is the reasoning, kept
as it was written before the decision.

**Decisions (2026-11-05), answering §10:**
1. Direction: D's compiler on C's job ledger (D's interpreter is the job
   runtime; D's journal is the ledger). C's free-roaming workers are dropped.
2. Priority order as in §1.3.
3. Escalation: local 26B first; cloud arms benched on synthetic data, live arm
   chosen by quality and terms (#57).
4. General knowledge is answered and labelled; anything current is declined.
5. Proactivity is its own later project, except a note when a job you started
   ends.
6. One timeline, side threads you start yourself, incognito, "forget that";
   quiet automatic topics for memory scoping only.
7. No app tools in conductor: a fixed set of step kinds with the app as a
   parameter, plus one glance endpoint per app for fast reads. App-panel
   activity is read live when needed; no feed.
8. Conductor stays on the 12B and accepts the swap after chess (S0: ~4 s).
9. Added: a code-built honest ack, barge-in and spoken corrections, situational
   context in the state block, memory you can see and edit, self-report from
   the ledger, and a persona in the narrator only.

## TL;DR

- **The product:** one endless conversation with one agent. It tells the
  truth about what happened, remembers across days without being managed,
  answers simple asks in a few seconds (by voice too), runs long jobs and
  reports back, and stays quiet unless asked. Fifteen scenarios below are
  the yardstick.
- **Four options, which differ in where control lives:**
  - **A. Switchboard, rebuilt.** Today's shape done right: the 12B loop
    delegates to app agents.
  - **B. One brain, app skills.** Conductor is the only agent, and apps ship
    skills.
  - **C. Talker and crew.** An async job runtime on a ledger, with side
    threads and proactivity.
  - **D. Intent compiler.** The 12B writes a small typed plan. Code checks it,
    runs it, journals it, and pauses it for confirmation. A narrator speaks
    from the receipts.
- **Recommendation: D.** Get there through foundations every option needs,
  and add C's job and event pieces as later phases. If spike D1 fails, fall
  back to C.
- **Findings you may not expect:**
  - Swapping to the 26B takes about 7 s, not 100 s, when the weights are in
    page cache, so a local escalation tier is realistic.
  - Conductor is the last agent in the fleet that still replays chat turns,
    which is the shape of the self-poisoning bug.
  - Today's conductor has two live honesty bugs on its failure paths (§2.2).
- **You need to answer eight questions (§10).**

---

## 1. The product: "the only agent I use"

### 1.1 Scenarios (the yardstick, and the seed of the new evals)

| # | You say | Done looks like | Stresses |
|---|---|---|---|
| S1 | "what's due today?" | an answer from PCC's data, ≤3 s warm | single-app read, speed |
| S2 | "add a task to water the plants tomorrow" | one task with the right date, and the reply names it | single-app write, honesty |
| S3 | "download the new Caribou album and add a task to sort it this weekend" | the album is queued in music, and a PCC task titled with the album's **real** name is due Sat/Sun | cross-app, data passed between apps |
| S4 | "grab this whole playlist <url>" … then 20 min later | "playlist's in, 38 of 40, two didn't match" arrives on its own, on the surface you're using | long-running job, reporting back |
| S5 | "what did we decide about the garage shelves last week?" | an answer that cites where it came from, or "I don't have that" | episodic memory across days and apps |
| S6 | Tue: "grab Odessa by Caribou". Fri: "put that Caribou song in the house folder" | Odessa is filed under house | reference resolution across days |
| S7 | kitchen, voice only: "push the dennis thing to friday" | the dentist task moves, with one spoken line and no screen | voice, an STT slip, no visible trajectory |
| S8 | "what's the weather tomorrow?" / "how long do I boil an egg?" | an honest "no app does weather"; general knowledge answered or declined per your policy (Q4) | out-of-fleet asks, capability gaps |
| S9 | "wipe the kitchen project" → "yes" | it asks once in the app's own words, acts only after the yes, and says what went to the trash | confirmation and follow-through (#37) |
| S10 | "sort my whole library" … "actually, stop" | it stops at the next boundary, says exactly what got filed, and offers undo | cancel partway through |
| S11 | after S3: "undo that" | the task is trashed, and it says plainly that the download can't be undone | undo across apps |
| S12 | "let's play chess"; later "how did my last game go?" | the board opens with the intent; the result is read back without opening the board | place apps (`open:`) vs reading from them |
| S13 | morning, unprompted, or "anything I should know?" | one short brief (3 overdue tasks, 5 liked songs to sort), and nothing else interrupts | proactivity without noise |
| S14 | "plan my week around the garage project" | a concrete plan as tasks and Focus entries, confirmed before anything is written | hard planning (the 12B's frontier `plan_week` is 0/10) |
| S15 | after S3: "do the same for their other album" | the same two steps for the new album | reusing a past action as a template |

### 1.2 Surfaces

| Surface | State | What it demands |
|---|---|---|
| Web chat (desktop, and phone over Tailscale) | live | trajectory and undo UI are fine here |
| Voice in the web UI (push-to-talk, hands-free) | live (`voice.md`) | a spoken ack on slow turns (exists: "Asking chess…") |
| Spoken house assistant: room satellites through HA Assist/Wyoming (`home-assistant-voice.md` Ph 3–4) | planned | audio only, no screen, wake word, several rooms. Replies must stand alone in one or two sentences, confirmations happen by voice, and an announcement needs a target room. The delegate contract must not assume text. |
| App panels you still use directly (PCC chat, chess board, the music sort panel) | live | conductor can't see what happened there unless apps publish activity (Q7) |
| Push notifications (ntfy is named in the HA plan) | none | job completions and the daily brief |
| MCP (Claude Code as a client) | live, stdio | keep it working |

### 1.3 What matters most (proposed order, Q2)

1. **Honest.** It never claims what didn't happen, and it says what did,
   including partial work and failures. (Hard constraint.)
2. **Continuity without context management.** One endless conversation.
   References across days work, there's no "new chat" ritual, and the cost
   of a turn doesn't grow with history.
3. **Fast.** A simple text ask takes ≤3 s warm. First audio comes ≤~4 s after
   you stop talking. Anything slower gets an immediate ack.
4. **In control.** It confirms before anything destructive, can be cancelled
   at any time, and can undo whatever is undoable.
5. **Broad.** New apps join without changes to conductor's code, and
   cross-app asks work.
6. **Quietly proactive.** A daily brief and the callbacks you asked for.
   Nothing else.

---

## 2. Where we stand

### 2.1 Today's conductor against the scenarios

| Scenario | Today | Why |
|---|---|---|
| S1, S2 | works (routing baseline 18/18; `music-save` 13/20 xfail #54) | — |
| S3, S15 | works only if the 12B chains two asks and copies the title right; no eval | free-text relay. #55: titles are dropped (kept in 4–31 of 40 runs) |
| S4 | no | synchronous request/response, with no job ids or callbacks |
| S5, S6 | no | memory is the text turns of the current conversation, and subagent threads are keyed per conversation |
| S7 | web voice works; satellites don't exist | voice is per-app and web-only |
| S8 | refuses everything outside the apps | by design ("no abilities of your own") |
| S9 | asks first (the operational-voice rule); the "yes" follow-through has no eval (#37) | confirmation lives in the prompt |
| S10 | no Stop in conductor, and app cancel isn't propagated | — |
| S11 | no | there's no record of what to invert |
| S12 | `open_chess` works; reading chess state doesn't | chess is `open:`-only |
| S13 | no | no events, no scheduler |
| S14 | no | the 12B's ceiling (PCC frontier `plan_week` 0/10, thinking on or off) |

### 2.2 Live bugs noticed along the way (filed in Part B as P1 `bug`, not fixed)

- **B1: work that happened is reported as "unavailable, try again."** PCC and
  music persist the partial trajectory and return 502/504/500 when a run dies
  after its tools ran (`project-command-center/backend/app/api/routes_agent.py`,
  the `AgentRunFailed` path). Conductor maps every 5xx and every read timeout
  to `DelegateUnavailable` (`fleet/delegate.py`), and the model is then told
  "Report it as unavailable; try again shortly" (`fleet/tools.py`
  `_to_tool_error`). The result: writes that happened get reported as not
  happening, and the suggested retry duplicates them. No `Idempotency-Key` is
  sent (chess supports one, and the contract doesn't define it).
- **B2: a write vanishes from conductor's transcript.**
  - If conductor's own provider fails after a delegate call succeeded, the
    route returns 502 and never stores the assistant turn (`api/routes_agent.py`).
  - If a limit stop comes after delegate calls ran, the reply is null, and
    `history_for_loop` skips content-less turns.
  - Either way, the next turn doesn't know an app write happened. The root
    cause is #52; in Part B I'll add the honesty consequence to #52 rather
    than file a duplicate.
- **Not bugs, but worth knowing:**
  - `STANDARD.md` §8 and `model-profile.md` say every agent shares
    gemma-4-12b, but chess runs gemma-4-26b-a4b (chess#433), so a conductor
    turn right after chess play pays a model swap.
  - `model-profile.md` has no latency envelope yet. It's an open item in the
    master plan, and §5 below has the numbers it would hold.
  - Music relays untrusted web text (YouTube titles, Genius lyrics,
    SoundCloud tags) into conductor's planner context, and that context has
    write tools available (§3.11).

### 2.3 Lessons the fleet already paid for

| Lesson | Evidence | What it means for conductor |
|---|---|---|
| Code owns actions; the model owns understanding and speech. Every fix that worked moved a decision into code | chess `agent-framework-research-2026-07-14.md`, chess CLAUDE.md | orchestration decisions that have a right answer (order, data passing, confirmation, cancel, undo) belong in code |
| Safety rules bind only in operational voice with a worked example; a structural gate binds always | conductor PR #6; chess `GATED_TOOLS` (origin- and version-bound); PCC #416 (prose rewordings flat or worse) | confirmation becomes a code gate with a token, with the prompt rule kept as a backstop |
| The 12B is schema-sensitive: minimizing chess's schemas collapsed `undo_and_replace`, and one PCC docstring line took `date_next_friday` from 20/20 to 0/20 | chess CLAUDE.md, PCC #466 | whatever catalog conductor shows must be small and fixed, with its bytes pinned and A/B'd |
| Gather before you plan | PCC #415: 237/245 → 239/240, iterations 3.36 → 2.67; chess #451 | search the user's words (ledger, entities) before the planner runs, not through a tool |
| Planner → narrator: the phase that speaks can't act | chess `long_capture[poisoned]` 1/5 → 5/5; PCC #418 (+0.6 s) | the narrator speaks from code-built receipts |
| History as recall, never as chat turns; agents self-poison from their own false replies | PCC poisoned family 2/15; chess #361/#372 | conductor still replays chat turns (`services/conversations.py`), so port PCC's `recall.py` |
| State blocks beat lookups; the binding limit is prefill time, not the window | PCC #412/#484 (`date_push` 0/10 → 10/10); a 24.5k-token history is ≈55 s | build a fresh state block every turn: open jobs, pending questions, recent entities |
| A delegate speaks for itself through a code-built handoff | PCC #482, music #45 | the apps already build a run record, so make it typed fields instead of prose |
| Relaying by paraphrase drops facts | #55 | the narrator gets entities as data, and the claim checks include "named what it touched" |
| Model knowledge lives in per-model, per-phase profiles; mismatched sampling looks like model failure | chess `model-profiles.md`; Qwen 15/18 → 18/18 on its own sampling | any escalation model gets its own profile and its own eval run |
| Sampled Wilson gate, paired A/B, frontier tier, trajectory walks | `STANDARD.md` §6, `evalstats.py` | session evals use the same gate |
| Tracing and context capture are the first tool you reach for | PCC #400/#401, chess traces | conductor has only `delegate_call` log lines, so it needs per-hop traces with ids |
| Measure speech accuracy offline; no speech guards | chess `speech-accuracy.md`; PCC `claims.py` (94–96% backed) | claim checks run over conductor's narrator, offline |
| The prompt cache is the latency lever: byte-identical prefixes and append-only tails. More than ~2 distinct prefixes evict each other (≈6 s vs 0.4 s per call) | PCC agent-design "probe gotcha"; chess #362 | count the distinct prompt prefixes per turn in every design |

---

## 3. Research digest (outside in)

Each topic gives what the field says (source, date; anything before 2025 is
flagged), then what we take from it. Full links are in Sources.

**3.1 Orchestration shapes.**
- Anthropic's research system (2025-06-13) is orchestrator-worker:
  - It beat a single agent by 90%.
  - It costs ~15× the tokens of chat, and token use explains ~80% of the
    variance.
  - It's a poor fit where agents must share context or depend on each other.
  - Its lead agent waits synchronously on its workers, which is a stated
    bottleneck.
- Cognition (2025-06-12): share full traces, and avoid parallel writers,
  because "actions carry implicit decisions."
- LangChain (2025-06-16) reconciles the two: reads parallelize, writes need
  one writer.
- OpenAI Agents SDK (accessed 2026-11-05): a **handoff** passes ownership of
  the answer, and **agents-as-tools** keeps it with the manager.
- Blackboard designs (arXiv 2507.01701, 2025-07; 2510.01285, 2025-09)
  improve discovery, but nobody owns the outcome.
- MAST (2025-03, v3 2025-10): multi-agent failures are mostly design
  failures (specification, inter-agent misalignment, verification), not model
  failures.
- Anthropic, "Building effective agents" (2024-12, *older*): use the simplest
  pattern that works.
- *Take:* one writer per request, parallel reads only, and the owner of each
  answer named. More agents means more tokens and more misalignment, and both
  cost more on one GPU.

**3.2 Plan-then-execute vs interleaved.**
- The design-patterns paper (arXiv 2506.08837, 2025-06) and CaMeL
  (2503.18813, 2025-03) fix the control flow from trusted input before any
  untrusted data is read.
- The code-then-execute variant generalizes plans into programs.
- The cost: no adapting mid-plan without a re-plan. Arguments can still be
  steered by data.
- *Take:* this is the strongest structural answer to injection, and it also
  cuts model calls per request.

**3.3 Long-running work and reporting back.**
- MCP 2025-11-25 added experimental **Tasks**: call now, fetch later, with a
  task id, status, result, cancel and a retention window. The draft spec
  moves it into an extension.
- A2A v1.0 (2026-03; Linux Foundation, now under the Agentic AI Foundation
  since 2026-08) has a task lifecycle (submitted → working → input-required →
  completed / failed / canceled), streaming, and push notifications.
- Claude Code runs subagents with their own context, in the background.
- *Take:* the vocabulary has converged (task id, status, cancel,
  input-required). Borrow it and skip the full A2A stack (cross-org auth,
  agent cards).

**3.4 "One endless conversation" products.**
- Claude:
  - Memory is one synthesized summary per project, with incognito chats and
    user edits (2025-09-11; Pro/Max 2025-10-23).
  - Chat search is RAG exposed as a tool call (2025-08).
  - Context editing clears stale tool results, and a memory tool stores files
    outside the context (2025-09-29).
- ChatGPT keeps saved memories plus "reference chat history" (2025-04-10).
- Anthropic's context-engineering post (2025-09-29):
  - Compaction should be tuned for recall first.
  - Clearing tool results is the cheapest compaction.
  - Sub-agents return 1–2k-token digests.
  - Prefer just-in-time retrieval by reference.
- Chroma "Context Rot" (2025-07-14): all 18 models tested degrade as the
  input grows, even on trivial tasks.
- *What makes context feel unlimited:* scoping (projects), a summary injected
  every turn, retrieval over past turns as a tool, and clearing old tool
  output.
- *What it costs:* background summarization and tokens.
- *What still breaks:*
  - Summaries drop details whose importance shows up later.
  - Stale or wrong memories persist.
  - Memory bleeds across topics (hence project scoping).
  - Vague references miss in retrieval.
- *Take:* keep the per-turn context bounded and the store unbounded. Nothing
  model-written goes on the hot path (chess #405 already found that a
  model-written story self-poisons and costs 3–5 s of GPU per turn).

**3.5 Memory.**
- Zep/Graphiti (2501.13956, 2025-01) is bi-temporal: facts get valid-from and
  invalid-at stamps and are invalidated, never deleted.
- Mem0 (2504.19413, 2025-04) extracts facts, then decides
  ADD/UPDATE/DELETE/NOOP against what's stored.
- Letta sleep-time compute (2504.13171, 2025-04) consolidates memory while
  idle.
- Working memory is the turn's context; episodic memory is what happened
  (our receipts); semantic memory is distilled facts (the
  `personal-second-brain.md` librarian).
- *Take:*
  - The journal of receipts is the episodic store, and it's exact.
  - Semantic facts are consolidated off the hot path while the GPU is idle
    (nights), with one writer and citations back to receipts.
  - Forgetting means invalidating and ranking down by recency, with a user
    "forget that."

**3.6 Delegation interfaces.**
- Anthropic: vague task specs make workers duplicate effort and misread the
  task. Workers can store outputs and hand back **references** instead of
  copies (2025-06-13 appendix).
- MCP 2025-06-18 added typed tool output (`outputSchema`,
  `structuredContent`). MCP 2025-03-26 added tool annotations (read-only,
  destructive, idempotent).
- *Take:* pass typed results by reference and compress them in code.
  Free-text relay is #55's failure mode.

**3.7 Small-model orchestration.**
- NVIDIA (2506.02153, 2025-06): use heterogeneous systems, with small models
  for routine calls and large ones only where needed.
- Home Assistant's "prefer handling commands locally" tries built-in
  sentence matching first and falls back to the LLM. User reports from
  2025–26 describe latency surprises and fallbacks that never happen.
- DeepMind Talker-Reasoner (2410.08328, 2024-10, *older*) and OpenAI's
  realtime "chat-supervisor" pattern (2025): a fast talker keeps the
  conversation going while a slower reasoner works.
- BFCL v4 (updated 2025-12-16) now scores multi-turn, memory and agentic
  tool use.
- Our own data: the 12B holds every gate near its ceiling, but it fails
  composite planning, self-poisons, and binds descriptive rules ~50% of the
  time.
- *Take:*
  - Narrow the model to understanding and let code scaffold the rest.
  - Constrained decoding (`json_schema`, which works per `model-profile.md`)
    removes format failures, which leaves decision failures.
  - Escalate composite planning only.
  - A pre-router may narrow the menu but must never decide the action
    (chess forbids literal parsers for language).

**3.8 Concurrency on one GPU.**
- llama-server slots share a unified KV pool and continuous batching merges
  decode steps. Adding slots raises aggregate throughput but slows each
  request.
- We measured 4 slots, with slot selection by prompt similarity
  (chess #362).
- *Take:* parallel **app** work (downloads, Stockfish, DB) is free.
  Parallel **model** work is a modest throughput gain at a latency cost. So
  side threads are realistic as async app work plus prioritized model calls.

**3.9 Proactivity.**
- ChatGPT Pulse (2025-09): once a day, 5–10 cards that expire after 24 h,
  steered by the user.
- LangChain ambient agents (2025-01): three patterns (notify, question,
  review) and an "agent inbox."
- Timing research ("Goldilocks window," 2504.09332, 2025-04; ProMemAssist,
  2025-07): when an interruption lands matters as much as what it says.
- *Take:*
  - Digests on a cadence, an inbox, and interrupts only on explicit opt-in.
  - A proactive rule is a stored plan on a schedule, not a model deciding to
    talk.

**3.10 Interop: MCP and A2A vs our `delegate-api.md`.**
- Today's contract is a stateful chat thread plus one synchronous message
  call, returning text and the tool-call trace.
- It has no task ids, progress, cancel, input-required, typed result,
  idempotency, or event push.
- *Take:*
  - Extend our contract with A2A's lifecycle words.
  - Use MCP's tool annotations and typed outputs as the shape for direct
    capabilities.
  - Move the apps' MCP servers from stdio to streamable HTTP if conductor
    calls tools directly.

**3.11 Safety in multi-agent systems.**
- Willison's "lethal trifecta" (2025-06-16): private data, untrusted content
  and a way to act, all in one context.
- Music puts untrusted web text into conductor's planner context, which has
  write tools (`ask_tasks`), and today's only barrier is a prompt rule.
- Confirmation should be owned by whoever owns the action: the app's
  structural gate, surfaced as a typed result.
- Budgets and cancel belong in code at step boundaries (PCC #324's pattern).
- *Take:*
  - Typed data channels.
  - Plan before data.
  - Any plan built after reading data is tainted, and its writes need
    confirmation.
  - Every hop is cancellable.

**3.12 Evaluating orchestrators.**
- τ²-bench (2506.07982, 2025-06): dual-control tasks and `pass^k`
  (succeed on all k runs).
- Who&When (2505.00212, ICML 2025): automatic failure attribution finds the
  right agent only 53.5% of the time and the right step 14.2%.
- Anthropic: start with ~20 real queries, and judge the end state, not the
  path.
- *Take:*
  - Session-level scenario evals check the end state per app plus claims.
  - Attribution comes **by construction**: typed per-hop records with step
    ids, so the hop that diverged can be read off the trace.

---

## 4. What conductor needs from the apps

| Need | PCC | music | chess | Contract today |
|---|---|---|---|---|
| Typed result with stable refs (app, kind, id, title) | code-built record, as text (#482) | code-built record, as text (#45) | `game_id`, `version` | `content` string only |
| Confirmation surfaced as data (+ token, + the app's own question) | prompt-level, conductor owns it | — | `GATED_TOOLS`, origin-bound | none |
| Idempotency on writes | — | — | `Idempotency-Key` | none |
| Cancel | `POST …/cancel` | `…/cancel` | (board-side) | none |
| Long jobs: id, status, progress | — | `/downloads/jobs` (REST) | progress over websocket | none |
| Events pushed out (job done, activity) | `activity_events` (pull) | — | websocket broadcast | none |
| Direct typed tools | MCP stdio | MCP stdio | MCP stdio | dev-only |
| Undo handle per write | trash/restore | undo filing (#74) | takeback | none |
| Destructive preview ("trash Kitchen and its 12 tasks?") | — | — | the confirm question | none |
| Fake world for conductor evals | fresh-DB fixtures | fake yt-dlp | `CHESSAPP_AGENT=off` | canned replies in conductor |

---

## 5. Hardware reality

| Fact | Number | Source |
|---|---|---|
| GPU | RTX 3060 12 GB, one resident model (llama-swap, exclusive) | `llama-swap/config.yaml` |
| 12B generation | ~112 tok/s shallow (~94 with MTP on tool calls), ~41 tok/s at 128k depth | config comments, PCC agent-design |
| 12B prefill | ~446–680 tok/s | same |
| Cached prefix vs full prefill | ~0.4 s vs ~6 s per call | PCC agent-design |
| Slots | 4, with a unified KV pool | chess `latency-measurement.md` |
| PCC delegated run | gate mean 1.45 s; frontier median 2.9 s, p90 6.0 s | PCC #415, #417 |
| Narrator call | +~0.6 s | PCC #418 |
| Voice | speech end → transcript p50 2.5 s; first audio p50 4.9 s | chess #340 |
| Swap 12B ↔ 26B | **~6–7 s with weights in page cache**; ~100 s from disk | config (measured 2026-10-08) |
| RAM | 31 GB; the weights are 6.7 GB (12B) + 15 GB (26B), so ~22 GB to keep both cached | measured today |
| 26B-A4B | ~60 tok/s gen, ~760 tok/s prefill. Thinking: frontier median 17.8 s, p90 37.1 s; `plan_week` 6/10 vs the 12B's 0/10; gate slightly worse than the 12B with thinking off | PCC #417 |
| Chess's brain | gemma-4-26b-a4b (chess#433) | config |

**Where the time goes.**
- A quick delegated ask is almost all model time (2–4 calls).
- A download, a Stockfish analysis or a likes sync is almost all app time
  (seconds to minutes).
- So async side threads are real: they wait on app work and need the GPU
  only in short bursts.
- **Parallel model calls** (llama-server batching 4 slots) buy modest
  throughput and slow every call, so a background job must yield to a
  foreground voice turn at call boundaries (spike C1).
- Apps call llama-swap directly, so a fleet-wide priority needs a small gate
  in front of it, or a llama-swap feature.

**Escalation tiers (Q3).**

| Tier | When | Latency | Cost | Privacy | If unavailable |
|---|---|---|---|---|---|
| 12B (default) | everything | ~1 s per call | 0 | local | — |
| Local 26B-A4B, thinking | asks flagged as hard plans; repeated validation failures | swap ~7 s + 15–40 s + swap back ~7 s. While it's resident, every 12B caller in the fleet waits or swaps | 0 | local | the 12B gives a best effort and says so, or declines |
| Cloud Claude Sonnet 5.5 | opt-in per ask; never on voice by default | ~3–15 s | ~$0.03 per escalation (8k in, 1.5k out at $2/$10 per MTok); ~$3/month at 3 a day | your words, recent records and task/song titles leave the machine. API traffic isn't used for training by default; retention follows Anthropic's commercial terms (verify) | local 26B, else decline |
| Cloud Claude Haiku 5.5 | cheap classification, second opinion | ~1–3 s | ~$0.002 per call ($0.10/$0.50 per MTok) | same | same |
| Gemini free tier | **rejected** | — | free | free-tier inputs are used to improve Google's products and may be human-reviewed | — |

- Expected frequency: ≤1–3 escalations a day (<5% of turns), judging by how
  rarely real asks look like PCC's frontier composites.
- Prices are from the Anthropic pricing table cached 2026-10-06. Verify
  before relying on them.
- Any cloud tier needs two written rules amended: `STANDARD.md` §8 ("no
  cloud model providers") and the cloud line in PCC's "Do not build yet"
  list.

---

## 6. Foundations every option needs

These land first whichever option you pick. They fix §2.2 and close gaps
without choosing a shape.

- **F1. Typed receipts on the delegate hop.**
  - Add the app's code-built record as a structured field beside `content`:
    done, refused, looked up or not run, plus entity refs, an undo handle and
    a job ref.
  - The change to `delegate-api.md` is additive. It fixes the #55 class.
- **F2. Recall and state block in conductor.** Port PCC's `recall.py` and
  `state_block.py`, and stop replaying chat turns.
- **F3. Planner/narrator split.** The narrator reads receipts and can't act.
- **F4. Honest failure paths.**
  - An app's 5xx error body carries its partial record, and conductor
    reports what got done.
  - Every delegate message carries an `Idempotency-Key` (chess's semantics).
  - Plus #52's safety nets and its stop vocabulary.
- **F5. Structural confirmation.**
  - Apps refuse destructive calls with `confirm_required`, a token and their
    own question (chess's pattern).
  - Conductor holds the token in its state block, and a "yes" resumes the
    refused call.
  - The prompt rule stays as a backstop.
- **F6. Stop.** Conductor's own `…/cancel`, propagated to the app's
  `…/cancel` (PCC and music have it).
- **F7. Tracing and context capture** with per-hop ids (a port of PCC
  #400/#401). This gives failure attribution by construction.
- **F8. Session evals.**
  - Scripted multi-turn sessions built from §1.1, checking the end state per
    app plus claims, under the vendored gate.
  - The routing goldens, including `refuse-playback` (#36) and `music-save`
    (#54), stay green until they're replaced.

---

## 7. The options

### Option A: Switchboard, rebuilt

```
 you ─► conductor turn (synchronous)
          state block + recall + gather
          PLANNER 12B ── ask_<app>(words) ──────► app agent (its own loop)
             │   ▲                                   │
             │   └──────── reply + typed receipt ◄───┘
             ├── open_<app>(intent)    (handoff, local)
             └── search_memory(q)      (ledger of conductor's turns)
          NARRATOR 12B (reads receipts, can't act)
 job poller (code) ── watches app job ids ── posts "done" notes + push
```

Keep today's shape: agent-as-tool, depth 1, request/response. Fix
everything inside it (F1–F8). Add a memory search tool over conductor's own
turns and receipts, a code poller that watches job ids the apps return, and
an embeddings pre-router that trims the prompt's fleet section to the likely
apps. Conductor stays a router with a voice. This is the closest to "a
fourth copy," and it's the yardstick for the others.

**Turns.**
- **S1:** the planner calls `ask_tasks` with your words. PCC's planner runs
  and returns a record. The narrator speaks from it.
- **S3:** `ask_music` returns a record (job 17, album "Our Love"). The
  planner then calls `ask_tasks` with a message naming the album. The title
  passes through the 12B planner, which is where the paraphrase risk sits;
  F1's refs reduce it but don't remove it.
- **S4:** music's record carries job 17. The poller watches
  `/downloads/jobs/17`, and when it finishes it posts a note turn and a push
  notification.
- **S9:** PCC returns `confirm_required` with a token, the narrator asks
  PCC's question, and on "yes" a local `confirm` tool sends the token back.

**Context and memory.** Each turn gets a bounded state block, recall and
gather results. Cross-day memory comes from `search_memory` over
conductor's ledger. App-side context stays in per-app threads, and things
you did in app panels are invisible. "Never think about context" is earned
only for what was said to conductor. S5 has holes.

**Model.** 12B only. Escalation fits badly, because swapping to the 26B
evicts the 12B the app agents need mid-turn. Expect about 0 escalations a
day.

**Ripple.**
- `delegate-api.md` v1.1, additive: a `record` field, `confirm_required` and
  a token, `Idempotency-Key`, cancel, and a job ref plus a status read.
- `STANDARD.md` §3 confirmation.
- PCC and music implement it.
- It keeps depth 1, REST, no fan-out and no shared library.

**Latency (warm, estimate).** S1 ≈ 4 s. S3 ≈ 7–9 s. A cross-app turn needs
3–5 distinct prompt prefixes against 4 slots, so cache misses add 3–6 s
each. Voice adds ~2.5 s of STT and ~1 s of TTS.

**Biggest risk.** It stays a relay. Every cross-app ask chains several 12B
decisions over nested loops, so errors and latency compound. Memory is
split across apps, and async work and proactivity are bolted on.

**Spike A1 (1 day).** Run S3 and S15-style asks 20 times through real PCC
and music (fresh worlds). Pass: end state ≥80% correct (Wilson), p50 ≤8 s.

**Size.** ~12 PRs (S×3, M×7, L×2) across conductor, agent-standard, PCC and
music.

### Option B: One brain, app skills

```
 you ─► conductor: the only agent loop on the request path
          pre-router (embeddings) picks 1–2 SKILLS
          PLANNER 12B + [skill = app tools + app prompt lines
                          + app state-block/gather providers]
             └── MCP over HTTP ──► app service layer (no app model runs)
          NARRATOR 12B
          one ledger of every tool call across all apps
 apps keep their own agent for their own panel, built from the same skill
```

Apps stop being agents on conductor's path. Each ships a **skill**: its
tools over streamable-HTTP MCP, the prompt lines it tuned, and endpoints
that provide its state block and gather. Conductor is the single agent
(Cognition's principle). It sees every tool result first-hand, keeps one
memory, and does cross-app work in one loop. A shared `agentkit` runs the
same skill inside each app and inside conductor, so a skill is evaluated in
both places.

**Turns.**
- **S1:** the PCC skill loads, the state block answers or `list_tasks`
  runs, and the narrator speaks.
- **S3:** music and PCC skills load. `queue_album` runs, then
  `create_task(title=<the album from the typed result>)`. One loop.
- **S6:** the ledger already holds Tuesday's track id, so the call is
  `file_artist` directly.
- **S9:** the app's tool gate refuses with a token, and confirm resumes.

**Context and memory.** One ledger across apps makes S5 and S6 strong. The
context is bounded (state block, recall, the loaded skills). "Never think
about context" is earned for everything conductor did, and for app-panel
activity only if apps feed it in.

**Model.** 12B. The 26B can escalate a whole run for flagged plans, which
swaps the loop. Expect rare escalations.

**Ripple (large).**
- `STANDARD.md` §1 and §4 rewritten around skills.
- A new `skill:` manifest block replaces `agent:` for conductor.
- `delegate-api.md` is retired for conductor.
- The **`agentkit` library becomes required**.
- Every app's gate is re-baselined with its skill running inside the
  composite prompt.
- MCP moves from stdio to HTTP.
- It drops "app agents are the only way in" and "no shared library," and
  depth goes to 0.

**Latency.** S1 ≈ 2.5 s. S3 ≈ 4–5 s. When the set of loaded skills changes,
the changed tail (~1–2k tokens) is prefilled again, ~2–4 s.

**Biggest risk.** Prompt interference. PCC's lines were tuned with PCC alone
in the prompt. Next to conductor's rules, a second skill and recall, the
12B's tool choice may drift. One docstring line already cost PCC 20/20 →
0/20. Eval ownership also moves out of the apps.

**Spike B1 (2–3 days).** Run PCC's and music's gate scenarios through a
composite prompt (conductor base + both skills) against each app alone,
with paired seeds. Pass: no scenario below its floor, and no McNemar-
significant regression.

**Size.** ~22 PRs (L-heavy, including the `agentkit` extraction), across
every repo.

### Option C: Talker and crew

```
 surfaces (chat · voice · satellites · push)
      │ utterance                         ▲ speech / notes / briefs
      ▼                                   │
  TALKER 12B (fast; reads state; opens/joins jobs; never writes itself)
      │                                   │ reads
      ▼                                   │
  LEDGER (jobs · events · receipts · topics/side threads · facts)
      ▲ progress/results          ▲ app events (webhooks)   ▲ schedules
      │                           │                         │
  CREW: workers (12B loops with their own small context)
      ├── async tasks ──► app agents (depth ≤ 2 by budget; parallel reads)
      ├── plan jobs ────► 26B thinking (runs in the background)
      └── nightly: consolidate facts (librarian), build the brief
  GPU GATE: priority queue (voice > chat > background), aware of the resident model
```

Conductor becomes a runtime instead of one synchronous loop.
- A fast talker owns the conversation. It answers from state and memory when
  it can. When there's work, it opens or joins a **job** with A2A's
  lifecycle (submitted → working → input-required → done / failed /
  cancelled).
- Workers are conductor sub-agents with small private contexts. They
  delegate to app agents asynchronously and fan out reads.
- Everything lands on a ledger. Topics are side threads.
- Apps push events in. The night crew consolidates memory and writes the
  morning brief while the GPU is idle.
- Depth 1 becomes a budget.

**Turns.**
- **S4:** the talker says "queued, I'll tell you when it's in" within
  ~1.5 s. Music's events update the job. On completion, the notify policy
  routes it: a note if you're in chat, a push if you're away.
- **S5:** a memory job runs hybrid search over the ledger and facts (and
  PCC activity). The worker writes a cited answer, and the talker speaks it.
- **S10:** "stop" marks the job cancelled, workers propagate the cancel, the
  ledger shows what finished, and the talker reports and offers undo.
- **S14:** a plan job goes to the 26B in the background while the
  conversation carries on, and it comes back as a proposal you confirm.

**Context and memory.** Strongest of the four. The ledger is the system of
record, topics give scoped contexts, and nightly consolidation builds facts
with valid-from/invalid-at. Each turn's context is bounded: state (open
jobs, pending questions), recall, and retrieval. "Never think about
context" is earned end to end, but async narration has to stay honest about
state ("queued" is not "done").

**Model.** 12B for the talker and workers. The 26B runs plan jobs
asynchronously, so its 15–40 s plus the swap never blocks you; expect about
1–3 a day. Cloud is optional.

**Ripple (large).**
- `delegate-api.md` v2 goes async (create, status/stream, cancel,
  input-required), which is A2A-lite.
- A new `events:` manifest block, and apps emit webhooks.
- `STANDARD.md` §8: depth 1 → budgets, and parallel reads allowed.
- A GPU gate goes in front of llama-swap (every app's `LLAMACPP_BASE_URL`).
- `voice.md` gains out-of-band announcements.
- It drops depth 1, no-fan-out and request/response-only.

**Latency.** Acks <1.5 s. Simple answers ≈ A (~4 s). Long work never
blocks. Background model calls compete with the foreground unless the gate
pauses them at call boundaries.

**Biggest risk.** Complexity, and honesty under async state: the talker
must never say "done" for a job that's only queued. Also, one GPU makes
"parallel" mostly a matter of waiting on apps.

**Spike C1 (2 days).** (a) GPU contention: background generation plus
scripted foreground turns, measuring p50/p95 with and without a gate.
(b) Async honesty: the talker narrates a ledger with jobs in mixed states.
Pass: foreground p95 within 1.3× of idle with the gate, and claims ≥95%
backed.

**Size.** ~25 PRs (XL), across every repo plus a new gate service.

### Option D: Intent compiler (the bold one)

```
 utterance ─► gather + state block + recall (bounded)
                  │
           PLANNER 12B ─ constrained JSON ─► typed PLAN
                  │       a = ask_music("download the new Caribou album")
                  │       b = ask_tasks("add a task to sort {a.album.title} this weekend")
                  ▼
           VALIDATOR (code): types · refs exist · policy · budgets · taint
                  │ ok                     │ invalid → one repair │ hard → escalate (26B) │ ambiguous → ask_user
                  ▼
           INTERPRETER (code)
             · runs steps; independent reads in parallel; long steps become jobs
             · pauses at destructive steps; "yes" resumes the SAME plan
             · Stop cancels between steps; every step journals a receipt
                  ▼
           JOURNAL (receipts, entity refs, undo handles) ── memory, undo, reuse
                  ▼
           NARRATOR 12B (reads receipts; can't act)

 capabilities: ask_<app>(words) → typed record   (the app's agent, depth 1)
               <app>.<tool>(typed args) → typed out   (direct; reads first)
               open_<app>(intent) · ask_user(candidates) · think(q, inputs)
               recall(q) · notify(...) · undo(ref)
```

**The model understands, and code acts.**
- Each utterance becomes a small typed program over a fixed house catalog.
  The model decides **what**. Code decides **whether and how**: it
  validates, applies policy, executes and journals.
- The plan is fixed before any untrusted data is read, so a song title
  can't steer a write. A plan built after reading data (a repair, or a
  `think` step's output) is tainted, and its writes need confirmation.
- Single-app asks compile to one `ask_<app>(your words)` step, so the app's
  agent keeps its domain know-how, its eval gate keeps its meaning, and the
  cost is what it is today.
- Cross-app asks become data flow between typed records, not a 12B copying
  a title (#55).
- Undo, "do the same for…," "what did we do Tuesday?" and "stop" are
  operations over the journal, not model recollection.
- Fuzzy judgment over data is an explicit, bounded `think` step with no
  tools.
- Long steps become jobs. Proactive rules are stored plans on a schedule.

**Turns.**
- **S3:** the plan above (~100–150 output tokens). Step `a` returns at once
  with the queued job and the album metadata. Step `b` gets the title from
  the typed record. The narrator speaks.
- **S9:** a plan step `ask_tasks(...)` or `tasks.trash_project(@pcc:project:7)`.
  The app answers `confirm_required` with its own question, and the
  interpreter parks the plan in the journal. "Yes" resumes it with no
  re-plan, which fixes #37 by construction.
- **S11:** the interpreter reads the last plan's undo handles and runs the
  inverses. Where a capability declares none ("a download can't be
  un-downloaded"), the narrator says so.
- **S15:** recall shows the last plan as typed data, and the planner emits
  the same template with the new album.
- **S7:** gather surfaces "Dentist appointment" for "dennis." The planner
  picks it, or emits `ask_user` with the candidates (PCC's #420 pattern).

**Context and memory.**
- The journal is exact episodic memory, and the entity registry resolves
  "that one" and "the Caribou song."
- Words (yours, and the narrator's, labelled as words) sit beside receipts
  for S5.
- Facts are consolidated off the hot path in a later phase.
- Each turn's context is bounded; the catalog is a fixed, cached prefix with
  pinned bytes.
- "Never think about context" is earned by bounded turns, an unbounded
  journal and code-level references.

**Model.** The 12B plans and narrates, with `json_schema` constrained
decoding. Plan synthesis escalates to the 26B (thinking) when validation
fails twice or the planner emits `escalate`. Slow escalations run as jobs.
Expect about 1–3 a day. Cloud is optional (Q3).

**Ripple (medium-large).**
- A new `capability-catalog.md` contract:
  - MCP tools over streamable HTTP with **annotations** (read-only,
    destructive, idempotent) and **typed outputs**.
  - Entity refs, an undo/inverse declaration, a destructive preview,
    `Idempotency-Key`, and version preconditions.
- `delegate-api.md` stays, for `ask_<app>`, and gains F1/F4/F5/F6.
- `STANDARD.md` §1: annotations and output models become required.
- §8: depth stays 1 for `ask`, and direct capabilities nest nothing.
  Independent reads may run in parallel.
- `app-yaml` gets a `catalog:` block.
- It drops request/response-only (jobs), no-fan-out (reads), the shallow
  loop (replaced by a plan plus at most one repair) and the single-utterance
  goldens. It keeps depth 1 and no `agentkit` (the interpreter lives in
  conductor; the catalog is a contract).

**Latency.**
- S1 is ≈ today's ~4 s through `ask_pcc`, and ≈2 s with a direct read.
- S3 ≈ 5–8 s, or ≈3–4 s with direct capabilities.
- Plan tokens are the cost: a 5-step plan is ~300 tokens, ≈3 s.
- The whole turn uses 2 conductor prefixes.

**Biggest risk.** Whether the 12B writes correct typed programs: data flow,
references, the right step granularity. This is a cross-app version of
`plan_week`.

**Spike D1 (2–3 days, runnable now).**
- 40 utterances drawn from §1.1 (single-app, cross-app, references,
  confirms, refusals), 5 samples each.
- A catalog of ~25 capabilities built from the real PCC and music
  signatures, decoded with `json_schema`.
- Measure valid-plan rate, correct-plan rate (against labels) and latency,
  paired against today's loop on the single-app overlap.
- Pass:
  - single-app correctness not significantly below today's routing
  - cross-app ≥80% correct (Wilson)
  - ≥95% valid on the first try
  - single-step plans p50 ≤1.5 s

**Size.** ~20 PRs (L), across conductor, agent-standard, PCC and music.
Chess only if Q7 opens reads.

---

## 8. Comparison

| | A Switchboard | B One brain | C Talker & crew | D Compiler |
|---|---|---|---|---|
| S1–S2 simple | ok, ~4 s | good, ~2.5 s | ok, ~4 s (ack <1.5 s) | good, 2–4 s |
| S3, S15 cross-app, reuse | weak to ok (12B chain, paraphrase) | good (one loop) | ok (worker loop) | **good** (typed data flow, plan reuse) |
| S4 long jobs | ok (poller bolt-on) | ok (bolt-on) | **good** (native) | ok → good (jobs as steps; events later) |
| S5–S6 memory | weak to ok (conductor-only) | good (one ledger) | **good** (ledger, facts, topics) | good (exact journal; facts later) |
| S7 voice | ok | ok | **good** (acks, announcements) | good (fast, code ack) |
| S9 confirm + follow-through | ok (F5) | ok | ok | **good** (pause and resume the same plan) |
| S10 cancel | ok (F6) | ok | good | **good** (between steps, exact) |
| S11 undo across apps | weak | ok | ok | **good** (journal of inverses) |
| S13 proactive | weak | weak | **good** | ok (stored plans + digest; needs events) |
| S14 hard planning | weak | weak to ok | **good** (async escalation) | ok (escalated plan synthesis, can go async) |
| Honesty | ok | ok | at risk (async narration) | **best** (receipts; plan fixed before data) |
| Injection resistance | weak | weak | weak to ok | **best** |
| Code owns actions | ok | ok | ok | **best** |
| Feasible on the 12B alone | yes | uncertain (B1) | yes | likely (D1) |
| Prompt prefixes per cross-app turn | 3–5 | 2 + skill variants | 3–5 + background | 2 (+ apps' own for `ask`) |
| Contract ripple | small (additive) | large (`agentkit`, skills, eval ownership) | large (async v2, events, gate) | medium-large (catalog) |
| Build size | ~12 PRs (M) | ~22 (XL) | ~25 (XL) | ~20 (L) |
| Spike to retire the main risk | A1, 1 day | B1, 3 days | C1, 2 days | D1, 3 days |

**Revisitable decisions by option.**

| Decision | A | B | C | D |
|---|---|---|---|---|
| Delegation depth 1 | keep | drop (depth 0) | budget ≤2 | keep for `ask`; direct calls nest nothing |
| No parallel fan-out | keep | keep | drop (reads) | drop for independent reads |
| Shallow loop, `max_iterations` 6 | keep | grows to ~10 | per worker | replaced by plan + ≤1 repair |
| Request/response only | keep + poller | keep | drop (async tasks) | sync, plus jobs for long steps |
| REST delegate contract | v1.1 additive | retired for conductor | v2 async | kept for `ask`; MCP/HTTP for direct |
| App agents are the only way in | keep | drop | keep | drop for typed direct calls (reads first, Q7) |
| A bigger model is the last resort | keep | escalation | async escalation | escalation for plan synthesis |
| No `agentkit` | keep | **drop** | keep | keep |
| Single-utterance route goldens | + session evals | replace | replace | replace with plan-level + session evals |

---

## 9. Recommendation: D, phased, with C's runtime later

1. **It's the fleet's own winning rule applied to orchestration.** "Code
   owns actions; the model owns understanding and speech." Chess and PCC got
   their reliability by moving decisions into code. D moves orchestration's
   decisions there too: order, data passing, confirmation, cancel, undo and
   retry.
2. **It aims at the 12B's measured weaknesses instead of hoping they go
   away.**
   - Multi-step judgment: one plan, validated.
   - Relay loss: typed data flow.
   - Self-poisoning: memory is the journal, not the model's words.
   - Injection: the plan is fixed before data arrives.
3. **It keeps what's proven.** App agents still own their domains through
   `ask_<app>`, their gates still mean something, and a single-app ask costs
   what it costs today.
4. **It can be built gradually.**
   - Phase 1 is F1–F8, which every option needs anyway.
   - The compiler then runs in shadow mode beside today's loop. It logs the
     plans it would have run and is compared on live traffic.
   - It then takes over single-app routing behind a flag, and cross-app
     after that.
   - The routing goldens gate every switch. Today's conductor keeps working
     until the compiler path beats it.
5. **C's strengths come later, on D's journal.** Jobs, events, the daily
   brief, the GPU gate and sleep-time memory all build on the journal, which
   is the ledger C wants.

**Fallback.** If D1 fails (the 12B can't compile cross-app plans reliably),
go to C, with A's loop inside the workers. The foundations carry over
unchanged.

**Why not B.** It has the largest ripple, it takes eval ownership away from
the apps, and its core risk (prompt interference) is the failure mode the
fleet keeps measuring.

**Why not A.** It's the fourth copy. It fixes the bugs but leaves cross-app,
memory, async and proactivity bolted on.

**Spikes for Part B.** llama-swap is reachable from this session (all three
models listed, GPU idle), so the spikes run on the real model before the plan
is final:
- **S0:** swap timing 12B ↔ 26B under today's RAM pressure.
- **D1:** the plan-compiler bench (or the spike for the option you pick).
- **R1:** a #55-style relay probe with typed receipts in the narrator's
  brief.

---

## 10. Questions for you

1. **Direction:** D (recommended), or A, B or C?
2. **Priorities:** is §1.3's order right (honest > continuity > fast >
   control > broad > proactive)?
3. **Escalation:**
   - Local 26B-A4B only (no privacy change), or cloud too?
   - If cloud: which provider, opt-in per ask, never on voice, and which
     data may leave?
   - Cloud means amending `STANDARD.md` §8.
4. **Out-of-fleet asks (S8):** should conductor answer general knowledge
   from the model, labelled as not coming from an app, or keep refusing?
5. **Proactivity:**
   - What may reach you unprompted? (a) only callbacks you asked for;
     (b) plus one daily brief; (c) plus urgent events.
   - Through which channels: a chat note, a phone push, or spoken on
     satellites?
6. **One endless conversation:** replace the conversation list with one
   timeline and automatic topics, or keep manual conversations as well?
7. **Reach into apps:**
   - May conductor call app tools directly, without the app's agent: reads
     only, or writes too?
   - Should conductor's memory see what you do in app panels? That needs
     activity feeds from the apps.
8. **The chess model:** accept the ~7 s swap after chess play, or let
   conductor run on whichever model is resident? The second option needs a
   second profile and a second eval run.

---

## Sources

Dates are publication dates; "accessed" marks living docs. Items marked
*older* predate 2025.

**Vendor engineering and products**
- Anthropic, "How we built our multi-agent research system," 2025-06-13. https://www.anthropic.com/engineering/multi-agent-research-system
- Anthropic, "Effective context engineering for AI agents," 2025-09-29. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Anthropic, "Building effective agents," 2024-12-19 (*older*). https://www.anthropic.com/engineering/building-effective-agents
- Claude, "Memory," 2025-09-11, updated 2025-10-23. https://claude.com/blog/memory
- Claude, "Context management" (context editing, memory tool), 2025-09-29. https://claude.com/blog/context-management
- Claude Help, "Managing memory in Claude" (chat search as RAG tool calls), accessed 2026-11-05. https://support.claude.com/en/articles/11817273-managing-memory-in-claude
- Claude Code docs, subagents, accessed 2026-11-05. https://code.claude.com/docs/en/sub-agents
- Cognition, "Don't Build Multi-Agents," 2025-06-12. https://cognition.com/blog/dont-build-multi-agents
- LangChain, "How and when to build multi-agent systems," 2025-06-16. https://www.langchain.com/blog/how-and-when-to-build-multi-agent-systems
- LangChain, "Introducing ambient agents," 2025-01. https://blog.langchain.com/introducing-ambient-agents
- OpenAI Help, "How does reference saved memories work" (reference chat history launched 2025-04-10), accessed 2026-11-05. https://help.openai.com/en/articles/11146739-how-does-reference-saved-memories-work
- OpenAI, ChatGPT Pulse, 2025-09 (via launch coverage). https://www.tomsguide.com/ai/chatgpt-pulse-is-here-now-ai-starts-the-chat-and-curates-your-feed
- OpenAI Agents SDK, multi-agent orchestration guide, accessed 2026-11-05. https://openai.github.io/openai-agents-js/guides/multi-agent
- OpenAI, openai-realtime-agents (chat-supervisor pattern), 2025. https://github.com/openai/openai-realtime-agents
- Home Assistant community, "prefer handling commands locally" reports, 2025–2026. https://community.home-assistant.io/t/why-prefer-handling-commands-locally-increases-the-time-from-2-5s-to-52s/947662
- Anthropic model pricing, the claude-api reference cached 2026-10-06 (verify on the live pricing page).
- Gemini API free vs paid data use, accessed 2026-11-05 (secondary; verify on https://ai.google.dev/gemini-api/docs/pricing). https://ampm-aiops.com/en/guides/gemini-free-tier-data-tradeoff-2026/

**Protocols**
- MCP specification 2025-11-25 and SEP-1686 Tasks (experimental; moving to an extension in the draft). https://modelcontextprotocol.info/seps/1686-tasks/ · https://workos.com/blog/mcp-2025-11-25-spec-update
- MCP 2025-06-18, structured tool output (`outputSchema`); 2025-03-26, tool annotations. https://modelcontextprotocol.io/specification/2025-06-18/server/tools
- A2A protocol, v1.0 2026-03; Linux Foundation release 2026-04. https://a2a-protocol.org · https://github.com/a2aproject

**Papers and reports**
- Talker-Reasoner, arXiv 2410.08328, 2024-10 (*older*). https://arxiv.org/abs/2410.08328
- Small Language Models are the Future of Agentic AI (NVIDIA), arXiv 2506.02153, 2025-06. https://arxiv.org/abs/2506.02153
- Design Patterns for Securing LLM Agents against Prompt Injections, arXiv 2506.08837, 2025-06. https://simonwillison.net/2025/Jun/13/prompt-injection-design-patterns/
- CaMeL: Defeating Prompt Injections by Design, arXiv 2503.18813, 2025-03. https://arxiv.org/abs/2503.18813
- S. Willison, "The lethal trifecta," 2025-06-16. https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/
- Which Agent Causes Task Failures and When? (Who&When), arXiv 2505.00212, ICML 2025. https://arxiv.org/abs/2505.00212
- Why Do Multi-Agent LLM Systems Fail? (MAST), arXiv 2503.13657, 2025-03 (v3 2025-10). https://arxiv.org/abs/2503.13657
- τ²-bench, arXiv 2506.07982, 2025-06. https://github.com/sierra-research/tau2-bench
- Chroma, "Context Rot," 2025-07-14. https://research.trychroma.com/context-rot
- Zep: A Temporal Knowledge Graph Architecture for Agent Memory, arXiv 2501.13956, 2025-01. https://arxiv.org/abs/2501.13956
- Mem0, arXiv 2504.19413, 2025-04. https://arxiv.org/abs/2504.19413
- Sleep-time Compute (Letta), arXiv 2504.13171, 2025-04. https://arxiv.org/abs/2504.13171
- Blackboard LLM multi-agent systems: arXiv 2507.01701 (2025-07), arXiv 2510.01285 (2025-09). https://arxiv.org/abs/2507.01701 · https://arxiv.org/abs/2510.01285
- The Goldilocks Time Window for proactive interventions, arXiv 2504.09332, 2025-04. https://arxiv.org/abs/2504.09332
- BFCL v4 leaderboard, updated 2025-12-16. https://gorilla.cs.berkeley.edu/leaderboard.html
- llama.cpp server README (slots, unified KV, continuous batching), accessed 2026-11-05. https://github.com/ggml-org/llama.cpp/tree/master/tools/server

**Workspace sources read**
- `agent-standard/`: `STANDARD.md`, `AGENTS-MASTER-PLAN.md`, `delegate-api.md`, both `app-yaml-*` files, `model-profile.md`, `conductor-prompt-material.md`, `voice.md`, `VOICE-PLAN.md`
- `conductor/`: `backend/app/{ai,fleet,tools,mcp,api,services}`, `docs/agent-evals.md`, issues #36–#38, #40, #52–#55
- PCC: `docs/agent-design.md`, `agent-frontier.md`, `agent-tracing.md`, `agent-context-capture.md`, `api/routes_agent.py`
- chess: `CLAUDE.md`, `docs/agent-framework-research-2026-07-14.md`, `planner-narrator.md`, `turn-memory.md`, `second-brain.md`, `persistence-and-identity.md`, `model-profiles.md`, `latency-measurement.md`
- music: `app.yaml`, README Conventions, `docs/agent-evals.md`, `api/routes_agent.py`
- `llama-swap/config.yaml`
- `future-plans/`: `home-assistant-voice.md`, `personal-second-brain.md`, `hub-app.md`, `music-agent.md`
