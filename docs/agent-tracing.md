# Tracing, context capture and replay

Three diagnostics. All are off by default and never cost a turn: a write that
fails is logged and dropped.

| Env var | Writes | Module |
| --- | --- | --- |
| `CONDUCTOR_TRACE_PATH` | one JSONL record per turn | `app/ai/trace.py` |
| `CONDUCTOR_CONTEXT_PATH` | one JSONL line per model call, byte for byte (heavy) | `app/ai/context_capture.py` |

## What a turn record holds

The turn id is `run_id`: its log lines, its trace record, its captured calls
and `AgentRunResult.run_id` all carry it. The record also holds:

- the conversation, utterance and history it was handed;
- `calls[]`: role (`planner`/`narrator`), status (`ok`/`truncated`/`bad_args`/`failed`), ms,
  tokens, finish reason;
- `surface` and `narration` (#65): how the reply was made (`path`: narrated,
  respoken, appended, handoff, confirm, fallback, planner), what code
  appended, and the narrator's calls and ms;
- `tools[]`: every tool call as the loop recorded it;
- `hops[]`: one per delegate call. Each hop has its `hop_id`, the app, the
  subagent thread, the app's `X-Request-ID` (`app_request_id`), what the app
  ran, its stop reason or error class, and ms; for a v1.1 app also the
  hop's `idempotency_key` and the app's typed record (`app_record`, its
  receipts, also from a failure body);
- the stop reason, error, reply, limits, and the model with hashes of the
  system prompt and tool schemas.

Every delegate request carries `X-Agent-Turn: <run_id>` and
`X-Agent-Hop: <hop_id>`. The `delegate_call` log line carries `hop_id` and
`app_request_id`, so conductor's log, its trace and the app's own request
logs join. Apps don't record the hop headers in their own traces yet; that
is the contract track's (agent-standard#19).

## Reading and replaying

```bash
cd backend
.venv/bin/python scripts/read_trace.py "$CONDUCTOR_TRACE_PATH" --last 5
.venv/bin/python scripts/replay_turn.py "$CONDUCTOR_TRACE_PATH" "$CONDUCTOR_CONTEXT_PATH" <run_id>
```

Replay runs today's loop and provider offline. Model responses come from the
capture, tool answers from the trace, and the system prompt and tool schemas
are the captured ones. It prints `FAITHFUL` when every request is rebuilt byte
for byte and the stop, reply and tool calls match. Otherwise it prints the
first request that drifted. A traced misfire is a ready-made regression case.

For the deployed container, set the paths in an override file outside
`~/deploy` (CD refuses a dirty clone), pointing at a mounted volume.
