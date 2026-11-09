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

- the conversation, the utterance, the opening it was handed (`context`:
  state block and recall) and the compiled `plan`;
- `calls[]`: role (`compiler`/`narrator`/`speaker`; `planner` on old records), status (`ok`/`truncated`/`bad_args`/`failed`), ms,
  tokens, finish reason, and (#152) `cached_tokens` (prompt tokens the
  server's cache served) and `prefill_ms` (time reading the rest), each `null`
  when the server didn't say; a turn's `cached_tokens` is their sum;
- `surface` and `narration` (#65): how the reply was made (`path`: narrated,
  respoken, appended, handoff, confirm, fallback, `speaker:*` and `code:*`
  for replies code built), what code
  appended, and the narrator's calls and ms;
- `tools[]`: every step's call as the interpreter recorded it;
- `hops[]`: one per delegate call. Each hop has its `hop_id`, the app, the
  subagent thread, the app's `X-Request-ID` (`app_request_id`), what the app
  ran, its stop reason or error class, and ms; for a v1.1 app also the
  hop's `idempotency_key` and the app's typed record (`app_record`, its
  receipts, also from a failure body);
- the stop reason, error, reply, and the model with the planner prompt's
  hash.

Schema 2 (#75) dropped the loop's `history`, `iterations`, `corrections`,
`limits` and tool-schema fields.

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

Replay runs today's interpreter and provider offline over today's fleet
(`FLEET_MANIFEST_DIR`). Model responses come from the capture and step
answers from the trace; the planner prompt is rebuilt from the manifests, so a
manifest change since shows as a drifted request. A turn that resumed a
paused plan or acted on the ledger reads state the trace doesn't keep. It prints `FAITHFUL` when every request is rebuilt byte
for byte and the stop, reply and tool calls match. Otherwise it prints the
first request that drifted. A traced misfire is a ready-made regression case.

For the deployed container, set the paths in an override file outside
`~/deploy` (CD refuses a dirty clone), pointing at a mounted volume.
