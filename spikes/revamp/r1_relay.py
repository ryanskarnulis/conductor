"""R1: does a narrator speaking from a typed receipt keep what the app touched?

Arm "today": conductor's real system prompt and tools, the real loop's second
call (user ask -> ask_music call -> music's handoff text as the tool result).
Arm "receipt": a narrator with no tools reading a code-built typed receipt.
The handoff text is rebuilt from music's run_record.handoff wording.

    python r1_relay.py ARM RUNS
    python r1_relay.py --report
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from common import OUT, append, chat, load, wilson

sys.path.insert(0, "/home/rskarnulis/Documents/workspace/conductor/backend")

from app.ai.loop import build_system_prompt  # noqa: E402
from app.fleet.manifests import Fleet, discover_fleet  # noqa: E402
from app.fleet.tools import render_fleet_section  # noqa: E402

PATH = OUT / "r1.jsonl"
USER = "download this song please https://example.com/watch?v=abc123"
TITLES = {"odessa": ("Odessa", "Caribou"), "real": ("Real Title", "Real Artist")}


def _fleet() -> Fleet:
    fleet = discover_fleet(Path("/home/rskarnulis/Documents/workspace"))
    seen: set[str] = set()
    apps = []
    for app in fleet.apps:
        if app.name not in seen:
            seen.add(app.name)
            apps.append(app)
    return Fleet(apps=tuple(apps))


FLEET = _fleet()
SYSTEM = build_system_prompt(date(2026, 11, 5), fleet_section=render_fleet_section(FLEET))
TOOLS = [
    {"type": "function", "function": {
        "name": f"ask_{a.name}", "description": a.agent.description,
        "parameters": {"type": "object", "properties": {"message": {"type": "string"}},
                       "required": ["message"]}}}
    for a in FLEET.agent_apps() if a.agent
] + [
    {"type": "function", "function": {
        "name": f"open_{a.name}", "description": a.open.description,
        "parameters": {"type": "object", "properties": {"intent": {"type": "string"}},
                       "required": ["intent"]}}}
    for a in FLEET.open_apps() if a.open
]


def handoff(title: str, artist: str) -> str:
    result = json.dumps({"id": "a1b2c3", "title": title, "artist": artist,
                         "filename": f"{title}.mp3", "folder": "Unsorted",
                         "duration_seconds": 312})
    return (
        "Music's record of this run (built by the app from its tool results; nothing "
        'outside "Done" was saved or changed):\n'
        "Done:\n"
        f'- #1 download_song {{"query": "https://example.com/watch?v=abc123"}} -> {result}\n\n'
        "The agent's note (its reading of the results, not a record): "
        "\"bet. it's in the Unsorted folder now.\"\n\n[music did: download_song]"
    )


NARRATOR = """\
You are the household's local assistant, speaking to the person after their \
request ran. You did nothing yourself: below are the receipts the apps \
returned, built by code from what actually happened. Say what happened in one \
or two short, natural sentences. Name what each step touched by its title or \
name, exactly as the receipt writes it. Never claim anything a receipt doesn't \
show; if a step was refused or not run, say so plainly."""


def receipt(title: str, artist: str) -> str:
    return json.dumps({
        "steps": [{
            "id": "s1", "app": "music", "outcome": "done", "did": "saved one song",
            "touched": [{"ref": "music:track:812", "kind": "song", "title": title,
                         "artist": artist, "folder": "Unsorted"}],
            "undo": "none", "app_note": "bet. it's in the Unsorted folder now.",
        }]
    }, indent=1)


def receipt2(title: str, artist: str) -> str:
    """The narrator's view: titles only; refs stay with the interpreter."""
    return json.dumps({
        "steps": [{
            "app": "music", "outcome": "done", "did": "saved one song",
            "items": [{"title": title, "artist": artist, "folder": "Unsorted"}],
            "can_undo": False, "app_note": "bet. it's in the Unsorted folder now.",
        }]
    }, indent=1)


def run(arm: str, runs: int) -> None:
    for key, (title, artist) in TITLES.items():
        for index in range(runs):
            if arm == "today":
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER},
                    {"role": "assistant", "content": "", "tool_calls": [{
                        "id": "call_1", "type": "function", "function": {
                            "name": "ask_music",
                            "arguments": json.dumps({"message": USER})}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": handoff(title, artist)},
                ]
                reply, message = chat("gemma-4-12b", messages, tools=TOOLS, max_tokens=300)
                extra_calls = len(message.get("tool_calls") or [])
            else:
                messages = [
                    {"role": "system", "content": NARRATOR},
                    {"role": "user", "content": f"The person said: {USER}\n\nReceipts:\n"
                                                f"{(receipt2 if arm == 'receipt2' else receipt)(title, artist)}"},
                ]
                reply, _ = chat("gemma-4-12b", messages, max_tokens=300)
                extra_calls = 0
            text = reply.content.strip()
            kept = title.lower() in text.lower()
            row = {"arm": arm, "title": key, "run": index, "kept": kept,
                   "extra_tool_calls": extra_calls, "seconds": round(reply.seconds, 2),
                   "reply": text}
            append(PATH, row)
            print(f"{arm} {key} {'KEPT' if kept else 'drop'} {text[:110]!r}", flush=True)


def report() -> None:
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in load(PATH):
        cells[(row["arm"], row["title"])].append(row)
    for (arm, title), rows in sorted(cells.items()):
        kept = sum(r["kept"] for r in rows)
        lo, hi = wilson(kept, len(rows))
        extra = sum(1 for r in rows if r["extra_tool_calls"])
        print(f"{arm:8} {title:7} kept {kept}/{len(rows)} ci [{lo:.2f}, {hi:.2f}]"
              f"  extra tool calls {extra}")


if __name__ == "__main__":
    if sys.argv[1] == "--report":
        report()
    else:
        run(sys.argv[1], int(sys.argv[2]))
