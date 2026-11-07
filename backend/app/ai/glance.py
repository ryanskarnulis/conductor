"""A glance as a receipt (#73): what a ``look`` step puts on the record.

An app's glance (``../agent-standard/app-yaml-glance-block.md``) is a small
JSON object it defines; the rows it names are entity refs
(``{app, kind, id, title}``) and any outside text is flagged
``"untrusted": true``. This module turns one into the two things the rest of
conductor reads, with no model in between:

- the **view**: the glance cut down and made safe to hand the narrator. An
  untrusted title is cleaned (one line, no quotes or braces, clipped) and
  marked ``outside_text``, the way receipts already mark it; lists and text
  are capped, so a glance can't spend more than a few hundred tokens;
- one **record entry**, ``looked_up``, whose ``touched`` is every ref the
  glance names: the same shape as an app's delegate receipt, so the ledger,
  the narrator's must-say titles and the state block's recent entities treat
  a look like any other read, with no writes.

Pure: no I/O.
"""

from __future__ import annotations

import json
from typing import Any

from app.ai.dataflow import clean

# A glance is ~150 tokens by contract; these keep a careless one from being more.
MAX_DEPTH = 5
MAX_ITEMS = 20
MAX_TEXT = 200
MAX_REFS = 20

# The entry's tool name on the record: the glance itself, not one of the app's tools.
GLANCE_TOOL = "glance"


def receipt(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The view (as compact JSON text) and the ``looked_up`` entry for a glance."""
    refs: list[dict[str, Any]] = []
    view = _view(payload, 0, refs)
    touched = _dedupe(refs)[:MAX_REFS]
    entry: dict[str, Any] = {
        "tool": GLANCE_TOOL,
        "outcome": "looked_up",
        "touched": touched,
        "count": len(touched),
    }
    return json.dumps(view, ensure_ascii=False, separators=(",", ":")), entry


def _is_ref(node: dict[str, Any]) -> bool:
    return (
        isinstance(node.get("kind"), str)
        and node.get("id") is not None
        and isinstance(node.get("title"), str)
    )


def _view(node: Any, depth: int, refs: list[dict[str, Any]], outside: bool = False) -> Any:
    """``node`` capped and cleaned. ``outside``: it sits inside an object
    flagged untrusted, so its text is outside text too."""
    if isinstance(node, dict):
        if depth >= MAX_DEPTH:
            return None
        flagged = outside or node.get("untrusted") is True
        if _is_ref(node):
            ref = {
                "app": str(node.get("app") or ""),
                "kind": node["kind"],
                "id": str(node["id"]),
                "title": clean(node["title"]) if flagged else _clip(node["title"]),
            }
            if flagged:
                ref["untrusted"] = True
            refs.append(ref)
            shown = dict(ref)
            if flagged:
                shown.pop("untrusted")
                shown["outside_text"] = True
            return shown
        shown = {
            str(key): _view(value, depth + 1, refs, flagged)
            for key, value in list(node.items())[:MAX_ITEMS]
            if key != "untrusted"
        }
        if flagged and not outside:
            shown["outside_text"] = True
        return shown
    if isinstance(node, list):
        return [_view(item, depth + 1, refs, outside) for item in node[:MAX_ITEMS]]
    if isinstance(node, str):
        return clean(node) if outside else _clip(node)
    if node is None or isinstance(node, bool | int | float):
        return node
    return _clip(str(node))


def _clip(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= MAX_TEXT else flat[: MAX_TEXT - 1].rstrip() + "…"


def _dedupe(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for ref in refs:
        key = (ref["app"], ref["kind"], ref["id"])
        if key not in seen:
            seen.add(key)
            unique.append(ref)
    return unique
