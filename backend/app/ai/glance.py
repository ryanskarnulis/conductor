"""A glance as a receipt (#73): what a ``look`` step puts on the record.

An app's glance (``../agent-standard/app-yaml-glance-block.md``) is a small
JSON object it defines; the rows it names are entity refs
(``{app, kind, id, title}``) and any outside text is flagged
``"untrusted": true``. This module turns one into the two things the rest of
conductor reads, with no model in between:

- the **view**: the glance cut down and made safe to hand the narrator. An
  untrusted title is cleaned (one line, no quotes or braces, clipped) and
  marked ``outside_text``, the way receipts already mark it; lists and text
  are capped, so a glance can't spend more than a few hundred tokens. A ref
  keeps its other scalar fields beside the four ref keys (#161), bounded and
  cleaned like its title; when one is a last-changed time (``last_move_at``),
  code states the age as a plain fact on the household's clock;
- one **record entry**, ``looked_up``, whose ``touched`` is every ref the
  glance names: the same shape as an app's delegate receipt, so the ledger,
  the narrator's must-say titles and the state block's recent entities treat
  a look like any other read, with no writes.

Pure apart from the household clock (:mod:`app.clock`).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from app import clock
from app.ai.dataflow import clean

# A glance is ~150 tokens by contract; these keep a careless one from being more.
MAX_DEPTH = 5
MAX_ITEMS = 20
MAX_TEXT = 200
MAX_REFS = 20
# A ref's scalar fields beyond app, kind, id and title (#161).
MAX_REF_FIELDS = 8

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


_REF_KEYS = frozenset({"app", "kind", "id", "title"})
# Keys code writes itself, so an app's own can't stand in for them.
_RESERVED = frozenset({"untrusted", "outside_text", "age"})
# An app's field names are code-like words; anything else is dropped, not shown.
_FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,31}")
# The contract's optional age (``last_move_at``): "last move", "last update".
_AGE_FIELD = re.compile(r"(?:last_(?P<what>[a-z]+(?:_[a-z]+)*)|(?P<updated>updated))_at")


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
            shown |= _extras(node, flagged)
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


def _extras(node: dict[str, Any], flagged: bool) -> dict[str, Any]:
    """A ref's other scalar fields (#161): bounded in count and length, no
    nested values, cleaned like the title when the ref is outside text. A
    last-changed time becomes its age, never the raw UTC stamp."""
    extras: dict[str, Any] = {}
    age: str | None = None
    for key, value in node.items():
        if not isinstance(key, str) or key in _REF_KEYS or key in _RESERVED:
            continue
        if not _FIELD_NAME.fullmatch(key) or value is None or isinstance(value, dict | list):
            continue
        stamp = _AGE_FIELD.fullmatch(key)
        if stamp is not None:
            age = age or _age(value, stamp.group("what") or "update")
            continue
        if len(extras) >= MAX_REF_FIELDS:
            continue
        if isinstance(value, str):
            extras[key] = clean(value) if flagged else _clip(value)
        elif isinstance(value, bool | int | float):
            extras[key] = value
        else:
            extras[key] = _clip(str(value))
    if age is not None:
        extras["age"] = age
    return extras


def _age(value: Any, what: str) -> str | None:
    """"last move 19 days ago": days between the household's calendar day of
    ``value`` (ISO 8601, UTC unless it says otherwise) and its today (#130).
    ``None`` when it isn't a readable time."""
    if not isinstance(value, str):
        return None
    try:
        when = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    days = max((clock.local_today() - clock.local_day(when)).days, 0)
    ago = "today" if days == 0 else "yesterday" if days == 1 else f"{days} days ago"
    return f"last {what.replace('_', ' ')} {ago}"


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
