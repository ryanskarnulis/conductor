"""Per-tool-call plumbing shared by every tool consumer: identity, and what ran.

PCC's ``runtime`` also owns a ``tool_session`` context manager (a DB session
plus ``activity_events`` actor binding) because its tools write to a local
database. Conductor's tools don't: they are delegate calls to other apps'
agents, and each write is attributed downstream via the ``X-Agent-Actor`` header
those apps honor. So what survives here is per-call context rather than a
session: the identity :func:`app.tools.registry.call_tool` binds for the
duration of a call, which a tool body can read to know who it is acting as, and
what the app reported doing while it answered.
"""

from __future__ import annotations

from contextvars import ContextVar

# Which agent identity the current tool call acts as. ``registry.call_tool``
# always sets this per dispatch (the in-app loop passes ``agent:loop``); the
# default is only a fallback for a tool read outside a dispatch.
current_tool_actor: ContextVar[str] = ContextVar("current_tool_actor", default="agent:loop")


# What the app did while answering the current delegate call — the names of the
# tools *it* ran. Set by the ask tool, read by whoever dispatched it
# (``app/ai/loop.py``), and deliberately kept off the model's `role: tool`
# message: the model gets the app's reply, and a UI that needs to know what a
# turn actually did should not have to parse a paraphrase to find out.
#
# The dispatcher owns the lifetime — it sets ``None`` before each call and
# resets after — so one call's answer can never be read as the next one's.
delegated_tools: ContextVar[tuple[str, ...] | None] = ContextVar("delegated_tools", default=None)
