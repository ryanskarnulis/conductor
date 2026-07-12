"""Per-tool-call plumbing shared by every tool consumer: the acting identity.

PCC's ``runtime`` also owns a ``tool_session`` context manager (a DB session
plus ``activity_events`` actor binding) because its tools write to a local
database. Conductor's tools don't: they are delegate calls to other apps'
agents, and each write is attributed downstream via the ``X-Agent-Actor`` header
those apps honor. So all that survives here is the actor contextvar — the
identity :func:`app.tools.registry.call_tool` binds for the duration of a call,
which a tool body can read to know who it is acting as.
"""

from __future__ import annotations

from contextvars import ContextVar

# Which agent identity the current tool call acts as. ``registry.call_tool``
# always sets this per dispatch (the in-app loop passes ``agent:loop``); the
# default is only a fallback for a tool read outside a dispatch.
current_tool_actor: ContextVar[str] = ContextVar("current_tool_actor", default="agent:loop")
