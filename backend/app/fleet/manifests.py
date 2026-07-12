"""Fleet discovery: read the sibling apps' ``app.yaml`` manifests.

The fleet is declarative. Each app in the workspace drops an ``app.yaml`` at its
repo root (schema in ``../gateway/README.md``); an app that ships a standard
agent adds an ``agent:`` block (``../agent-standard/app-yaml-agent-block.md``).
Conductor scans ``{fleet_manifest_dir}/*/app.yaml``, records every well-formed
app as a fleet member, and — for the ones with an ``agent:`` block — learns the
delegate base path and the routing hints. Adding an app to the fleet is then
purely declarative: no conductor code changes.

Discovery never crashes on a bad manifest. A file that isn't valid YAML, isn't
a mapping, or is missing a required field is skipped with a ``structlog``
warning; conductor's own manifest is skipped silently (it is the delegation
root and must never be a delegate of itself); a malformed ``agent:`` block
degrades the app to a non-agent fleet member (recorded, but it gets no tool).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import yaml

from app.config import get_settings

logger = structlog.get_logger(__name__)

# Conductor's own manifest name — skipped during discovery. Conductor is the
# delegation root; the depth-1 rule (../agent-standard/STANDARD.md) forbids it
# from ever being another agent's delegate, so it never builds a tool for
# itself even if its own app.yaml is inside the scanned directory.
SELF_APP_NAME = "conductor"


@dataclass(frozen=True)
class AgentSpec:
    """An app's ``agent:`` block: what the agent does and how to reach it."""

    # One sentence, third person — becomes the delegate tool's description.
    description: str
    # Delegate API base path on the app's upstream, e.g. "/api/agent".
    api: str
    # Routing hints; conductor embeds them in its system prompt.
    examples: tuple[str, ...]


@dataclass(frozen=True)
class FleetApp:
    """One fleet member. ``agent`` is ``None`` for apps without an agent."""

    name: str
    title: str
    # host:port — the manifest's upstream, with the host half rewritten when
    # `fleet_upstream_host` is set (docker); the port is always preserved.
    upstream: str
    agent: AgentSpec | None

    @property
    def has_agent(self) -> bool:
        return self.agent is not None

    @property
    def agent_base_url(self) -> str:
        """Base URL of this app's delegate API, e.g. ``http://127.0.0.1:8100/api/agent``.

        Raises :class:`ValueError` for a non-agent app — callers gate on
        :attr:`has_agent` (or iterate :meth:`Fleet.agent_apps`) first.
        """
        if self.agent is None:
            raise ValueError(f"{self.name} has no agent block")
        return f"http://{self.upstream}{self.agent.api}"


@dataclass(frozen=True)
class Fleet:
    """The discovered fleet: every member, agent-bearing or not."""

    apps: tuple[FleetApp, ...]

    def agent_apps(self) -> tuple[FleetApp, ...]:
        """Members that ship an agent (the ones conductor builds tools for)."""
        return tuple(app for app in self.apps if app.has_agent)

    def non_agent_apps(self) -> tuple[FleetApp, ...]:
        """Members without an agent — known to the fleet, but not delegable."""
        return tuple(app for app in self.apps if not app.has_agent)

    def get(self, name: str) -> FleetApp | None:
        return next((app for app in self.apps if app.name == name), None)


def _rewrite_host(upstream: str, upstream_host: str) -> str | None:
    """Return ``upstream`` with its host replaced by ``upstream_host`` (port kept).

    Returns ``None`` if ``upstream`` isn't ``host:port``. An empty
    ``upstream_host`` leaves the manifest value verbatim.
    """
    host, sep, port = upstream.rpartition(":")
    if not sep or not host or not port:
        return None
    if not upstream_host:
        return upstream
    return f"{upstream_host}:{port}"


def _parse_agent(block: Any) -> AgentSpec | None:
    """An :class:`AgentSpec` from a manifest ``agent:`` block, or ``None``.

    ``None`` means "no usable agent" — either the block is absent or it is
    malformed (missing ``description``/``api``, or wrong types). A malformed
    block is treated as absent so the app stays a fleet member without a tool.
    """
    if block is None:
        return None
    if not isinstance(block, dict):
        return None
    description = block.get("description")
    api = block.get("api")
    if not isinstance(description, str) or not description.strip():
        return None
    if not isinstance(api, str) or not api.startswith("/"):
        return None
    raw_examples = block.get("examples") or []
    examples: tuple[str, ...] = ()
    if isinstance(raw_examples, list):
        examples = tuple(str(e) for e in raw_examples if isinstance(e, str))
    return AgentSpec(description=description.strip(), api=api, examples=examples)


def _load_manifest(path: Path, *, upstream_host: str, self_name: str) -> FleetApp | None:
    """Parse one ``app.yaml`` into a :class:`FleetApp`, or ``None`` to skip it."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("fleet_manifest_unreadable", path=str(path), error=str(exc))
        return None
    if not isinstance(raw, dict):
        logger.warning("fleet_manifest_not_a_mapping", path=str(path))
        return None

    name = raw.get("name")
    title = raw.get("title")
    upstream = raw.get("upstream")
    if not isinstance(name, str) or not name:
        logger.warning("fleet_manifest_missing_field", path=str(path), field="name")
        return None
    if name == self_name:
        # Conductor's own manifest — expected, not an error.
        logger.debug("fleet_manifest_skipped_self", path=str(path), name=name)
        return None
    if not isinstance(title, str) or not title:
        logger.warning("fleet_manifest_missing_field", path=str(path), field="title", name=name)
        return None
    if not isinstance(upstream, str) or not upstream:
        logger.warning("fleet_manifest_missing_field", path=str(path), field="upstream", name=name)
        return None

    resolved_upstream = _rewrite_host(upstream, upstream_host)
    if resolved_upstream is None:
        logger.warning("fleet_manifest_bad_upstream", path=str(path), name=name, upstream=upstream)
        return None

    agent = _parse_agent(raw.get("agent"))
    if raw.get("agent") is not None and agent is None:
        logger.warning("fleet_agent_block_malformed", path=str(path), name=name)

    return FleetApp(name=name, title=title, upstream=resolved_upstream, agent=agent)


def discover_fleet(
    manifest_dir: Path,
    *,
    upstream_host: str = "",
    self_name: str = SELF_APP_NAME,
) -> Fleet:
    """Scan ``{manifest_dir}/*/app.yaml`` and return the discovered :class:`Fleet`.

    Deterministic (apps sorted by directory name). Bad manifests are skipped
    with a warning, never raised; ``self_name`` is skipped silently. When
    ``upstream_host`` is set, every app's upstream host is rewritten to it
    (port preserved) — docker reaches host-bound apps via
    ``host.docker.internal``.
    """
    apps: list[FleetApp] = []
    if not manifest_dir.is_dir():
        logger.warning("fleet_manifest_dir_missing", path=str(manifest_dir))
        return Fleet(apps=())
    for path in sorted(manifest_dir.glob("*/app.yaml")):
        app = _load_manifest(path, upstream_host=upstream_host, self_name=self_name)
        if app is not None:
            apps.append(app)
    logger.info(
        "fleet_discovered",
        manifest_dir=str(manifest_dir),
        apps=len(apps),
        agents=sum(1 for app in apps if app.has_agent),
    )
    return Fleet(apps=tuple(apps))


def fleet_from_settings() -> Fleet:
    """The fleet as configured (``FLEET_MANIFEST_DIR`` / ``FLEET_UPSTREAM_HOST``)."""
    settings = get_settings()
    return discover_fleet(
        settings.fleet_manifest_dir,
        upstream_host=settings.fleet_upstream_host,
    )
