"""Conductor's model profiles: what it knows about one model, kept out of the code.

The profile format and its strict loader are the fleet's, vendored verbatim in
:mod:`app.ai.agentkit.model_profile` (canonical in
``../agent-standard/agentkit/``, agent-standard#32). This module is what is
conductor's own: its phases (:data:`SPEC`), its profile files, the cache, and
the warning for a model nobody has measured here.

A profile is one TOML file in ``app/ai/profiles/``, named for the llama-swap
model id it describes; the provider loads the one named for its model
(``LLAMACPP_MODEL``, ``CONDUCTOR_THINK_MODEL``). Conductor has four phases:

- ``[phases.planner]``, ``[phases.narrator]``, ``[phases.speaker]`` — the
  roles a turn's calls sample as (#150). Each takes ``max_tokens``, the most
  one call in that role may ask for (a prompt's own smaller cap still wins),
  and ``temperature`` to set the role apart from ``[sampling]``'s. None of
  them thinks, so ``thinking`` there is an error.
- ``[phases.think]`` — the escalation lane's call (#78): its ``max_tokens``
  cap and ``thinking``. It samples at ``[sampling]``'s.

A model with no file runs on :data:`DEFAULT_PROFILE` — no sampling opinion
(the server's), today's caps, the common thinking toggle — and a warning.

``tests/test_request_bytes.py`` pins that the shipped files send the bytes
conductor sent before profiles existed.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import structlog

from app.ai.agentkit.model_profile import (
    DEFAULT_NAME,
    ModelProfile,
    PhaseSettings,
    ProfileError,
    ProfileSpec,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_NAME",
    "DEFAULT_PROFILE",
    "DEFAULT_THINK_MAX_TOKENS",
    "NARRATOR",
    "PLANNER",
    "PROFILES_DIR",
    "SPEAKER",
    "SPEC",
    "THINK",
    "ModelProfile",
    "PhaseSettings",
    "ProfileError",
    "load_profile",
    "parse_profile",
]

logger = structlog.get_logger(__name__)

PROFILES_DIR = Path(__file__).with_name("profiles")

# Conductor's phases: the three roles a turn's calls sample as (#150), and the
# escalation lane's one call (#78).
PLANNER = "planner"
NARRATOR = "narrator"
SPEAKER = "speaker"
THINK = "think"

# The caps a model with no profile runs at: STANDARD.md §3's per-call cap for
# a turn's roles, and room for the thought and a dozen steps for a think
# (spike D1 ran thinking at 6000).
DEFAULT_MAX_TOKENS = 2048
DEFAULT_THINK_MAX_TOKENS = 6000

SPEC = ProfileSpec(
    phases={
        PLANNER: PhaseSettings(max_tokens=DEFAULT_MAX_TOKENS),
        NARRATOR: PhaseSettings(max_tokens=DEFAULT_MAX_TOKENS),
        SPEAKER: PhaseSettings(max_tokens=DEFAULT_MAX_TOKENS),
        THINK: PhaseSettings(max_tokens=DEFAULT_THINK_MAX_TOKENS, thinking=True),
    },
    phase_keys={
        PLANNER: ("temperature",),
        NARRATOR: ("temperature",),
        SPEAKER: ("temperature",),
        THINK: ("thinking",),
    },
)

DEFAULT_PROFILE = SPEC.default_profile()


def parse_profile(name: str, text: str, source: str) -> ModelProfile:
    """A profile from its TOML text. Strict: anything unknown or mistyped fails."""
    return SPEC.parse(name, text, source)


@cache
def load_profile(model: str) -> ModelProfile:
    """The profile for ``model``, or :data:`DEFAULT_PROFILE` with a warning when
    no file describes it. Cached: a profile is read once per process."""
    profile = SPEC.load(PROFILES_DIR, model, "app/ai/profiles")
    if profile is None:
        logger.warning("model_profile_missing", model=model, using=DEFAULT_NAME)
        return DEFAULT_PROFILE
    return profile
