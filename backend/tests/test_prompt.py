"""System-prompt composition: vendored personality + layered assembly."""

from __future__ import annotations

from datetime import date

from app.ai import loop as loop_module
from app.ai.loop import build_system_prompt


def test_vendored_personality_loads_with_its_header_stripped() -> None:
    personality = loop_module._GLOBAL_PERSONALITY

    assert not personality.startswith("<!--")
    assert "<!-- vendored" not in personality
    assert personality.startswith("Your personality: you are Glitch")
    # The brevity/honesty contract that must survive being layered in.
    assert "The character never overrides the job" in personality


def test_layers_compose_in_order_base_then_glitch_then_date() -> None:
    prompt = build_system_prompt(date(2026, 7, 11))

    # Layer 1: conductor's behavioral contract.
    assert "local AI conductor" in prompt
    assert "Act only through your tools" in prompt
    # Layer 2: the vendored house personality.
    assert "you are Glitch" in prompt
    # Layer 3: the dynamic date injection.
    assert "Today's date is 2026-07-11." in prompt

    base_at = prompt.index("Act only through your tools")
    glitch_at = prompt.index("you are Glitch")
    date_at = prompt.index("Today's date is 2026-07-11.")
    assert base_at < glitch_at < date_at


def test_date_is_actually_injected() -> None:
    assert "Today's date is 2020-01-02." in build_system_prompt(date(2020, 1, 2))
    assert "Today's date is 1999-12-31." in build_system_prompt(date(1999, 12, 31))


def test_base_prompt_covers_conductor_behavioral_contract() -> None:
    prompt = build_system_prompt(date(2026, 7, 11))

    # Tools-only action / no invented results / answer for an app from memory.
    assert "never answer for an app from memory" in prompt
    # App agents are the source of truth, relayed faithfully.
    assert "source of truth" in prompt
    assert "faithfully" in prompt
    # Clarify-on-ambiguity.
    assert "clarifying question" in prompt
    # Conductor owns destructive-op confirmation.
    assert "destructive" in prompt
    assert "confirm" in prompt
    # Out-of-fleet requests get an honest refusal.
    assert "no app in the fleet" in prompt
    # Local-first.
    assert "stays local" in prompt
