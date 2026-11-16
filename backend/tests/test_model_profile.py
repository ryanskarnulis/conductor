"""Per-model profiles (agent-standard#32): the shipped files, strictness, the default."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import structlog

from app.ai.model_profile import (
    DEFAULT_NAME,
    DEFAULT_PROFILE,
    PROFILES_DIR,
    ProfileError,
    load_profile,
    parse_profile,
)
from app.config import Settings


def _fleet_profile() -> Path | None:
    """``../agent-standard/model-profile.md`` beside the repo, if checked out.

    Found by walking up, so a git worktree under ``.claude/worktrees/`` finds
    it too. CI checks out this repo alone, so there the drift test skips (the
    sync-check job is what watches the fleet's files there).
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "agent-standard" / "model-profile.md"
        if candidate.is_file():
            return candidate
    return None


_FLEET_PROFILE = _fleet_profile()
# A row of the fleet profile's sampling table: | `top_k` | 64 (...) |
_FLEET_ROW = r"\|\s*`(temperature|top_p|top_k)`\s*\|\s*([\d.]+)"
# Its per-role row for conductor: | conductor planner | 1.0 | ...
_FLEET_PLANNER = r"\|\s*conductor planner\s*\|\s*([\d.]+)"

_ROLES = ("planner", "narrator", "speaker")


# --- the shipped profiles --------------------------------------------------------------


def test_the_12b_profile_holds_the_shipped_numbers() -> None:
    gemma = load_profile("gemma-4-12b")
    assert gemma.name == "gemma-4-12b"
    assert gemma.source == "app/ai/profiles/gemma-4-12b.toml"
    assert dict(gemma.sampling) == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    assert gemma.thinking_kwarg == "enable_thinking"
    for role in _ROLES:
        assert gemma.phase(role).max_tokens == 2048
        assert gemma.phase(role).temperature is None
        assert gemma.temperature_for(role) == 1.0
        assert not gemma.phase(role).thinking


def test_the_26b_profile_holds_the_think_lane_s_numbers() -> None:
    gemma = load_profile("gemma-4-26b-a4b")
    assert dict(gemma.sampling) == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    assert gemma.thinking_kwarg == "enable_thinking"
    assert gemma.phase("think").max_tokens == 6000
    assert gemma.phase("think").thinking is True


@pytest.mark.skipif(_FLEET_PROFILE is None, reason="needs the ../agent-standard sibling")
def test_the_12b_matches_the_fleet_profile() -> None:
    # Copied, never edited here: drift is fixed by re-copying.
    assert _FLEET_PROFILE is not None
    text = _FLEET_PROFILE.read_text(encoding="utf-8")
    fleet = {key: float(value) for key, value in re.findall(_FLEET_ROW, text)}
    gemma = load_profile("gemma-4-12b")
    assert fleet == {key: float(value) for key, value in dict(gemma.sampling).items()}
    assert f'"{gemma.thinking_kwarg}"' in text
    planner = re.search(_FLEET_PLANNER, text)
    assert planner is not None
    assert float(planner.group(1)) == gemma.temperature_for("planner")


def test_every_profile_file_loads_and_is_named_for_its_model() -> None:
    files = sorted(PROFILES_DIR.glob("*.toml"))
    assert files, "no profiles shipped"
    for path in files:
        profile = load_profile(path.stem)
        assert profile.name == path.stem
        assert profile is not DEFAULT_PROFILE


@pytest.mark.parametrize("setting", ["llamacpp_model", "conductor_think_model"])
def test_the_configured_default_models_have_profiles(setting: str) -> None:
    model = Settings.model_fields[setting].default
    assert (PROFILES_DIR / f"{model}.toml").is_file()


# --- strict loading --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "bogus = 1\n",  # unknown top-level key
        "[sampling]\ntemprature = 1.0\n",  # misspelt sampling key
        "[phases.turn]\nmax_tokens = 100\n",  # unknown phase
        "[phases.planner]\nmax_token = 100\n",  # misspelt phase key
        "[sampling]\ntop_k = 1.5\n",  # wrong type
        "[sampling]\ntop_p = 1.5\n",  # out of range
        "thinking_kwarg = true\n",  # wrong type
        "[phases.planner]\nmax_tokens = 0\n",
        # No turn role thinks, and the think call samples at [sampling]'s: a
        # key the code would not read is an error, never ignored.
        "[phases.planner]\nthinking = true\n",
        "[phases.narrator]\nthinking = true\n",
        "[phases.speaker]\nthinking_budget = 100\n",
        "[phases.think]\ntemperature = 0.3\n",
        "sampling = 3\n",  # a table that is not one
        "[sampling\n",  # not TOML
    ],
)
def test_a_profile_that_says_anything_unknown_or_mistyped_fails(text: str) -> None:
    with pytest.raises(ProfileError):
        parse_profile("broken", text, "test")


def test_an_empty_profile_is_the_default_settings_under_its_own_name() -> None:
    profile = parse_profile("blank", "", "test")
    assert profile.name == "blank"
    assert dict(profile.sampling) == {}
    assert profile.thinking_kwarg is None
    assert dict(profile.phases) == dict(DEFAULT_PROFILE.phases)


def test_the_default_profile_has_no_sampling_opinion_and_today_s_caps() -> None:
    assert dict(DEFAULT_PROFILE.sampling) == {}
    assert DEFAULT_PROFILE.thinking_kwarg == "enable_thinking"
    assert [DEFAULT_PROFILE.phase(role).max_tokens for role in _ROLES] == [2048] * 3
    think = DEFAULT_PROFILE.phase("think")
    assert (think.max_tokens, think.thinking) == (6000, True)


def test_a_model_with_no_profile_runs_on_the_default_and_says_so() -> None:
    load_profile.cache_clear()
    try:
        with structlog.testing.capture_logs() as logs:
            profile = load_profile("some-new-model")
    finally:
        load_profile.cache_clear()
    assert profile is DEFAULT_PROFILE
    assert profile.name == DEFAULT_NAME
    assert any(
        entry["event"] == "model_profile_missing"
        and entry["log_level"] == "warning"
        and entry["model"] == "some-new-model"
        for entry in logs
    )
