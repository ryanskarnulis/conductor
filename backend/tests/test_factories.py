"""The settings-backed factories that wire config into the provider and the turn runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ai import trace as run_trace
from app.ai.interpreter import conductor_from_settings
from app.ai.providers.llamacpp import (
    ROLE_SAMPLING,
    provider_from_settings,
    role_sampling_from_settings,
)
from app.config import get_settings
from app.fleet.manifests import Fleet
from tests.scripted_provider import ScriptedProvider


def test_provider_from_settings_reads_configured_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://example:8200/v1")
    monkeypatch.setenv("LLAMACPP_MODEL", "test-model")
    get_settings.cache_clear()
    try:
        with provider_from_settings() as provider:
            assert provider._base_url == "http://example:8200/v1"
            assert provider._model == "test-model"
    finally:
        get_settings.cache_clear()


def test_conductor_from_settings_wires_the_cap_the_tracer_and_the_narrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONDUCTOR_MAX_TOKENS", "321")
    monkeypatch.setenv("CONDUCTOR_NARRATOR", "0")
    monkeypatch.setenv("CONDUCTOR_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    get_settings.cache_clear()
    try:
        conductor = conductor_from_settings(ScriptedProvider([]), Fleet(apps=()))
        assert conductor._rt.max_tokens == 321
        assert conductor._rt.narrate is False
        assert isinstance(conductor._rt.tracer, run_trace.JsonlTracer)
        assert conductor._rt.tracer.path == tmp_path / "trace.jsonl"
    finally:
        get_settings.cache_clear()


def test_conductor_from_settings_narrates_and_keeps_no_trace_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONDUCTOR_TRACE_PATH", raising=False)
    get_settings.cache_clear()
    try:
        conductor = conductor_from_settings(ScriptedProvider([]), Fleet(apps=()))
        assert conductor._rt.narrate is True
        assert conductor._rt.tracer is None
        assert conductor._rt.max_tokens == 2048
    finally:
        get_settings.cache_clear()


def test_every_role_samples_the_model_profile_unless_the_planner_is_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONDUCTOR_PLANNER_TEMPERATURE", raising=False)
    get_settings.cache_clear()
    try:
        assert role_sampling_from_settings() == ROLE_SAMPLING
        assert {(s.temperature, s.top_p, s.top_k) for s in ROLE_SAMPLING.values()} == {
            (1.0, 0.95, 64)
        }
        monkeypatch.setenv("CONDUCTOR_PLANNER_TEMPERATURE", "0")
        get_settings.cache_clear()
        sampling = role_sampling_from_settings()
        assert (sampling["planner"].temperature, sampling["planner"].top_p) == (0.0, 0.95)
        assert sampling["narrator"] == ROLE_SAMPLING["narrator"]
        assert sampling["speaker"] == ROLE_SAMPLING["speaker"]
    finally:
        get_settings.cache_clear()


def test_a_planner_temperature_out_of_range_fails_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONDUCTOR_PLANNER_TEMPERATURE", "3")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="CONDUCTOR_PLANNER_TEMPERATURE"):
            get_settings()
    finally:
        get_settings.cache_clear()
