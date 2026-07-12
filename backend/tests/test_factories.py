"""The settings-backed factories that wire config into the provider and loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ai.loop import loop_from_settings
from app.ai.providers.llamacpp import provider_from_settings
from app.config import get_settings
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


def test_loop_from_settings_uses_configured_max_iterations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONDUCTOR_MAX_ITERATIONS", "2")
    get_settings.cache_clear()
    try:
        loop = loop_from_settings(ScriptedProvider([]))
        assert loop._max_iterations == 2
    finally:
        get_settings.cache_clear()
