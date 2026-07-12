from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # chdir to a dir with no .env so only the typed defaults are in play.
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.app_env == "development"
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 8301


def test_empty_optional_env_var_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A compose .env may supply a var with an empty value (e.g. APP_ENV=).
    # Without env_ignore_empty this would override the typed default with "".
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APP_ENV", "")

    settings = Settings()

    assert settings.app_env == "development"  # empty env ignored -> default wins


def test_llm_and_loop_settings_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.llamacpp_base_url == "http://127.0.0.1:8200/v1"
    assert settings.llamacpp_model == "gemma-4-12b"
    assert settings.llamacpp_timeout_seconds == 300.0
    # Conductor's loop is deliberately shallow (see CLAUDE.md).
    assert settings.conductor_max_iterations == 6


def test_llm_and_loop_settings_env_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://host.docker.internal:8200/v1")
    monkeypatch.setenv("LLAMACPP_MODEL", "some-other-model")
    monkeypatch.setenv("LLAMACPP_TIMEOUT_SECONDS", "42.5")
    monkeypatch.setenv("CONDUCTOR_MAX_ITERATIONS", "3")

    settings = Settings()

    assert settings.llamacpp_base_url == "http://host.docker.internal:8200/v1"
    assert settings.llamacpp_model == "some-other-model"
    assert settings.llamacpp_timeout_seconds == 42.5
    assert settings.conductor_max_iterations == 3
