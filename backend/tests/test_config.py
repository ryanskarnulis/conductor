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
