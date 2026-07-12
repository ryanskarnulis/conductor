from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_ignore_empty: an empty env var (e.g. a blank optional value in a
    # compose .env) is treated as unset and falls back to the default, rather
    # than being parsed as "" — which would crash a typed-optional field.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_ignore_empty=True
    )

    app_env: str = "development"

    # Host the API binds to. Constitution default is loopback-only; set to
    # "0.0.0.0" in .env to expose the API on the LAN.
    api_host: str = "127.0.0.1"

    # Port the dev API binds to (`python -m app.main`). 8301 sits in
    # conductor's 8300-8399 workspace block (8300 is the docker-published
    # frontend). The docker image binds its own container-internal port in
    # the Dockerfile CMD, independent of this.
    api_port: int = 8301

    # Explicit CORS allow-list (the local Vite dev server, port 5174 — 5173
    # is PCC's).
    cors_origins: list[str] = [
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]
    # Also allow the Vite dev server when loaded from a private-LAN address
    # (host IP may change via DHCP, so match the range rather than a fixed IP).
    cors_origin_regex: str | None = (
        r"http://(localhost|127\.0\.0\.1"
        r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|192\.168\.\d{1,3}\.\d{1,3}"
        r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}):5174"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
