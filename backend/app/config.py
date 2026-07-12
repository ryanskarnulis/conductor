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

    # --- LLM provider (shared gemma-4-12b behind llama-swap) -----------------
    # These are the connection settings; the sampling/thinking values that are
    # model knowledge live in the provider with a pointer to
    # ../agent-standard/model-profile.md (the canonical source).
    llamacpp_base_url: str = "http://127.0.0.1:8200/v1"
    llamacpp_model: str = "gemma-4-12b"
    # One generous read timeout: a cold model load through llama-swap is ~100s
    # before the first byte; warm calls never approach it.
    llamacpp_timeout_seconds: float = 300.0

    # Conductor's loop is deliberately shallower than the app loops' ~10: each
    # iteration may wrap a full subagent loop (a delegate call fans out into
    # that app's own bounded loop), so latency stacks. See CLAUDE.md.
    conductor_max_iterations: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
