from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The workspace root — the parent of the conductor repo — computed from this
# file's location so the dev default holds no matter where the repo is checked
# out. config.py lives at <repo>/backend/app/config.py, so parents[3] is the
# directory that contains the conductor repo and its sibling app clones (each
# with an app.yaml conductor discovers). Docker overrides this via
# FLEET_MANIFEST_DIR=/fleet (a read-only mount of that same directory).
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # env_ignore_empty: an empty env var (e.g. a blank optional value in a
    # compose .env) is treated as unset and falls back to the default, rather
    # than being parsed as "" — which would crash a typed-optional field.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_ignore_empty=True
    )

    app_env: str = "development"

    # SQLite file for conversations + the delegate-thread map, relative to
    # backend/ (where dev commands run). Docker overrides with an absolute
    # /data path backed by a volume.
    database_url: str = "sqlite:///../data/conductor.db"

    # Host the API binds to. Constitution default is loopback-only; set to
    # "0.0.0.0" in .env to expose the API on the LAN.
    api_host: str = "127.0.0.1"

    # Comma-separated IPs/CIDRs of trusted reverse proxies (the nginx
    # container's compose subnet, in docker). Only requests from these peers
    # get their client key from X-Forwarded-For; see app/api/request_ip.py.
    trusted_proxy_ips: str = ""

    # Per-IP requests/min on the one model-calling endpoint
    # (POST /agent/conversations/{id}/messages).
    agent_messages_per_min: int = 10

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

    # --- voice (shared workspace speech service) -----------------------------
    # ../speech/: Speaches STT on 8400, Kokoro-FastAPI TTS on 8410. Env var
    # names are the fleet voice contract's (../agent-standard/voice.md). Unset
    # speech_base_url = voice off: /api/voice endpoints answer 503, everything
    # else untouched.
    speech_base_url: str | None = None
    # TTS on its own server (the house-voice Kokoro container); unset means
    # speech_base_url serves both STT and TTS.
    tts_base_url: str | None = None
    stt_model: str = "Systran/faster-whisper-small"
    tts_model: str = "speaches-ai/Kokoro-82M-v1.0-ONNX"
    tts_voice: str = "af_heart"

    # Per-IP cap on the /voice endpoints, rate-limited like the agent surface.
    # STT/TTS round-trips are cheap CPU work but proxy to a shared service;
    # 30/min covers a lively hands-free conversation with headroom (the ack
    # beats during slow delegated turns ride this budget too).
    voice_requests_per_min: int = 30

    # Per-IP cap on the fleet action proxy (`app/api/routes_fleet.py`). Higher
    # than the agent surface because that is the whole point of it: one answer
    # per click, given by somebody working down a list at their own speed, with
    # no model turn behind any of them. Still capped — a page in a loop is a
    # page in a loop.
    fleet_actions_per_min: int = 120

    # --- fleet discovery + delegation ---------------------------------------
    # Where to find the fleet manifests: one `<app>/app.yaml` per sibling app.
    # Dev default is the workspace root (parent of this repo); docker sets
    # FLEET_MANIFEST_DIR=/fleet (a read-only mount of that directory).
    fleet_manifest_dir: Path = _WORKSPACE_ROOT

    # Rewrites the HOST half of every manifest's `upstream` (host:port) — the
    # port is always kept. Empty means "use the manifest host verbatim" (dev,
    # where apps are reachable on 127.0.0.1). Docker sets
    # FLEET_UPSTREAM_HOST=host.docker.internal so the container reaches apps
    # bound on the host.
    fleet_upstream_host: str = ""

    # Per-turn ceiling on how many times conductor may call any one app's
    # delegate tool within a single master loop run. Exceeding it is a domain
    # error the model sees, so it stops hammering one app and moves on.
    conductor_delegate_calls_per_turn_per_app: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
