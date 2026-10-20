# vendored — canonical: speech/client/python/speech_client.py; re-copy, never edit
"""Speech client: STT + TTS against the shared workspace speech service.

Speaches (STT) and Kokoro-FastAPI (TTS) both speak the OpenAI audio API, so
the client is two POSTs over plain httpx, no SDK: `/audio/transcriptions`
(multipart) and `/audio/speech` (JSON → mp3). The response body is validated
by a Pydantic wire model at the boundary; failures raise typed errors, never
best-effort parses. The httpx clients are injected in tests
(`httpx.MockTransport`), so live audio models never run in a suite.

App-agnostic on purpose: everything domain-specific arrives as a
`Vocabulary` (the STT prompt plus optional transcript/TTS text repairs), and
settings stay in the host app, which builds the client from its own config.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, ValidationError

# Speaches serves faster-whisper models by HF repo id; `small` is a good
# home-box default (CPU-friendly, solid English accuracy for short commands).
DEFAULT_STT_MODEL = "Systran/faster-whisper-small"
# Kokoro is Speaches' recommended TTS: ~82M params, natural voices, CPU-fine.
DEFAULT_TTS_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
DEFAULT_TTS_VOICE = "af_heart"

# STT/TTS of a short utterance is seconds even on CPU; only a cold model load
# on the speech server is slow. Generous read, short connect (a dead server
# fails fast).
TIMEOUT = httpx.Timeout(60.0, connect=10.0)


def _unchanged(text: str) -> str:
    return text


@dataclass(frozen=True)
class Vocabulary:
    """An app's domain layer over the generic client.

    `stt_prompt` is required by the voice standard: whisper conditions on it
    as if it preceded the audio, biasing recognition toward the app's phrases
    and mimicking their formatting. `normalize` repairs transcript slips with
    exactly one domain reading; `speakable` re-renders display text into what
    the TTS can say. Both default to passing text through unchanged."""

    stt_prompt: str
    normalize: Callable[[str], str] = _unchanged
    speakable: Callable[[str], str] = _unchanged


class SpeechError(Exception):
    """Base for everything a speech round-trip can raise."""


class SpeechRequestError(SpeechError):
    """No usable HTTP response: connect/timeout failure or a non-200 status."""


class SpeechResponseError(SpeechError):
    """The server answered 200 but the body failed validation."""


class _TranscriptionResponse(BaseModel):
    """The one field of the OpenAI transcription response the app uses."""

    text: str


@dataclass
class SpeechClient:
    """STT + TTS bound to OpenAI-compatible backends and model choices.

    One backend serves both by default; `tts_client` splits TTS onto its own
    server (the fleet house voice lives in a Kokoro-FastAPI container while
    STT stays on Speaches)."""

    client: httpx.Client
    vocabulary: Vocabulary
    stt_model: str = DEFAULT_STT_MODEL
    tts_model: str = DEFAULT_TTS_MODEL
    tts_voice: str = DEFAULT_TTS_VOICE
    tts_client: httpx.Client | None = None

    def transcribe(self, audio: bytes, filename: str = "audio.webm") -> str:
        """Audio bytes (any container whisper accepts; the browser sends
        webm/opus or wav) → plain text for the app's pipeline, biased by the
        vocabulary prompt and repaired by its normalizer. The filename's
        extension is how the backend sniffs the container format."""
        try:
            response = self.client.post(
                "audio/transcriptions",
                files={"file": (filename, audio)},
                data={"model": self.stt_model, "prompt": self.vocabulary.stt_prompt},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SpeechRequestError(f"transcription request failed: {exc}") from exc
        try:
            parsed = _TranscriptionResponse.model_validate_json(response.content)
        except ValidationError as exc:
            msg = f"malformed transcription response: {exc}"
            raise SpeechResponseError(msg) from exc
        return self.vocabulary.normalize(parsed.text)

    def speak(self, text: str) -> bytes:
        """Text → spoken audio bytes. mp3, because every browser `<audio>`
        plays it and it's small enough for LAN round-trips. The text passes
        through the vocabulary's `speakable` on the way out — the caller's
        string is display text."""
        client = self.tts_client if self.tts_client is not None else self.client
        try:
            response = client.post(
                "audio/speech",
                json={
                    "model": self.tts_model,
                    "voice": self.tts_voice,
                    "input": self.vocabulary.speakable(text),
                    "response_format": "mp3",
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SpeechRequestError(f"speech request failed: {exc}") from exc
        return response.content

    def close(self) -> None:
        self.client.close()
        if self.tts_client is not None:
            self.tts_client.close()


def create_speech_client(
    *,
    base_url: str,
    vocabulary: Vocabulary,
    tts_base_url: str | None = None,
    stt_model: str = DEFAULT_STT_MODEL,
    tts_model: str = DEFAULT_TTS_MODEL,
    tts_voice: str = DEFAULT_TTS_VOICE,
    client: httpx.Client | None = None,
    tts_client: httpx.Client | None = None,
) -> SpeechClient:
    """Build a SpeechClient against a real Speaches (e.g. host:8400/v1).

    `tts_base_url` points TTS at its own OpenAI-compatible server (the
    Kokoro-FastAPI house voice); without it the one backend serves both.
    `client`/`tts_client` inject fakes in tests; otherwise the factory builds
    real httpx clients."""
    if client is None:
        client = httpx.Client(base_url=base_url, timeout=TIMEOUT)
    if tts_client is None and tts_base_url:
        tts_client = httpx.Client(base_url=tts_base_url, timeout=TIMEOUT)
    return SpeechClient(
        client=client,
        vocabulary=vocabulary,
        stt_model=stt_model,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_client=tts_client,
    )
