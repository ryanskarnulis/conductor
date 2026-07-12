"""Speech layer: STT + TTS via the shared workspace speech service.

Conductor's implementation of the fleet voice contract
(``../agent-standard/voice.md``; chess is the reference, PCC the closest
sibling — this module diffs against ``project-command-center``'s
``app/ai/speech.py`` with only the STT prompt as the delta). Speaches (STT)
and Kokoro-FastAPI (TTS) both speak the OpenAI audio API, so — like the
llama.cpp provider — the client is two POSTs over plain httpx, no SDK:
``/audio/transcriptions`` (multipart) and ``/audio/speech`` (JSON → mp3).
The response body is validated by a Pydantic wire model at the boundary;
failures raise typed errors, never best-effort parses. The httpx clients are
injected in tests (``httpx.MockTransport``), so live audio models never run
in the suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from pydantic import BaseModel, ValidationError

from app.config import get_settings

# STT/TTS of a short utterance is seconds even on CPU; only a cold model load
# on the speech server is slow. Generous read, short connect (a dead server
# fails fast) — same shape as the llama.cpp provider's timeout.
_TIMEOUT = httpx.Timeout(60.0, connect=10.0)

# Whisper conditions on this prompt as if it preceded the audio, biasing
# recognition toward conductor's vocabulary and mimicking its formatting:
# fleet routing phrasing and the app names the router must hear correctly.
# Required by the voice standard; changes gate on the routing eval harness
# (the prompt shapes what text enters the same /api/agent pipeline the evals
# exercise).
STT_PROMPT = (
    "Fleet routing commands: ask chess to play e4, ask tasks what's due "
    "today, what's overdue this week, add a task to buy milk tomorrow, "
    "start a new chess game, what's the board look like, move my knight, "
    "mark the task done, plan my day, which apps can I talk to."
)


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

    One backend serves both by default; ``tts_client`` splits TTS onto its
    own server (the fleet house voice lives in a Kokoro-FastAPI container
    while STT stays on Speaches)."""

    client: httpx.Client
    stt_model: str = "Systran/faster-whisper-small"
    tts_model: str = "speaches-ai/Kokoro-82M-v1.0-ONNX"
    tts_voice: str = "af_heart"
    stt_prompt: str = STT_PROMPT
    tts_client: httpx.Client | None = None

    def transcribe(self, audio: bytes, filename: str = "audio.webm") -> str:
        """Audio bytes (any container whisper accepts; the browser sends
        webm/opus or wav) → plain text destined for the agent pipeline,
        vocabulary-biased via the STT prompt. The filename's extension is how
        the backend sniffs the container format."""
        try:
            response = self.client.post(
                "audio/transcriptions",
                files={"file": (filename, audio)},
                data={"model": self.stt_model, "prompt": self.stt_prompt},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SpeechRequestError(f"transcription request failed: {exc}") from exc
        try:
            parsed = _TranscriptionResponse.model_validate_json(response.content)
        except ValidationError as exc:
            raise SpeechResponseError(f"malformed transcription response: {exc}") from exc
        return parsed.text

    def speak(self, text: str) -> bytes:
        """Text → spoken audio bytes. mp3, because every browser ``<audio>``
        plays it and it's small enough for LAN round-trips."""
        client = self.tts_client if self.tts_client is not None else self.client
        try:
            response = client.post(
                "audio/speech",
                json={
                    "model": self.tts_model,
                    "voice": self.tts_voice,
                    "input": text,
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


def speech_client_from_settings() -> SpeechClient | None:
    """Build a ``SpeechClient`` from settings, or ``None`` when voice is off.

    Voice is optional exactly like the brain: no ``SPEECH_BASE_URL`` means no
    client and the voice endpoints answer 503; the rest of the app is
    untouched."""
    settings = get_settings()
    if not settings.speech_base_url:
        return None
    tts_client = (
        httpx.Client(base_url=settings.tts_base_url, timeout=_TIMEOUT)
        if settings.tts_base_url
        else None
    )
    return SpeechClient(
        client=httpx.Client(base_url=settings.speech_base_url, timeout=_TIMEOUT),
        stt_model=settings.stt_model,
        tts_model=settings.tts_model,
        tts_voice=settings.tts_voice,
        tts_client=tts_client,
    )
