"""Voice API: STT/TTS proxies to the shared workspace speech service.

The browser posts audio here and gets text back (``/voice/transcribe``), or
posts text and gets mp3 back (``/voice/speak``); the transcript then goes
down the exact same ``/agent`` pipeline as typed input — voice never gets its
own path to the fleet. Both endpoints are rate-limited per client IP like the
agent surface, answer 503 when no speech service is configured, and 502 when
the upstream fails. Nothing here touches the database or the delegate tools.
"""

from __future__ import annotations

from collections.abc import Generator

import structlog
from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, field_validator

from app.ai.speech import SpeechClient, SpeechError, speech_client_from_settings
from app.api.rate_limit import rate_limit

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/voice",
    tags=["voice"],
    dependencies=[Depends(rate_limit("voice", per_min_attr="voice_requests_per_min"))],
)


def get_speech_client() -> Generator[SpeechClient | None, None, None]:
    """The client per settings (None = voice off); tests override this."""
    speech = speech_client_from_settings()
    try:
        yield speech
    finally:
        if speech is not None:
            speech.close()


def _require(speech: SpeechClient | None) -> SpeechClient:
    if speech is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="voice unavailable: no speech service configured",
        )
    return speech


class SpeakRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class TranscriptionResult(BaseModel):
    text: str


@router.post("/transcribe", response_model=TranscriptionResult)
async def transcribe(
    audio: UploadFile,
    speech: SpeechClient | None = Depends(get_speech_client),
) -> TranscriptionResult:
    """STT proxy: browser audio in, plain text out.

    The client feeds the text into the same agent pipeline as typed input;
    this endpoint itself reads and writes nothing."""
    client = _require(speech)
    data = await audio.read()
    try:
        text = client.transcribe(data, filename=audio.filename or "audio.webm")
    except SpeechError as exc:
        logger.error("voice_transcribe_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"speech service error: {exc}",
        ) from exc
    logger.info("voice_transcribed", audio_bytes=len(data), text_length=len(text))
    return TranscriptionResult(text=text)


@router.post("/speak")
def speak(
    request: SpeakRequest,
    speech: SpeechClient | None = Depends(get_speech_client),
) -> Response:
    """TTS proxy: text in, mp3 out — the audio for whatever reply the client
    decided to voice (voice in → voice out; typed turns stay silent)."""
    client = _require(speech)
    try:
        audio = client.speak(request.text)
    except SpeechError as exc:
        logger.error("voice_speak_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"speech service error: {exc}",
        ) from exc
    logger.info("voice_spoken", text_length=len(request.text), audio_bytes=len(audio))
    return Response(content=audio, media_type="audio/mpeg")
