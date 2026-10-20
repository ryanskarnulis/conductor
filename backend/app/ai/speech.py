"""Conductor's voice vocabulary and settings wiring over the fleet speech client.

The client itself — the OpenAI-audio wire calls, typed errors, factory — is
the vendored ``speech_client`` (canonical in ``../speech/client/python/``, per
the fleet voice contract ``../agent-standard/voice.md``). This module holds
what is conductor's own: the whisper prompt, bundled as ``VOCABULARY``, and the
settings-driven factory.
"""

from __future__ import annotations

from app.ai.speech_client import SpeechClient, Vocabulary, create_speech_client
from app.config import get_settings

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


VOCABULARY = Vocabulary(stt_prompt=STT_PROMPT)


def speech_client_from_settings() -> SpeechClient | None:
    """Build a ``SpeechClient`` from settings, or ``None`` when voice is off.

    Voice is optional exactly like the brain: no ``SPEECH_BASE_URL`` means no
    client and the voice endpoints answer 503; the rest of the app is
    untouched."""
    settings = get_settings()
    if not settings.speech_base_url:
        return None
    return create_speech_client(
        base_url=settings.speech_base_url,
        vocabulary=VOCABULARY,
        tts_base_url=settings.tts_base_url,
        stt_model=settings.stt_model,
        tts_model=settings.tts_model,
        tts_voice=settings.tts_voice,
    )
