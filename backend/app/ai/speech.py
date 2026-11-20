"""Conductor's voice vocabulary and settings wiring over the fleet speech client.

The client itself — the OpenAI-audio wire calls, typed errors, factory — is
the vendored ``speech_client`` (canonical in ``../speech/client/python/``, per
the fleet voice contract ``../agent-standard/voice.md``). This module holds
what is conductor's own: the whisper prompt, bundled as ``VOCABULARY``, and the
settings-driven factory.
"""

from __future__ import annotations

import re
from datetime import date

from app import clock
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


_ISO_DATE = re.compile(r"(?<![\w-])(\d{4})-(\d{2})-(\d{2})(?![\w-])")


def _ordinal(day: int) -> str:
    if 10 <= day % 100 <= 20:
        return f"{day}th"
    return f"{day}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th') }"


def speakable(text: str) -> str:
    """Display text as it should be said: a date written 2026-11-22 becomes
    "November 22nd". Read as written, the voice says the digits and "minus"
    for the dashes. The year is said only when it isn't this one (household
    clock). Anything that isn't a real date is left alone."""
    this_year = clock.local_today().year

    def said(found: re.Match[str]) -> str:
        try:
            day = date(int(found.group(1)), int(found.group(2)), int(found.group(3)))
        except ValueError:
            return found.group(0)
        words = f"{day.strftime('%B')} {_ordinal(day.day)}"
        return words if day.year == this_year else f"{words}, {day.year}"

    return _ISO_DATE.sub(said, text)


VOCABULARY = Vocabulary(stt_prompt=STT_PROMPT, speakable=speakable)


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
