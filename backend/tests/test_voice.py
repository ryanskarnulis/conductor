"""Voice endpoints: STT/TTS proxied through the shared speech service.

The browser never talks to the speech servers directly — it posts audio to
the app, which forwards it and hands back plain text destined for the same
agent pipeline as typed input. The speech layer speaks the OpenAI audio wire
format over plain httpx per the fleet contract (``../agent-standard/voice.md``);
tests inject an ``httpx.MockTransport``, so no live speech server (or audio
model) is ever required.
"""

from __future__ import annotations

import json
from collections.abc import Generator

import httpx
import pytest
from fastapi.testclient import TestClient

from app.ai.speech import (
    STT_PROMPT,
    SpeechClient,
    SpeechRequestError,
    SpeechResponseError,
    speech_client_from_settings,
)
from app.api import routes_voice
from app.config import get_settings
from app.main import app


class FakeSpeechServer:
    """An OpenAI-audio-shaped server behind ``httpx.MockTransport``.

    Records every request so tests can assert on the wire: multipart fields
    for /audio/transcriptions, the JSON body for /audio/speech.
    """

    def __init__(
        self,
        text: str = "ask tasks what's due today",
        audio: bytes = b"mp3-bytes",
        status: int = 200,
        body: dict[str, str] | None = None,
    ) -> None:
        self.text = text
        self.audio = audio
        self.status = status
        self.body = body  # overrides the transcription JSON body when set
        self.requests: list[httpx.Request] = []

    def _handler(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, text="upstream sad")
        if request.url.path.endswith("/audio/transcriptions"):
            body = self.body if self.body is not None else {"text": self.text}
            return httpx.Response(200, json=body)
        if request.url.path.endswith("/audio/speech"):
            return httpx.Response(200, content=self.audio, headers={"content-type": "audio/mpeg"})
        return httpx.Response(404)

    def client(self, base_url: str = "http://speech:8400/v1") -> httpx.Client:
        return httpx.Client(base_url=base_url, transport=httpx.MockTransport(self._handler))


# --- SpeechClient unit --------------------------------------------------------


def test_transcribe_forwards_audio_and_returns_text() -> None:
    server = FakeSpeechServer(text="ask chess to play e4")
    speech = SpeechClient(client=server.client(), stt_model="whisper-test")
    text = speech.transcribe(b"opus-bytes", filename="clip.webm")
    assert text == "ask chess to play e4"
    (request,) = server.requests
    assert request.url.path == "/v1/audio/transcriptions"
    body = request.read()
    assert b'filename="clip.webm"' in body
    assert b"opus-bytes" in body
    assert b"whisper-test" in body


def test_transcribe_biases_whisper_with_the_routing_vocabulary_prompt() -> None:
    server = FakeSpeechServer()
    speech = SpeechClient(client=server.client())
    speech.transcribe(b"opus-bytes")
    (request,) = server.requests
    assert STT_PROMPT.encode() in request.read()


def test_stt_prompt_covers_the_fleet_routing_vocabulary() -> None:
    # The prompt biases recognition toward conductor's routing phrasing —
    # crucially the fleet app names the router must hear correctly — and
    # whisper also mimics its formatting.
    for term in ("ask", "chess", "tasks", "due", "apps"):
        assert term in STT_PROMPT


def test_speak_forwards_text_and_returns_audio_bytes() -> None:
    server = FakeSpeechServer(audio=b"kokoro-mp3")
    speech = SpeechClient(client=server.client(), tts_model="tts-test", tts_voice="af_test")
    assert speech.speak("Asked chess.") == b"kokoro-mp3"
    (request,) = server.requests
    assert request.url.path == "/v1/audio/speech"
    assert json.loads(request.read()) == {
        "model": "tts-test",
        "voice": "af_test",
        "input": "Asked chess.",
        "response_format": "mp3",
    }


def test_speak_uses_the_dedicated_tts_client_when_given() -> None:
    stt_server = FakeSpeechServer()
    tts_server = FakeSpeechServer(audio=b"house-voice-mp3")
    speech = SpeechClient(client=stt_server.client(), tts_client=tts_server.client())
    assert speech.speak("Done.") == b"house-voice-mp3"
    assert stt_server.requests == []
    assert speech.transcribe(b"opus-bytes") == "ask tasks what's due today"
    assert len(tts_server.requests) == 1  # STT never touches the TTS client


# --- typed errors -------------------------------------------------------------


def test_transcribe_upstream_http_error_raises_request_error() -> None:
    speech = SpeechClient(client=FakeSpeechServer(status=500).client())
    with pytest.raises(SpeechRequestError):
        speech.transcribe(b"opus-bytes")


def test_transcribe_unreachable_server_raises_request_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(base_url="http://speech:8400/v1", transport=httpx.MockTransport(refuse))
    with pytest.raises(SpeechRequestError):
        SpeechClient(client=client).transcribe(b"opus-bytes")


def test_transcribe_malformed_body_raises_response_error() -> None:
    speech = SpeechClient(client=FakeSpeechServer(body={"nope": "x"}).client())
    with pytest.raises(SpeechResponseError):
        speech.transcribe(b"opus-bytes")


def test_speak_upstream_http_error_raises_request_error() -> None:
    speech = SpeechClient(client=FakeSpeechServer(status=503).client())
    with pytest.raises(SpeechRequestError):
        speech.speak("Done.")


# --- settings factory ---------------------------------------------------------


def test_factory_returns_none_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "speech_base_url", None)
    assert speech_client_from_settings() is None


def test_factory_builds_split_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "speech_base_url", "http://speech:8400/v1")
    monkeypatch.setattr(get_settings(), "tts_base_url", "http://kokoro:8410/v1")
    speech = speech_client_from_settings()
    assert speech is not None
    assert "speech:8400" in str(speech.client.base_url)
    assert speech.tts_client is not None
    assert "kokoro:8410" in str(speech.tts_client.base_url)
    speech.close()


def test_factory_single_backend_without_tts_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "speech_base_url", "http://speech:8400/v1")
    monkeypatch.setattr(get_settings(), "tts_base_url", None)
    speech = speech_client_from_settings()
    assert speech is not None
    assert speech.tts_client is None
    speech.close()


# --- API endpoints -------------------------------------------------------------


@pytest.fixture
def voice_client() -> Generator[TestClient, None, None]:
    """App client with a fake speech backend injected."""
    server = FakeSpeechServer()
    app.dependency_overrides[routes_voice.get_speech_client] = lambda: SpeechClient(
        client=server.client()
    )
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_transcribe_endpoint_returns_text(voice_client: TestClient) -> None:
    response = voice_client.post(
        "/api/voice/transcribe",
        files={"audio": ("clip.webm", b"opus-bytes", "audio/webm")},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "ask tasks what's due today"}


def test_speak_endpoint_returns_audio(voice_client: TestClient) -> None:
    response = voice_client.post("/api/voice/speak", json={"text": "Asked chess."})
    assert response.status_code == 200
    assert response.content == b"mp3-bytes"
    assert response.headers["content-type"] == "audio/mpeg"


def test_speak_rejects_blank_text(voice_client: TestClient) -> None:
    assert voice_client.post("/api/voice/speak", json={"text": "   "}).status_code == 422


def test_transcribe_without_speech_service_is_503() -> None:
    app.dependency_overrides[routes_voice.get_speech_client] = lambda: None
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/voice/transcribe",
                files={"audio": ("clip.webm", b"opus-bytes", "audio/webm")},
            )
        assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_speak_without_speech_service_is_503() -> None:
    app.dependency_overrides[routes_voice.get_speech_client] = lambda: None
    try:
        with TestClient(app) as client:
            response = client.post("/api/voice/speak", json={"text": "hi"})
        assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_transcribe_upstream_failure_is_502() -> None:
    server = FakeSpeechServer(status=500)
    app.dependency_overrides[routes_voice.get_speech_client] = lambda: SpeechClient(
        client=server.client()
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/voice/transcribe",
                files={"audio": ("clip.webm", b"opus-bytes", "audio/webm")},
            )
        assert response.status_code == 502
    finally:
        app.dependency_overrides.clear()


def test_speak_upstream_failure_is_502() -> None:
    server = FakeSpeechServer(status=503)
    app.dependency_overrides[routes_voice.get_speech_client] = lambda: SpeechClient(
        client=server.client()
    )
    try:
        with TestClient(app) as client:
            response = client.post("/api/voice/speak", json={"text": "hi"})
        assert response.status_code == 502
    finally:
        app.dependency_overrides.clear()


def test_voice_endpoints_are_rate_limited(
    voice_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "voice_requests_per_min", 1)
    ok = voice_client.post("/api/voice/speak", json={"text": "one"})
    assert ok.status_code == 200
    throttled = voice_client.post("/api/voice/speak", json={"text": "two"})
    assert throttled.status_code == 429
    assert "Retry-After" in throttled.headers
