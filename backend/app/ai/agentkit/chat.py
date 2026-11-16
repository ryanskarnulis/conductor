# vendored — canonical: agent-standard/agentkit/chat.py; re-copy, never edit
"""The chat wire: what a completion is, and what can go wrong getting one.

Canonical in ``../agent-standard/agentkit/chat.py``; every app vendors it
verbatim as ``agentkit/chat.py`` inside its own package, with one
``# vendored …`` first line. It imports the standard library and pydantic, and
nothing from the app.

llama-server speaks the OpenAI chat API. This file is the half of a provider
that never varies: the response validated at the boundary, and the typed
errors a call can end in. Nothing is best-effort parsed.

- The errors (:class:`ProviderError` and its three kinds), each carrying a
  :class:`ProviderFailure`: whether asking again could work.
- What a call is offered and answers with: :class:`ToolSpec`,
  :class:`ToolCall`, :class:`ChatResult`, :class:`Usage`,
  :class:`ServerStamp`.
- Reading the response: :func:`parse_completion` for a body,
  :class:`StreamedTurn` for a stream, :func:`chat_result` for either.
  :func:`request_failed` and :func:`bad_status` build the errors of a call
  that got no usable body.

Two wire quirks are handled here. Chain-of-thought arrives in a separate
``reasoning_content`` field: it is validated, so an unexpected shape fails
loudly, but it is never answer text and never goes back into history
(:meth:`ChatResult.to_message` drops it). And what llama-server says about
itself (``timings``, ``model``, ``system_fingerprint``, the cache counts) is
read leniently: it is diagnostics, and a server build that shapes it
differently costs a record a field, never a run its call.

What the request says stays the app's: its payload, its HTTP flow, its
logging and its context capture.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class ProviderFailure(StrEnum):
    """Which kind of "no" a completion attempt got: a field, not a message.

    The exception type says where the failure happened; this says whether
    asking again could plausibly work, so a caller never has to read a message
    to tell a restarting server from a request the server refuses identically
    forever.
    """

    #: The socket never answered: connect refused, reset, timeout.
    UNREACHABLE = "unreachable"
    #: A 5xx. llama-swap's answer while the upstream model is restarting.
    SERVER_ERROR = "server_error"
    #: A 4xx: the server understood the request and refuses it. A context
    #: overrun is this, and the same bytes get the same refusal.
    REJECTED = "rejected"
    #: 200, but the body failed wire validation. Version skew, not a crash.
    MALFORMED_RESPONSE = "malformed_response"
    #: The model's tool-call arguments weren't a JSON object. A loop answers
    #: this with a correction, not by ending the run.
    BAD_TOOL_ARGUMENTS = "bad_tool_arguments"

    @property
    def transient(self) -> bool:
        """Whether re-sending the identical request could get a different
        answer. Only the two that mean "the server wasn't there"."""
        return self in _TRANSIENT_FAILURES


_TRANSIENT_FAILURES = frozenset({ProviderFailure.UNREACHABLE, ProviderFailure.SERVER_ERROR})


class ProviderError(Exception):
    """Base for everything a completion attempt can raise.

    ``failure`` defaults to UNREACHABLE so an unclassified failure (a test
    double, a raise site that forgets) reads as a dead server, which is the
    cheap mistake: retrying a deterministic failure wastes a few samples, while
    calling a restarting server deterministic aborts a whole eval suite.
    """

    def __init__(
        self, message: str, failure: ProviderFailure = ProviderFailure.UNREACHABLE
    ) -> None:
        super().__init__(message)
        self.failure = failure


class ProviderRequestError(ProviderError):
    """No usable HTTP response: connect/timeout failure or a non-200 status."""


class ProviderResponseError(ProviderError):
    """The server answered 200 but the body failed validation."""

    def __init__(
        self, message: str, failure: ProviderFailure = ProviderFailure.MALFORMED_RESPONSE
    ) -> None:
        super().__init__(message, failure)


class ToolCallArgumentsError(ProviderResponseError):
    """The model emitted tool-call arguments that aren't a JSON object.

    Carries the tool name so a correction turn can name the offender. The
    server did answer, so the call's cost is known: ``usage`` and ``server``
    carry it, ``None`` when not given.
    """

    def __init__(
        self,
        tool_name: str,
        detail: str,
        *,
        usage: Usage | None = None,
        server: ServerStamp | None = None,
    ) -> None:
        super().__init__(f"tool call {tool_name!r}: {detail}", ProviderFailure.BAD_TOOL_ARGUMENTS)
        self.tool_name = tool_name
        self.usage = usage
        self.server = server


def request_failed(exc: Exception) -> ProviderRequestError:
    """The error of a request that never got an answer."""
    return ProviderRequestError(f"llama-server request failed: {exc}", ProviderFailure.UNREACHABLE)


def bad_status(status_code: int, body: str) -> ProviderRequestError:
    """The error of a non-200 answer. A 5xx is the server having a bad moment;
    a 4xx is the server having read the request and refusing it, which it will
    do again identically."""
    failure = ProviderFailure.SERVER_ERROR if status_code >= 500 else ProviderFailure.REJECTED
    return ProviderRequestError(f"llama-server returned {status_code}: {body[:500]}", failure)


class ToolSpec(BaseModel):
    """One callable tool: name, description, JSON Schema for the arguments."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolCall(BaseModel):
    """A tool call with its arguments already parsed from the wire's JSON string."""

    id: str
    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    """The server's counts for one call. ``cached_tokens`` is how much of the
    prompt its cache held, ``None`` when it did not say: never a zero that
    reads as measured."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int | None = None


class ServerStamp(BaseModel):
    """The server's own account of one call.

    ``model`` is what the server says it ran (the weights behind the alias)
    and ``fingerprint`` its build. ``server_ms`` is its prompt plus generation
    time (the call's wall clock minus this is queueing, transport and any cold
    load) and ``prefill_ms`` the prompt part alone. ``prompt_tokens`` and
    ``cached_tokens`` are its slot's counts: the prompt tokens it processed
    and the ones its KV cache already held. Every field is ``None`` when the
    server did not say.
    """

    model: str | None = None
    fingerprint: str | None = None
    server_ms: int | None = None
    prefill_ms: float | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None


class WireFunction(BaseModel):
    name: str
    arguments: str | None = None


class WireToolCall(BaseModel):
    id: str = ""
    function: WireFunction


class WireMessage(BaseModel):
    content: str | None = None
    # The chain-of-thought channel: validated so unexpected shapes fail loudly,
    # but never surfaced as answer text and never sent back in history.
    reasoning_content: str | None = None
    tool_calls: list[WireToolCall] = []


class WireChoice(BaseModel):
    message: WireMessage
    finish_reason: str | None = None


class WireUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # OpenAI's place for the cache hit. Untyped: read leniently in ``usage``.
    prompt_tokens_details: Any = None


class WireCompletion(BaseModel):
    """One completion as the server sent it."""

    choices: list[WireChoice] = Field(min_length=1)
    usage: WireUsage | None = None
    # Diagnostics, deliberately untyped on the wire and read leniently below:
    # a mismatch here must never fail the completion.
    model: Any = None
    system_fingerprint: Any = None
    timings: Any = None

    def read_usage(self) -> Usage | None:
        """The call's counts, or ``None`` when the server sent no ``usage``.
        The cache hit is llama-server's ``timings.cache_n`` first, then
        OpenAI's ``prompt_tokens_details.cached_tokens``."""
        wire = self.usage
        if wire is None:
            return None
        details = wire.prompt_tokens_details
        cached = _whole(self._timings().get("cache_n"))
        if cached is None and isinstance(details, dict):
            cached = _whole(details.get("cached_tokens"))
        return Usage(
            prompt_tokens=wire.prompt_tokens,
            completion_tokens=wire.completion_tokens,
            total_tokens=wire.total_tokens,
            cached_tokens=cached,
        )

    def stamp(self) -> ServerStamp | None:
        """What the server said about itself, or ``None`` when it said nothing."""
        timings = self._timings()
        prompt_ms = _number(timings.get("prompt_ms"))
        predicted_ms = _number(timings.get("predicted_ms"))
        stamp = ServerStamp(
            model=_text(self.model),
            fingerprint=_text(self.system_fingerprint),
            server_ms=None
            if prompt_ms is None or predicted_ms is None
            else round(prompt_ms + predicted_ms),
            prefill_ms=prompt_ms,
            prompt_tokens=_whole(timings.get("prompt_n")),
            cached_tokens=_whole(timings.get("cache_n")),
        )
        return None if stamp == ServerStamp() else stamp

    def _timings(self) -> dict[str, Any]:
        return self.timings if isinstance(self.timings, dict) else {}


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _number(value: Any) -> float | None:
    """A JSON number that is not negative, or ``None`` (a bool is not one)."""
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        return None
    return float(value)


def _whole(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


class ChatResult(BaseModel):
    """One validated completion turn. ``content`` is answer text only."""

    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: Usage | None
    # The server's account of the call; never sent back in history. ``None``
    # from a server (or a test double) that said nothing.
    server: ServerStamp | None = None

    def to_message(self) -> dict[str, Any]:
        """This turn as an assistant message for the next request's history.

        Tool arguments are re-serialized to the wire's JSON-string form;
        ``reasoning_content`` deliberately never round-trips.
        """
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "type": "function",
                    "id": call.id,
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in self.tool_calls
            ]
        return message


def tool_result_message(tool_call_id: str, content: str) -> dict[str, Any]:
    """The ``role: tool`` message answering one :class:`ToolCall`."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def parse_completion(text: str) -> WireCompletion:
    """A 200 response's body, validated against the wire models."""
    try:
        return WireCompletion.model_validate_json(text)
    except ValidationError as exc:
        raise ProviderResponseError(f"llama-server response failed wire validation: {exc}") from exc


def chat_result(completion: WireCompletion) -> ChatResult:
    """The completion as the turn a loop reads: answer text, and each tool
    call's arguments parsed. Arguments that are not a JSON object are a
    :class:`ToolCallArgumentsError`; none at all is an empty object."""
    choice = completion.choices[0]
    calls: list[ToolCall] = []
    for wire_call in choice.message.tool_calls:
        raw = wire_call.function.arguments
        if raw is None or not raw.strip():
            arguments: Any = {}
        else:
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ToolCallArgumentsError(
                    wire_call.function.name,
                    f"arguments are not valid JSON ({exc.msg})",
                    usage=completion.read_usage(),
                    server=completion.stamp(),
                ) from exc
        if not isinstance(arguments, dict):
            raise ToolCallArgumentsError(
                wire_call.function.name,
                "arguments are not a JSON object",
                usage=completion.read_usage(),
                server=completion.stamp(),
            )
        calls.append(ToolCall(id=wire_call.id, name=wire_call.function.name, arguments=arguments))
    return ChatResult(
        content=choice.message.content,
        tool_calls=calls,
        finish_reason=choice.finish_reason,
        usage=completion.read_usage(),
        server=completion.stamp(),
    )


_DATA = "data:"


class StreamedTurn:
    """One streamed completion put back together, line by line.

    Feed it the event stream's lines with :meth:`feed`; :meth:`completion` is
    the turn in the shape a plain call's body has. ``first`` is the kind of
    the first meaningful delta: ``"content"`` (text that is not only
    whitespace) or ``"tool_calls"``; ``None`` until one arrives. Reasoning
    deltas are never meaningful: they are not answer text.
    """

    def __init__(self) -> None:
        self.first: str | None = None
        self.content = ""
        self.reasoning = ""
        self.calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.done = False
        self._extra: dict[str, Any] = {}

    def feed(self, line: str) -> None:
        """One line of the event stream. Lines that carry no data are skipped;
        ``[DONE]`` sets ``done``."""
        if not line.startswith(_DATA):
            return
        data = line[len(_DATA) :].strip()
        if data == "[DONE]":
            self.done = True
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                f"llama-server stream chunk is not JSON: {exc.msg}"
            ) from exc
        self.add(chunk)

    def add(self, chunk: Any) -> None:
        if not isinstance(chunk, dict):
            raise ProviderResponseError("llama-server stream chunk is not an object")
        for key in ("usage", "timings", "model", "system_fingerprint"):
            if chunk.get(key) is not None:
                self._extra[key] = chunk[key]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                raise ProviderResponseError("llama-server stream choice is not an object")
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise ProviderResponseError("llama-server stream delta is not an object")
            for part in delta.get("tool_calls") or []:
                self._add_call(part)
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str):
                self.reasoning += reasoning
            text = delta.get("content")
            if isinstance(text, str) and text:
                self.content += text
                if text.strip():
                    self.first = self.first or "content"

    def _add_call(self, part: Any) -> None:
        """One piece of a tool call: its id and name once, its arguments in parts."""
        function = part.get("function") or {} if isinstance(part, dict) else None
        if not isinstance(function, dict):
            raise ProviderResponseError("llama-server stream tool call is not an object")
        index = part.get("index", len(self.calls))
        name, arguments, call_id = function.get("name"), function.get("arguments"), part.get("id")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not isinstance(name, str | None)
            or not isinstance(arguments, str | None)
            or not isinstance(call_id, str | None)
        ):
            raise ProviderResponseError("llama-server stream tool call is malformed")
        self.first = self.first or "tool_calls"
        call = self.calls.setdefault(index, {"id": "", "function": {"name": "", "arguments": ""}})
        call["id"] = call_id or call["id"]
        call["function"]["name"] += name or ""
        call["function"]["arguments"] += arguments or ""

    def completion(self) -> WireCompletion:
        """The turn so far, validated as a plain call's body is."""
        body = {
            "choices": [
                {
                    "message": {
                        "content": self.content or None,
                        "reasoning_content": self.reasoning or None,
                        "tool_calls": [self.calls[index] for index in sorted(self.calls)],
                    },
                    "finish_reason": self.finish_reason,
                }
            ],
            **self._extra,
        }
        try:
            return WireCompletion.model_validate(body)
        except ValidationError as exc:
            raise ProviderResponseError(
                f"llama-server stream failed wire validation: {exc}"
            ) from exc


def body_text(body: bytes) -> str:
    """A wire body as the text it is. llama-server speaks UTF-8 both ways; a
    body that is not is kept with its bad bytes replaced, not lost."""
    return body.decode("utf-8", errors="replace")
