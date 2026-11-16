# vendored — canonical: agent-standard/agentkit/model_profile.py; re-copy, never edit
"""Model profiles: what an agent knows about one model, kept out of the code.

Canonical in ``../agent-standard/agentkit/model_profile.py``; every app vendors
it verbatim as ``agentkit/model_profile.py`` inside its own package, with one
``# vendored …`` first line. Standard library only, and no import from the app.

A profile is one TOML file named for the llama-swap model id it describes. It
holds what changes with the model and nothing else:

- ``thinking_kwarg`` — the ``chat_template_kwargs`` key that turns the model's
  thinking channel on and off. Absent (or ``""``) for a model with no toggle:
  the request then carries no ``chat_template_kwargs`` at all.
- ``effort_kwarg`` — the ``chat_template_kwargs`` key a model's template reads
  for how hard to think, or absent for a model with no such levels.
- ``[sampling]`` — the request's ``temperature``, ``top_p`` and ``top_k``. A
  key left out is a field the request leaves out, so the server's own default
  applies.
- ``[phases.<phase>]`` — one table per model phase the app has. Every phase
  takes ``max_tokens``, the per-call cap. The app's :class:`ProfileSpec` says
  which of the other keys each phase takes: ``temperature`` (the phase's own,
  over ``[sampling]``'s), ``thinking``, ``thinking_budget`` (the most thinking
  tokens one call may spend), ``effort`` (the level sent under
  ``effort_kwarg``), ``deadline_s`` (the wall clock a whole run may take), and
  any field the app adds by subclassing :class:`PhaseSettings`.
- ``crutches`` — guidance that exists only because this model needed it, by
  name. The app's spec lists the names it knows.
- ``quirks`` — standing measured facts about the model, as text for the reader.

What varies by app lives in the app, in a :class:`ProfileSpec`: the phase
names and their defaults, the keys each phase takes, the crutch names, and an
upper bound on ``max_tokens`` if the app's context budget sets one. The app
also owns where the files are, the cache, and the warning for a model with no
file (:meth:`ProfileSpec.load` returns ``None`` for one).

Loading is strict: an unknown key, phase or crutch, or a value of the wrong
type, is a :class:`ProfileError`, because a typo in a profile is a model
quietly run on settings nobody chose. That includes a key the app's loop would
not read: a phase takes only the keys its spec names.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Protocol

# What a profile is named when no file described the model.
DEFAULT_NAME = "default"

# Sampling keys in the order the request carries them.
SAMPLING_KEYS = ("temperature", "top_p", "top_k")

_TOP_LEVEL_KEYS = frozenset(
    {"sampling", "thinking_kwarg", "effort_kwarg", "phases", "crutches", "quirks"}
)


class ProfileError(ValueError):
    """A profile file that does not say what a profile must."""


class ProfileDir(Protocol):
    """Where profile files live: a ``pathlib.Path`` or an ``importlib.resources``
    directory."""

    def __truediv__(self, name: str) -> Any: ...


@dataclass(frozen=True)
class PhaseSettings:
    """One phase's knobs. ``temperature=None`` samples at the profile's own.

    An app with knobs of its own subclasses this (a frozen dataclass too) and
    adds fields whose defaults are a ``bool`` or a whole number; a profile may
    then set them on the phases whose spec names them.
    """

    max_tokens: int
    temperature: float | None = None
    thinking: bool = False
    thinking_budget: int | None = None
    effort: str | None = None
    deadline_s: float | None = None


# What every phase's entry in ``describe()`` carries, whatever its spec.
_DESCRIBED = ("max_tokens", "temperature", "thinking")


@dataclass(frozen=True)
class ModelProfile:
    """One model's settings, resolved. ``source`` says where they came from."""

    name: str
    source: str
    # Only the keys the request sends; an absent key is the server's default.
    sampling: Mapping[str, float | int] = field(default_factory=lambda: MappingProxyType({}))
    thinking_kwarg: str | None = "enable_thinking"
    effort_kwarg: str | None = None
    phases: Mapping[str, PhaseSettings] = field(default_factory=lambda: MappingProxyType({}))
    # The keys each phase's table may set beyond ``max_tokens``, from the spec.
    phase_keys: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )
    crutches: frozenset[str] = frozenset()
    quirks: tuple[str, ...] = ()

    def phase(self, name: str) -> PhaseSettings:
        return self.phases[name]

    def temperature_for(self, phase: str) -> float | int | None:
        """What ``phase`` samples at: its own temperature, else the profile's,
        else ``None`` — the server's."""
        own = self.phases[phase].temperature
        return own if own is not None else self.sampling.get("temperature")

    def with_phase(self, name: str, **changes: Any) -> ModelProfile:
        """This profile with one phase's settings changed: an eval arm, or an
        override from the environment. Not checked against the spec."""
        phases = {**self.phases, name: replace(self.phases[name], **changes)}
        return replace(self, phases=MappingProxyType(phases))

    def describe(self) -> dict[str, Any]:
        """The profile as a manifest, a log line or an eval harness prints it."""
        phases: dict[str, dict[str, Any]] = {}
        for name, settings in self.phases.items():
            extra = [key for key in self.phase_keys.get(name, ()) if key not in _DESCRIBED]
            phases[name] = {key: getattr(settings, key) for key in (*_DESCRIBED, *extra)}
        return {
            "name": self.name,
            "source": self.source,
            "sampling": dict(self.sampling),
            "thinking_kwarg": self.thinking_kwarg,
            "effort_kwarg": self.effort_kwarg,
            "phases": phases,
            "crutches": sorted(self.crutches),
            "quirks": list(self.quirks),
        }


@dataclass(frozen=True)
class ProfileSpec:
    """What one app's profiles may say.

    ``phases`` names the app's phases and gives each its defaults: what a model
    with no file runs on, and what a file that leaves a key out gets.
    ``phase_keys`` lists, per phase, the keys its table may set beyond
    ``max_tokens``; a phase left out takes ``max_tokens`` alone.
    """

    phases: Mapping[str, PhaseSettings]
    phase_keys: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    known_crutches: frozenset[str] = frozenset()
    # The most any phase's ``max_tokens`` may be, when the app's context budget
    # reserves only so much for a completion.
    max_tokens_limit: int | None = None

    def __post_init__(self) -> None:
        for phase, keys in self.phase_keys.items():
            if phase not in self.phases:
                raise ValueError(f"phase_keys names {phase!r}, which is not a phase")
            known = {f.name for f in fields(self.phases[phase])}
            unknown = set(keys) - known
            if unknown:
                raise ValueError(f"phase {phase!r}: no such settings {sorted(unknown)}")

    def default_profile(self) -> ModelProfile:
        """What a model with no file runs on: no sampling opinion, the spec's
        phase defaults, the common thinking toggle, no crutches."""
        return ModelProfile(
            name=DEFAULT_NAME,
            source="built in",
            phases=MappingProxyType(dict(self.phases)),
            phase_keys=self._keys(),
        )

    def parse(self, name: str, text: str, source: str) -> ModelProfile:
        """A profile from its TOML text. Strict: anything unknown or mistyped fails."""
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ProfileError(f"profile {name!r}: {exc}") from exc
        fail = _Failer(name, source)
        _only(fail, "profile", data, _TOP_LEVEL_KEYS)

        thinking_kwarg = _text(fail, "thinking_kwarg", data.get("thinking_kwarg", ""))
        effort_kwarg = _text(fail, "effort_kwarg", data.get("effort_kwarg", ""))
        phases_data = data.get("phases", {})
        _only(fail, "phases", phases_data, self.phases.keys())
        phases = dict(self.phases)
        for phase, table in phases_data.items():
            phases[phase] = self._phase(fail, phase, table, bool(effort_kwarg))

        crutches = frozenset(_texts(fail, "crutches", data.get("crutches", [])))
        unknown = crutches - self.known_crutches
        if unknown:
            raise fail(f"unknown crutches {sorted(unknown)}")
        return ModelProfile(
            name=name,
            source=source,
            sampling=_sampling(fail, data.get("sampling", {})),
            thinking_kwarg=thinking_kwarg or None,
            effort_kwarg=effort_kwarg or None,
            phases=MappingProxyType(phases),
            phase_keys=self._keys(),
            crutches=crutches,
            quirks=_texts(fail, "quirks", data.get("quirks", [])),
        )

    def load(self, directory: ProfileDir, model: str, shown_as: str) -> ModelProfile | None:
        """The profile in ``directory`` for ``model``, or ``None`` when no file
        describes it. ``shown_as`` is the directory as ``source`` should name it."""
        path = directory / f"{model}.toml"
        if not path.is_file():
            return None
        return self.parse(model, path.read_text(encoding="utf-8"), f"{shown_as}/{model}.toml")

    def _keys(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {phase: tuple(self.phase_keys.get(phase, ())) for phase in self.phases}
        )

    def _phase(
        self, fail: _Failer, phase: str, table: Any, has_effort_kwarg: bool
    ) -> PhaseSettings:
        where = f"phases.{phase}"
        _only(fail, where, table, {"max_tokens", *self.phase_keys.get(phase, ())})
        default = self.phases[phase]
        set_: dict[str, Any] = {}
        for key, value in table.items():
            at = f"{where}.{key}"
            if key == "max_tokens":
                set_[key] = _whole(fail, at, value, least=1, most=self.max_tokens_limit)
            elif key == "temperature":
                set_[key] = _number(fail, at, value, least=0)
            elif key == "deadline_s":
                if _number(fail, at, value, least=0) == 0:
                    raise fail(f"{at} must be a positive number")
                set_[key] = float(value)
            elif key == "effort":
                if not isinstance(value, str) or not value:
                    raise fail(f"{at} must be a level name")
                if not has_effort_kwarg:
                    # A level with no key to send it under would be silently dropped.
                    raise fail(f"{at} needs effort_kwarg")
                set_[key] = value
            elif key == "thinking_budget":
                set_[key] = _whole(fail, at, value, least=1)
            elif isinstance(getattr(default, key), bool):
                if not isinstance(value, bool):
                    raise fail(f"{at} must be a bool")
                set_[key] = value
            else:
                set_[key] = _whole(fail, at, value, least=0)
        settings = replace(default, **set_)
        budget = settings.thinking_budget
        if budget is not None and budget >= settings.max_tokens:
            # A budget at or past the cap is no budget: the cap cuts the
            # thought off first, and the reply ends without an answer.
            raise fail(f"{where}.thinking_budget must be below max_tokens ({settings.max_tokens})")
        return settings


class _Failer:
    """Builds this profile's errors, so each names the profile and its source."""

    def __init__(self, name: str, source: str) -> None:
        self._prefix = f"profile {name!r} ({source})"

    def __call__(self, message: str) -> ProfileError:
        return ProfileError(f"{self._prefix}: {message}")


def _only(fail: _Failer, where: str, table: Any, allowed: Any) -> None:
    if not isinstance(table, dict):
        raise fail(f"{where} must be a table")
    extra = set(table) - set(allowed)
    if extra:
        raise fail(f"unknown {where} keys {sorted(extra)}")


def _text(fail: _Failer, where: str, value: Any) -> str:
    if not isinstance(value, str):
        raise fail(f"{where} must be a string")
    return value


def _texts(fail: _Failer, where: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise fail(f"{where} must be a list of strings")
    return tuple(value)


def _number(fail: _Failer, where: str, value: Any, *, least: float) -> float | int:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < least:
        raise fail(f"{where} must be a number, {least} or more")
    return value


def _whole(fail: _Failer, where: str, value: Any, *, least: int, most: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise fail(f"{where} must be a whole number, {least} or more")
    if most is not None and value > most:
        raise fail(f"{where} must be at most {most}")
    return value


def _sampling(fail: _Failer, table: Any) -> Mapping[str, float | int]:
    _only(fail, "sampling", table, SAMPLING_KEYS)
    sampling: dict[str, float | int] = {}
    for key in SAMPLING_KEYS:
        if key not in table:
            continue
        where = f"sampling.{key}"
        if key == "top_k":
            sampling[key] = _whole(fail, where, table[key], least=0)
        else:
            value = _number(fail, where, table[key], least=0)
            if key == "top_p" and not 0 < value <= 1:
                raise fail(f"{where} must be above 0 and at most 1")
            sampling[key] = value
    return MappingProxyType(sampling)
