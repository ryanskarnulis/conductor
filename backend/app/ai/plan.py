"""Compiled plans (#70): the typed plan, its JSON schema, and the code checks.

The planner (the 12B under ``json_schema`` decoding, :mod:`app.ai.compiler`)
turns what the person said into one of three shapes:

- ``{"steps": [...]}``: typed steps over a fixed vocabulary of step kinds, the
  app a parameter (adding an app adds an enum value, never a tool);
- ``{"answer": "knowledge" | "decline" | "self" | "chat"}``: nothing needs to
  run;
- ``{"resume": "<their answer>"}``: the person answered the question a paused
  plan asked (offered only while one is paused on ``ask_user``).

Plans have no designed length. :data:`MAX_STEPS` is a ceiling against a
runaway, the same as the loop's per-run call cap.

Code owns what happens next. :func:`problems` is the validator: refs, order,
targets, policy and budgets; a plan with problems gets one repair, then the
turn falls back to the loop. :func:`fallback_reason` says whether the
interpreter can run a valid plan yet; v1 runs single-app ``ask``/``open``/
``ask_user``/``confirm`` plans and the direct answers, and everything else
(``look``, ``recall``, ``cancel``, ``amend``, ``undo``, ``think``, cross-app
plans, data flow between steps) falls back to the loop until its issue lands
(#72, #73, #74, #78, #80). Pure: no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from app.fleet.manifests import Fleet

# A ceiling against a runaway, not a design limit (decided 2026-11-06).
MAX_STEPS = 12
STEP_IDS = tuple(f"s{n}" for n in range(1, MAX_STEPS + 1))

AnswerKind = Literal["knowledge", "decline", "self", "chat"]
StepKind = Literal[
    "ask", "look", "open", "ask_user", "recall", "cancel", "amend", "confirm", "undo", "think"
]

# What interpreter v1 runs; the rest fall back to the loop.
RUNNABLE_KINDS: frozenset[str] = frozenset({"ask", "open", "ask_user", "confirm"})

_REF = re.compile(r"\{(s(?:[1-9]|1[0-2]))(?:\.[a-z_]+)?\}")


class Ask(BaseModel):
    id: str
    do: Literal["ask"]
    app: str
    request: str
    destructive: bool


class Look(BaseModel):
    id: str
    do: Literal["look"]
    app: str


class Open(BaseModel):
    id: str
    do: Literal["open"]
    app: str
    intent: str


class AskUser(BaseModel):
    id: str
    do: Literal["ask_user"]
    question: str
    options: list[str]


class Recall(BaseModel):
    id: str
    do: Literal["recall"]
    query: str


class Cancel(BaseModel):
    id: str
    do: Literal["cancel"]
    target: str


class Amend(BaseModel):
    id: str
    do: Literal["amend"]
    target: str
    change: str


class Confirm(BaseModel):
    id: str
    do: Literal["confirm"]
    answer: Literal["yes", "no"]


class Undo(BaseModel):
    id: str
    do: Literal["undo"]
    target: str


class Think(BaseModel):
    id: str
    do: Literal["think"]
    problem: str
    inputs: list[str]


Step = Annotated[
    Ask | Look | Open | AskUser | Recall | Cancel | Amend | Confirm | Undo | Think,
    Field(discriminator="do"),
]


class StepPlan(BaseModel):
    steps: list[Step]


class Direct(BaseModel):
    answer: AnswerKind


class Resume(BaseModel):
    resume: str


Plan = StepPlan | Direct | Resume
_PLAN: TypeAdapter[Plan] = TypeAdapter(Plan)


def parse(text: str) -> Plan:
    """The planner's JSON as a plan; raises ``ValueError`` when it isn't one."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return _PLAN.validate_json(text)


def dump(plan: Plan) -> dict[str, Any]:
    return plan.model_dump()


def load(body: dict[str, Any]) -> Plan:
    return _PLAN.validate_python(body)


# --- the schema ----------------------------------------------------------------------------


def schema(fleet: Fleet, *, resumable: bool = False) -> dict[str, Any]:
    """The ``json_schema`` the planner decodes under, from the fleet.

    Each step kind is an object with every field required and nothing else,
    which is what llama-server's strict grammar needs. ``resumable`` offers the
    ``resume`` shape, only while a plan is paused on ``ask_user``.
    """
    ask_apps = sorted(app.name for app in fleet.agent_apps())
    open_apps = sorted(app.name for app in fleet.open_apps())
    all_apps = sorted({*ask_apps, *open_apps})
    ids = {"type": "string", "enum": list(STEP_IDS)}

    def step(kind: str, **fields: dict[str, Any]) -> dict[str, Any]:
        properties = {"id": ids, "do": {"type": "string", "const": kind}, **fields}
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    text = {"type": "string"}
    kinds = []
    if ask_apps:
        kinds.append(
            step(
                "ask",
                app={"type": "string", "enum": ask_apps},
                request=text,
                destructive={"type": "boolean"},
            )
        )
    if all_apps:
        kinds.append(step("look", app={"type": "string", "enum": all_apps}))
    if open_apps:
        kinds.append(step("open", app={"type": "string", "enum": open_apps}, intent=text))
    kinds += [
        step("ask_user", question=text, options={"type": "array", "items": text}),
        step("recall", query=text),
        step("cancel", target=text),
        step("amend", target=text, change=text),
        step("confirm", answer={"type": "string", "enum": ["yes", "no"]}),
        step("undo", target=text),
        step("think", problem=text, inputs={"type": "array", "items": ids}),
    ]
    shapes: list[dict[str, Any]] = [
        {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {"anyOf": kinds},
                    "minItems": 1,
                    "maxItems": MAX_STEPS,
                }
            },
            "required": ["steps"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["knowledge", "decline", "self", "chat"]}
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    ]
    if resumable:
        shapes.append(
            {
                "type": "object",
                "properties": {"resume": text},
                "required": ["resume"],
                "additionalProperties": False,
            }
        )
    return {"anyOf": shapes}


# --- the validator ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Checks:
    """What the validator needs to know about now."""

    fleet: Fleet
    # Asks to one app in one turn (the delegate budget).
    calls_per_app: int = 3
    # Something a confirm step can answer: an app's question or conductor's own.
    confirmable: bool = False
    # A plan paused on ask_user, which a resume answers.
    resumable: bool = False
    # What the planner was shown about now (the state block and recall): an
    # ask_user may only offer options it names, never invented ones.
    context: str = ""


def problems(plan: Plan, checks: Checks) -> list[str]:
    """What code refuses to run, in words the planner can repair from."""
    if isinstance(plan, Direct):
        return []
    if isinstance(plan, Resume):
        if not checks.resumable:
            return ["resume with no question waiting"]
        return [] if plan.resume.strip() else ["resume with an empty answer"]
    steps = plan.steps
    found: list[str] = []
    if not steps:
        return ["a plan needs at least one step, or an answer"]
    if len(steps) > MAX_STEPS:
        found.append(f"{len(steps)} steps is over the ceiling of {MAX_STEPS}")
    seen: set[str] = set()
    asks: dict[str, int] = {}
    for index, step in enumerate(steps):
        expected = f"s{index + 1}"
        if step.id != expected:
            found.append(f"step {index + 1} has id {step.id}; ids go s1, s2, ... in order")
        for ref in _REF.findall(step_text(step)):
            if ref not in seen:
                found.append(f"{step.id} uses {{{ref}}} before {ref} runs")
        if isinstance(step, Think):
            for ref in step.inputs:
                if ref not in seen:
                    found.append(f"{step.id} thinks over {ref} before it runs")
        if isinstance(step, Ask):
            app = checks.fleet.get(step.app)
            if app is None or not app.has_agent:
                found.append(f"{step.id} asks {step.app}, which has no agent")
            asks[step.app] = asks.get(step.app, 0) + 1
            if not step.request.strip():
                found.append(f"{step.id} asks {step.app} nothing")
        if isinstance(step, Open):
            app = checks.fleet.get(step.app)
            if app is None or not app.has_open:
                found.append(f"{step.id} opens {step.app}, which can't be opened")
            if index != len(steps) - 1:
                found.append("open must be the last step")
        if isinstance(step, Look) and checks.fleet.get(step.app) is None:
            found.append(f"{step.id} looks at {step.app}, which isn't in the fleet")
        if isinstance(step, AskUser):
            if index != 0:
                found.append("ask_user must be the first step")
            invented = [option for option in step.options if not _named_in(option, checks.context)]
            if invented:
                found.append(
                    f"ask_user offers {', '.join(repr(o) for o in invented[:3])}, which State "
                    "doesn't name; to find which one they mean, ask the app instead"
                )
        if isinstance(step, Confirm):
            if index != 0:
                found.append("confirm must be the first step")
            if not checks.confirmable:
                found.append("confirm with nothing waiting on the person")
        seen.add(step.id)
    for app_name, count in asks.items():
        if count > checks.calls_per_app:
            found.append(
                f"{count} asks to {app_name}; at most {checks.calls_per_app} per turn — "
                "put them in one request"
            )
    return found


def fallback_reason(plan: Plan, *, confirm_app: str | None = None) -> str | None:
    """Why interpreter v1 can't run this valid plan yet, or ``None`` if it can.

    ``confirm_app`` is the app a ``confirm`` step answers, when it's an app's
    question.
    """
    if not isinstance(plan, StepPlan):
        return None
    apps: set[str] = set()
    asked_user = False
    for step in plan.steps:
        if step.do not in RUNNABLE_KINDS:
            return f"kind:{step.do}"
        if isinstance(step, AskUser):
            asked_user = True
        if isinstance(step, Ask | Open):
            apps.add(step.app)
        if isinstance(step, Confirm) and confirm_app is not None:
            apps.add(confirm_app)
        refs = set(_REF.findall(step_text(step)))
        if refs - ({"s1"} if asked_user else set()):
            return "data_flow"
    if len(apps) > 1:
        return "cross_app"
    return None


def _named_in(option: str, context: str) -> bool:
    """Whether State (or recall) names this option, ignoring case and quotes."""
    words = " ".join(option.lower().replace('"', " ").replace("'", " ").split())
    return bool(words) and words in " ".join(
        context.lower().replace('"', " ").replace("'", " ").split()
    )


def step_text(step: BaseModel) -> str:
    """Every free-text field of a step, for ref checks and the evals."""
    return " ".join(
        str(getattr(step, name))
        for name in ("request", "intent", "query", "question", "target", "change", "problem")
        if hasattr(step, name)
    )


def substitute(text: str, answers: dict[str, str]) -> str:
    """``{sN}`` replaced by what step N gave (v1: the person's answer to ask_user)."""
    return _REF.sub(lambda match: answers.get(match.group(1), match.group(0)), text)


PauseReason = Literal["ask_user", "gate", "app_confirm"]
PlanStatus = Literal["ran", "answered", "paused", "fell_back", "invalid"]


class TurnPlan(BaseModel):
    """What a turn did with its compiled plan, for the ledger and the trace."""

    body: dict[str, Any] | None = None
    status: PlanStatus
    # Why it fell back or was invalid, or where it stopped.
    note: str | None = None
    # Whether this plan is what acted (``False``: the loop took the turn).
    acted: bool = True
    # A paused plan: why, the 0-based index of the step it resumes at, and
    # the app a gate or confirm is about.
    pause: PauseReason | None = None
    resume_at: int | None = None
    pause_app: str | None = None
    ack: str | None = None
    repaired: bool = False


@dataclass(frozen=True)
class PausedPlan:
    """A compiled plan from the last turn, waiting on the person."""

    body: dict[str, Any]
    reason: PauseReason
    resume_at: int
    app: str | None = None


# --- the honest ack --------------------------------------------------------------------------


def ack(plan: Plan) -> str | None:
    """What code says at once for a slow plan: what's about to happen, never
    what's done. ``None`` for plans that answer fast (no app is asked)."""
    if not isinstance(plan, StepPlan):
        return None
    apps: list[str] = []
    for step in plan.steps:
        if isinstance(step, AskUser):
            return None  # the question is the reply, and it comes first
        if isinstance(step, Ask) and step.app not in apps:
            apps.append(step.app)
    if not apps:
        return None
    return f"On it, asking {' and '.join(apps)}."


def describe(plan: Plan) -> str:
    """One line for logs and the ledger: ``ask:tasks | open:chess``."""
    if isinstance(plan, Direct):
        return f"answer:{plan.answer}"
    if isinstance(plan, Resume):
        return "resume"
    return " | ".join(f"{step.do}:{getattr(step, 'app', '')}".rstrip(":") for step in plan.steps)
