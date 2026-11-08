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
interpreter can run a valid plan yet: it runs ``ask``/``open``/``ask_user``/
``confirm`` plans across apps, with ``{sN}`` data flow between them (#72),
``look`` (#73: a plain read of an app's glance, offered only for apps whose
manifest declares one), the control steps ``cancel``/``amend``/``undo`` on
their own (#74; on a running job through the jobs runtime, #76), and the
direct answers; every
other kind (``recall``, ``think``) falls back to the loop until its issue
lands (#78, #80). :func:`reads_only` says which asks may run in parallel.
Pure: no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Collection
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

# What the interpreter runs; the rest fall back to the loop.
RUNNABLE_KINDS: frozenset[str] = frozenset({"ask", "look", "open", "ask_user", "confirm"})
# Control steps (#74): they act on what's waiting or on what just ran, so each
# runs as a plan of its own (:mod:`app.ai.control`).
CONTROL_KINDS: frozenset[str] = frozenset({"cancel", "amend", "undo"})

# ``{sN}`` (or ``{sN.field}``): what step N touched, filled in by code (#72).
REF = re.compile(r"\{(s(?:[1-9]|1[0-2]))(?:\.[a-z_]+)?\}")
# The free-text fields a step can carry refs in.
STEP_TEXT_FIELDS = ("request", "intent", "query", "question", "target", "change", "problem")


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
    look_apps = sorted(app.name for app in fleet.glance_apps())
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
    if look_apps:
        kinds.append(step("look", app={"type": "string", "enum": look_apps}))
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
    found += _lost_refs(steps)
    for index, step in enumerate(steps):
        expected = f"s{index + 1}"
        if step.id != expected:
            found.append(f"step {index + 1} has id {step.id}; ids go s1, s2, ... in order")
        for ref in REF.findall(step_text(step)):
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
        if isinstance(step, Look):
            found += _look_problems(step, checks.fleet)
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


def _look_problems(step: Look, fleet: Fleet) -> list[str]:
    """A look needs an app with a glance. When it has none, the repair says
    what to do instead: the planner writes the words an ask needs, so code
    never invents them."""
    app = fleet.get(step.app)
    if app is None:
        return [f"{step.id} looks at {step.app}, which isn't in the fleet"]
    if app.has_glance:
        return []
    instead = (
        f"ask {step.app} instead"
        if app.has_agent
        else f"open {step.app} instead"
        if app.has_open
        else "leave it out"
    )
    return [f"{step.id} looks at {step.app}, which has no glance to read; {instead}"]


# A step named without braces ("the album from step s1"), which code can't fill.
_BARE_REF = re.compile(r"(?<![{\w])(?:step\s+)?(s(?:[1-9]|1[0-2]))(?![\w}])", re.IGNORECASE)
# A thing an ask will find out, described rather than named: "the new Caribou
# album", "their other album", "this playlist <url>". Only that step will know
# its real title. (An ask that names it, "the song Odessa", gives no trouble.)
_NOUNS = r"(?P<noun>album|song|track|playlist|record|ep|episode|mix|single)s?"
_UNNAMED = re.compile(
    r"\b(?:(?:the|their|his|her|its|my|that)\s+(?:\w+\s+){0,2}?(?:new|newest|latest|recent|"
    r"other|next)\s+(?:[\w'’]+\s+){0,3}?|(?:this|that)\s+)" + _NOUNS + r"\b",
    re.IGNORECASE,
)


# An ask that leaves the person's pointer unresolved: the app can't see State.
_UNRESOLVED = re.compile(
    r"\b(?:(?:their|his|her)\s+(?:[\w'’]+\s+){0,2}?|(?:the|my)\s+(?:other|next)\s+)"
    + _NOUNS
    + r"\b(?!\s+(?:by|from|of)\b)",
    re.IGNORECASE,
)


def _described(noun: str) -> re.Pattern[str]:
    """A later step pointing at that thing by description: "the album I just
    downloaded", "the Caribou album"."""
    return re.compile(
        rf"\b(?:the|that|this|my|their|its)\s+(?:[\w'’-]+\s+){{0,3}}?{noun}s?\b", re.IGNORECASE
    )


def _lost_refs(steps: list[Step]) -> list[str]:
    """Data flow the planner meant but didn't write as ``{sN}`` (#72): a bare
    ``s1``, or a later step describing what an earlier ask will find ("the
    album I just downloaded"). Either way the real title would be lost; the
    repair asks for the ref the prompt already teaches."""
    found: list[str] = []
    earlier: list[Step] = []
    for step in steps:
        text = step_text(step)
        bare = False
        if isinstance(step, Ask):
            unresolved = _UNRESOLVED.search(step.request)
            if unresolved is not None:
                phrase = unresolved.group(0)
                noun = unresolved.group("noun")
                found.append(
                    f'{step.id} asks {step.app} about "{phrase}", but {step.app} never sees '
                    f"State or this conversation, so it can't tell whose {noun} that is. "
                    f'Write the artist or title State names, like "the other {noun} by '
                    f'<the artist in State>"'
                )
        for match in _BARE_REF.finditer(text if isinstance(step, Ask | Open) else ""):
            ref = match.group(1).lower()
            if any(prior.id == ref for prior in earlier):
                found.append(f"{step.id} names {ref} without braces; write {{{ref}}}")
                bare = True
        if isinstance(step, Ask | Open) and not bare and not REF.search(text):
            for prior in earlier:
                # Several similar asks to one app, or one ask for a list of
                # things, aren't a lost ref: each names its own.
                if not isinstance(prior, Ask) or getattr(step, "app", None) == prior.app:
                    continue
                if "," in prior.request or " and " in prior.request:
                    continue
                unnamed = _UNNAMED.search(prior.request)
                if unnamed is None:
                    continue
                described = _described(unnamed.group("noun").lower()).search(text)
                if described is not None:
                    found.append(
                        f'{step.id} says "{described.group(0)}", which only {prior.id} will '
                        f"find out: write {{{prior.id}}} for its real name"
                    )
                    break
        earlier.append(step)
    return found


def fallback_reason(
    plan: Plan,
    *,
    confirm_app: str | None = None,
    pending: bool = False,
) -> str | None:
    """Why the interpreter can't run this valid plan yet, or ``None`` if it can.

    Cross-app plans run (#72): writes in order, one at a time. A ``look`` runs
    (#73) but gives nothing to ``{sN}``: a glance is several things, not a
    name, so a ref to one falls back. ``{sN}`` runs
    when step N can give a name: an ``ask`` (its receipt's title) or the
    ``ask_user`` (the person's answer). ``confirm_app`` is the app a
    ``confirm`` step answers, when it's an app's question; kept for the
    callers, it no longer changes the answer.

    A control step (#74) runs when it is the whole plan. ``cancel`` and
    ``amend`` act on what's ``pending`` (a paused plan or an app's question),
    or on a running job (#76, :mod:`app.ai.job_control`); ``pending`` is kept
    for the callers and no longer changes the answer.
    """
    del confirm_app, pending
    if not isinstance(plan, StepPlan):
        return None
    control = next((step for step in plan.steps if step.do in CONTROL_KINDS), None)
    if control is not None:
        if len(plan.steps) > 1:
            return f"kind:{control.do}"
        return None
    namers: set[str] = set()
    for step in plan.steps:
        if step.do not in RUNNABLE_KINDS:
            return f"kind:{step.do}"
        if set(REF.findall(step_text(step))) - namers:
            return "data_flow"
        if isinstance(step, Ask | AskUser):
            namers.add(step.id)
    return None


def has_control(plan: Plan) -> bool:
    """Whether a plan has a control step (``cancel``, ``amend``, ``undo``)."""
    return isinstance(plan, StepPlan) and any(step.do in CONTROL_KINDS for step in plan.steps)


_JOB_WORD = re.compile(r"\bjobs?\b", re.IGNORECASE)


def names_job(target: str, running_jobs: Collection[str]) -> bool:
    """Whether a control step's target points at a job: it says "job", or
    names a running one's ref (``music:31``, ``music:job:31`` and ``31`` all
    name ``music:31``)."""
    if _JOB_WORD.search(target):
        return True
    words = set(re.findall(r"[\w-]+", target.lower()))
    return any(job.rsplit(":", 1)[-1].lower() in words for job in running_jobs)


# What a request that only reads looks like: a question, with no word that
# asks for a change. Conservative on purpose: a read taken for a write only
# loses parallelism; a write taken for a read could run out of order.
_QUESTION = re.compile(
    r"^\s*(what|what's|whats|which|when|where|who|how|is|are|was|were|do|does|did|have|has|"
    r"any|anything|show|list|tell|check|find|look|search|see|read)\b",
    re.IGNORECASE,
)
_WRITE_WORDS = re.compile(
    r"\b(add|adds|create|make|new|put|delete|remove|trash|wipe|clear|move|push|reschedule|"
    r"rename|change|update|edit|set|mark|complete|finish|done|check off|tick|download|grab|"
    r"get|queue|fetch|save|sort|file|tag|cancel|stop|pause|resume|start|play|undo|restore|"
    r"archive|assign|schedule|remind|import|sync|retry|reset|resign)\b",
    re.IGNORECASE,
)


def reads_only(step: BaseModel) -> bool:
    """Whether a step only reads, so it may run alongside other reads (#72):
    a non-destructive ``ask`` with no refs, phrased as a question, with no
    word that asks for a change."""
    if not isinstance(step, Ask) or step.destructive:
        return False
    request = step.request.strip()
    if not request or REF.search(request):
        return False
    if not (_QUESTION.match(request) or request.endswith("?")):
        return False
    return _WRITE_WORDS.search(request) is None


def _named_in(option: str, context: str) -> bool:
    """Whether State (or recall) names this option, ignoring case and quotes."""
    words = " ".join(option.lower().replace('"', " ").replace("'", " ").split())
    return bool(words) and words in " ".join(
        context.lower().replace('"', " ").replace("'", " ").split()
    )


def step_text(step: BaseModel) -> str:
    """Every free-text field of a step, for ref checks and the evals."""
    return " ".join(str(getattr(step, name)) for name in STEP_TEXT_FIELDS if hasattr(step, name))


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
    # A control step that ended the waiting plan (#74): its ledger id, and
    # whether it was ``cancelled`` or ``amended`` (the amended words run on).
    ends: int | None = None
    ends_as: Literal["cancelled", "amended"] | None = None


@dataclass(frozen=True)
class PausedPlan:
    """A compiled plan from the last turn, waiting on the person."""

    body: dict[str, Any]
    reason: PauseReason
    resume_at: int
    app: str | None = None
    # Its ledger row, for a control step that ends it (#74).
    plan_id: int | None = None


@dataclass(frozen=True)
class Touched:
    """One thing a receipt touched, as the ledger kept it."""

    kind: str
    ref: str
    title: str | None = None
    untrusted: bool = False


@dataclass(frozen=True)
class UndoReceipt:
    """A write receipt an ``undo`` can act on (#74), from the ledger.

    ``outcome`` is ``done`` (the app's) or ``unknown`` (conductor's, when an
    app ran and said nothing it could read). ``handle``/``how`` are the app's
    undo, ``none`` why there is none; ``undone`` means a later step already
    undid it.
    """

    id: int
    plan_id: int
    app: str
    tool: str | None
    outcome: str
    touched: tuple[Touched, ...] = ()
    count: int | None = None
    handle: str | None = None
    how: str | None = None
    none: str | None = None
    undone: bool = False


@dataclass(frozen=True)
class Ledgered:
    """What the ledger says a control step acts on (#74).

    ``last_plan`` is the newest plan that ran steps, leaving out control
    plans and steps that undid something; ``receipts`` are the write
    receipts of the last few such plans, newest plan first, so a target that
    names something older can still find it. ``jobs`` are running jobs as
    ``app:ref``.
    """

    last_plan: int | None = None
    receipts: tuple[UndoReceipt, ...] = ()
    jobs: tuple[str, ...] = ()
    # The same running jobs with their latest report (#76), newest first.
    running: tuple[RunningJob, ...] = ()


@dataclass(frozen=True)
class RunningJob:
    """A job this conversation started that hasn't ended, as last reported."""

    app: str
    ref: str
    status: str
    done: int = 0
    total: int | None = None
    title: str | None = None
    untrusted: bool = False


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
