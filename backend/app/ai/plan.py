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
runaway.

Code owns what happens next. :func:`problems` is the validator: refs, order,
targets, policy and budgets; a plan with problems gets a repair.
:func:`unrunnable` names what the interpreter can't run in a valid plan: a
``{sN}`` on a step that gives no name. Every step kind runs (#75): ``ask``/
``open``/``ask_user``/``confirm`` across apps, with ``{sN}`` data flow
between them (#72), ``look`` (#73: a plain read of an app's glance, offered
only for apps whose manifest declares one), the control steps ``cancel``/
``amend``/``undo`` (#74; on a running job through the jobs runtime, #76),
``recall`` over the ledger (#80, :mod:`app.services.memory`), ``think`` (#78:
a background job whose proposal waits for a yes), and the direct answers.
:func:`reads_only` says which asks may run in parallel.
Pure: no I/O.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from app.ai.facts import APP as MEMORY_APP
from app.ai.facts import parse as parse_memory
from app.ai.facts import related, words_of
from app.ai.memory import terms
from app.fleet.manifests import Fleet

# A ceiling against a runaway, not a design limit (decided 2026-11-06).
MAX_STEPS = 12
STEP_IDS = tuple(f"s{n}" for n in range(1, MAX_STEPS + 1))

AnswerKind = Literal["knowledge", "decline", "self", "chat"]
StepKind = Literal[
    "ask", "look", "open", "ask_user", "recall", "cancel", "amend", "confirm", "undo", "think"
]

# Control steps (#74): they act on what's waiting or on what just ran, so each
# runs as a plan of its own (:mod:`app.ai.control`); one inside a longer plan
# is split off and runs first (#75).
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

# What the person's message is about (#151): the planner's first key, written
# before the plan's shape, so the hardest decision isn't made on the plan's
# first token. Code checks it agrees with the plan (:func:`disagrees`); it is
# not part of the plan and isn't stored with it.
ABOUT: tuple[str, ...] = (
    "an app",
    "conductor itself",
    "general knowledge",
    "the outside world",
    "the waiting question",
    "a running job",
    "small talk",
)


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return text


def parse(text: str) -> Plan:
    """The planner's JSON as a plan; raises ``ValueError`` when it isn't one."""
    return _PLAN.validate_json(_strip_fence(text))


def about_of(text: str) -> str | None:
    """The reply's ``about`` (#151), or ``None`` when it has none we know."""
    try:
        body = json.loads(_strip_fence(text))
    except ValueError:
        return None
    about = body.get("about") if isinstance(body, dict) else None
    return about if about in ABOUT else None


_RUNNING_HEADER = "Running jobs"
_WAITING_HEADER = "Waiting on the person"


def offered_about(checks: Checks) -> tuple[str, ...]:
    """The ``about`` values this turn's State supports (#151): "a running job"
    only when State lists one, "the waiting question" only when something
    waits on the person. Offered always, both drew plans for things that
    weren't there ("undo that" as a running job, "yes" with nothing waiting)."""
    jobs = any(line.startswith(_RUNNING_HEADER) for line in checks.context.splitlines())
    waiting = _waits(checks)
    return tuple(
        value
        for value in ABOUT
        if (value != "a running job" or jobs) and (value != "the waiting question" or waiting)
    )


def _waits(checks: Checks) -> bool:
    """Whether something waits on the person: a question a confirm or a
    resume answers, or the state block's waiting line."""
    return (
        checks.confirmable
        or checks.resumable
        or any(line.startswith(_WAITING_HEADER) for line in checks.context.splitlines())
    )


def dump(plan: Plan) -> dict[str, Any]:
    return plan.model_dump()


def load(body: dict[str, Any]) -> Plan:
    return _PLAN.validate_python(body)


# --- the schema ----------------------------------------------------------------------------


def schema(
    fleet: Fleet, *, resumable: bool = False, about: Collection[str] = ABOUT
) -> dict[str, Any]:
    """The ``json_schema`` the planner decodes under, from the fleet.

    Each step kind is an object with every field required and nothing else,
    which is what llama-server's strict grammar needs. ``resumable`` offers the
    ``resume`` shape, only while a plan is paused on ``ask_user``; ``about``
    the values State supports (:func:`offered_about`, #151).
    """
    # ``memory`` (#81) is conductor's own app: always there, run by code.
    ask_apps = sorted({app.name for app in fleet.agent_apps()} | {MEMORY_APP})
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
    # Every shape opens with ``about`` (#151), so it is decided first.
    about_values = [value for value in ABOUT if value in about]
    about_field = {"about": {"type": "string", "enum": about_values}}
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
                **about_field,
                "steps": {
                    "type": "array",
                    "items": {"anyOf": kinds},
                    "minItems": 1,
                    "maxItems": MAX_STEPS,
                },
            },
            "required": ["about", "steps"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                **about_field,
                "answer": {"type": "string", "enum": ["knowledge", "decline", "self", "chat"]},
            },
            "required": ["about", "answer"],
            "additionalProperties": False,
        },
    ]
    if resumable:
        shapes.append(
            {
                "type": "object",
                "properties": {**about_field, "resume": text},
                "required": ["about", "resume"],
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


# A confirm's repair when nothing waits at all (#176): since #151's `about` the
# 12B repaired "yes" into a decline; before it, into chat.
_NO_QUESTION = (
    ": State shows no question, so a yes or no answers nothing; when that is all they said, "
    'it is small talk: {"about": "small talk", "answer": "chat"}'
)


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
    found += _mixed_asks(steps, checks.fleet)
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
            if step.app != MEMORY_APP and (app is None or not app.has_agent):
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
                found.append(
                    "confirm with nothing waiting on the person"
                    + ("" if _waits(checks) else _NO_QUESTION)
                )
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


def _work_for(noun: str) -> re.Pattern[str]:
    """Words that ask for a new one of another app's things: "add a task",
    "make another task", and for tasks, "remind me"."""
    made = (
        rf"\b(?:add|make|create|put|set\s+up|set|schedule|write)\b(?:\s+[\w'’]+){{0,2}}?"
        rf"\s+(?:a|an|another|one\s+more)\s+(?:new\s+|quick\s+)?{noun}s?\b"
    )
    if noun == "task":
        made += r"|\bremind\s+(?:me|us)\b"
    return re.compile(made, re.IGNORECASE)


def _mixed_asks(steps: list[Step], fleet: Fleet) -> list[str]:
    """An ask that carries a second app's work (#127, S15): "download Caribou's
    other album and add a task to sort it", all sent to music. Music can't
    make the task, and the plan would claim it was asked. The repair splits
    it: each app gets its own step, the second writing ``{sN}`` for what the
    first finds. Another app's work is a new one of its things ("add a task",
    "remind me" for tasks); naming its things in passing ("add a task to sort
    my music") is not."""
    others = {app.name: _work_for(app.name.lower().removesuffix("s")) for app in fleet.agent_apps()}
    found: list[str] = []
    for step in steps:
        if not isinstance(step, Ask):
            continue
        for name, work in others.items():
            if name == step.app:
                continue
            carried = work.search(step.request)
            if carried is None:
                continue
            found.append(
                f'{step.id} asks {step.app} to "{carried.group(0)}", which is work for {name}: '
                f"{step.app} can't do it. Split it: ask {step.app} only its own part, then ask "
                f"{name} the rest as its own step, writing {{{step.id}}} for what {step.app} finds"
            )
            break
    return found


# A call line in State or recall and the person's words on it: this
# conversation's record (`… (asked: "…")`) and other conversations' hits
# (`…, asked "…": ask_tasks …`).
_ASKED = re.compile(r'asked:? "(?P<asked>[^"]*)"')
_APP_CALL = re.compile(r"\b(?:ask|look)_[a-z0-9_]+\b")


def recalled_answer(plan: Plan, words: str, context: str) -> list[str]:
    """A direct answer to a question an app answered before (#127, #122's
    leak): "what's due today?" again, with the earlier ``ask_tasks`` and its
    "Nothing's due" in State. That answer is as of then, so the plan asks or
    looks again. Matched on the words memory matches on (the same question
    asked again), only for a call that reached an app."""
    if not isinstance(plan, Direct) or plan.answer not in ("knowledge", "chat"):
        return []
    wanted = terms(words)
    if not wanted:
        return []
    for line in context.splitlines():
        if not _APP_CALL.search(line):
            continue
        for found in _ASKED.finditer(line):
            asked = found.group("asked").rstrip("…").strip()
            if asked and terms(asked) == wanted:
                return [
                    f'they asked "{asked}" before and an app answered it then; that answer is '
                    "as of then, not now, so never answer from it: ask or look at that app again"
                ]
    return []


# A question about conductor itself or what it keeps about the person (#134),
# which is never general knowledge: "what can you do?", "what did you just do?",
# and since #158 "what did you just say?" and "which apps are you connected to?".
_ABOUT_YOU = re.compile(
    r"^\W*(?:(?:so|and|hey|ok|okay)[\s,]+)?(?:what|which)\s+(?:else\s+)?(?:"
    r"can\s+you\s+(?:do|help)|are\s+you\s+(?:doing|working|up\s+to)|"
    r"did\s+you\s+(?:just\s+)?(?:do|say|said)|have\s+you\s+(?:just\s+)?(?:done|said)|"
    r"apps?\s+(?:are\s+you\s+connected\s+to|can\s+you\s+reach|"
    r"do\s+you\s+have(?:\s+access\s+to)?))\b",
    re.IGNORECASE,
)
_ABOUT_ME = re.compile(
    r"^\W*(?:what|how\s+much)\s+(?:else\s+)?do\s+you\s+(?:know|remember)\s+about\s+(?:me|myself)\b",
    re.IGNORECASE,
)


# "… just now", "… a moment ago": the last exchange, which State shows (#176).
_JUST = re.compile(r"\b(?:just|now|a\s+(?:moment|minute|second)\s+ago)\b", re.IGNORECASE)


def about_you(plan: Plan, words: str) -> list[str]:
    """A ``knowledge`` answer to a question about conductor itself ("what can
    you do?") or what it keeps about the person ("what do you know about
    me?"): since #81's memory line the 12B files these as general knowledge
    about half the time. Knowledge doesn't know conductor; the repair points
    at the answer that does. Only these few phrasings; ``chat`` stays.

    Since #151 "what did you say just now?" is filed as a recall instead
    (#176), which searches earlier conversations, not the last reply: a plan
    of only searches or thinks for what it just did or said goes back too."""
    if (
        isinstance(plan, StepPlan)
        and all(step.do in ("recall", "think") for step in plan.steps)
        and _ABOUT_YOU.search(words)
        and _JUST.search(words)
    ):
        return [
            "they asked what you just did or said, which State and the last exchange show; a "
            'recall searches earlier conversations, not your last reply: answer {"answer": '
            '"self"}'
        ]
    if not isinstance(plan, Direct) or plan.answer != "knowledge":
        return []
    if _ABOUT_ME.search(words):
        return [
            "they asked what you know about them, which isn't general knowledge: ask memory, "
            'or answer {"answer": "self"}'
        ]
    if _ABOUT_YOU.search(words):
        return [
            "they asked about you, which isn't general knowledge: answer "
            '{"answer": "self"} (the narrator answers from State and the app list)'
        ]
    return []


# What each ``about`` (#151) rules out, and what the repair says instead. Loose
# on purpose: only a plan the field contradicts goes back.
_ABOUT_SAYS: dict[str, str] = {
    "an app": "something an app keeps or does is planned as its steps, never a direct answer",
    "conductor itself": (
        "a question about you is never general knowledge, a decline or a think: "
        'answer {"answer": "self"} from State, or ask memory what you know about them'
    ),
    "general knowledge": 'general knowledge is {"answer": "knowledge"}',
    "the outside world": (
        "weather, news, scores, prices and devices aren't general knowledge or about you: "
        '{"answer": "decline"}, unless an app below does it'
    ),
    "the waiting question": (
        "their answer to what is waiting is a confirm, a resume, a cancel or an amend, never "
        "general knowledge or a decline"
    ),
    "a running job": (
        "a running job is stopped with cancel or changed with amend (target: its ref), never "
        "a new ask"
    ),
    "small talk": 'small talk is {"answer": "chat"}',
}


def _agrees(about: str, plan: Plan) -> bool:
    if isinstance(plan, Resume):
        return about == "the waiting question"
    if isinstance(plan, Direct):
        allowed = {
            "conductor itself": ("self", "chat"),
            "general knowledge": ("knowledge",),
            "the outside world": ("decline",),
            "the waiting question": ("self", "chat"),
            "a running job": ("self", "chat"),
            "small talk": ("chat", "self"),
        }.get(about, ())
        return plan.answer in allowed
    if about in ("general knowledge", "small talk"):
        return False
    if about == "conductor itself":
        # Never only a think. A recall passes (#180): "what did we decide
        # about X?" is labelled so and is right; a recall for what was just
        # said is :func:`about_you`'s, which reads the words.
        return any(step.do != "think" for step in plan.steps)
    if about == "a running job":
        kinds = {step.do for step in plan.steps}
        return "ask" not in kinds or bool(kinds & {"cancel", "amend"})
    return True


def disagrees(about: str | None, plan: Plan) -> list[str]:
    """The reply's ``about`` (#151) against its plan: the planner said what the
    message is about, then planned something that can't be about that ("general
    knowledge" with an ask, "an app" with a direct answer, "a running job" with
    a new ask). One repair, like #134's nudges; a reply with no ``about`` passes."""
    if about is None or about not in _ABOUT_SAYS or _agrees(about, plan):
        return []
    return [
        f'you wrote "about": "{about}", but the plan is {describe(plan)}: '
        f'{_ABOUT_SAYS[about]}. Make them agree: fix the plan, or "about" if that was wrong'
    ]


# "what do you remember about the fence project?" (#134): what was said or
# done before, unless it is about the person or something they saved.
_REMEMBER_ABOUT = re.compile(
    r"\b(?:what|do|did)\s+(?:do\s+|did\s+)?you\s+(?:still\s+)?(?:remember|recall)\s+"
    r"(?:about\s+|of\s+)?(?P<topic>[^?.!]+)",
    re.IGNORECASE,
)
_ABOUT_THEM = re.compile(r"^(?:me|myself|us|ourselves)\b", re.IGNORECASE)
_FACTS_HEADER = "Facts the person asked you to remember"


def remembered_topic(plan: Plan, words: str, context: str) -> list[str]:
    """ "What do you remember about the fence project?" planned as anything but
    a recall (#134: since #81's memory line the 12B asked tasks): unless a fact
    the person saved is about it (then memory is right), it asks what was said
    or done before, so the repair asks for a recall. Questions about the
    person themselves ("… about me?") are memory's and pass."""
    found = _REMEMBER_ABOUT.search(words)
    if found is None:
        return []
    topic = " ".join(found.group("topic").split()).strip(" ,;:")
    wanted = words_of(topic)
    if not wanted or _ABOUT_THEM.match(topic):
        return []
    if isinstance(plan, StepPlan) and any(step.do == "recall" for step in plan.steps):
        return []
    if any(wanted & words_of(fact) for fact in saved_in(context)):
        return []
    return [
        f'they asked what you remember about "{topic}", and none of the facts they saved is '
        "about it: that is a question about what was said or done before, so plan a recall "
        f'step for "{topic}" (other steps may follow it)'
    ]


# Words that narrow what is already under way (#160): "just the first ten",
# "make it the first twenty instead", "only five", "stop after ten".
_NUMBER = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|"
    r"twenty|thirty|forty|fifty|hundred|few|half)"
)
_NARROWS = re.compile(
    r"\b(?:just|only|instead|no\s+more\s+than|at\s+most|stop\s+(?:at|after))\b|"
    rf"\b(?:first|last)\s+(?:\w+\s+)?{_NUMBER}\b|\b{_NUMBER}\s+of\s+them\b",
    re.IGNORECASE,
)
# A running job as the state block lists it: "music:job:17 …" (the eval
# cases), "music job 31, …" (the ledger) or "music ask_music (job 9): …".
_JOB_LINE = re.compile(
    r"^- (?P<app>[a-z0-9_-]+)(?::job:|\s+job\s+|\s+\S+\s+\(job\s+)(?P<ref>[\w-]+)"
)


def running_in(context: str) -> list[tuple[str, str]]:
    """The running jobs State listed (#76), as ``(app, ref)``, in its order."""
    found: list[tuple[str, str]] = []
    inside = False
    for line in context.splitlines():
        if line.startswith(_RUNNING_HEADER):
            inside = True
            continue
        if not inside:
            continue
        match = _JOB_LINE.match(line)
        if match is None:
            break
        found.append((match.group("app"), match.group("ref")))
    return found


def narrows_job(plan: Plan, words: str, context: str) -> list[str]:
    """A new ask to an app whose job State lists as running, when the
    person's words narrow what is under way ("just grab the first ten songs",
    #160: an ask to music every time on main). That changes the job, so the
    repair asks for an amend of it. The jobs contract has no amend for a
    running job, so the amend then says it can't change it; a new ask would
    start a second download. A plan that already amends passes."""
    if not isinstance(plan, StepPlan) or not _NARROWS.search(words):
        return []
    if any(step.do == "amend" for step in plan.steps):
        return []
    jobs = running_in(context)
    for step in plan.steps:
        if not isinstance(step, Ask):
            continue
        job = next(((app, ref) for app, ref in jobs if app == step.app), None)
        if job is None:
            continue
        ref = f"{job[0]}:job:{job[1]}"
        return [
            f"{ref} is still running, and their words change it: amend it (target: "
            f'"{ref}", change: their words), never a new ask to {step.app}'
        ]
    return []


# Words that change something already said or under way (#176): "make it the
# first ten instead", "change that to Friday". "Instead of …" names its own
# thing ("add milk instead of eggs"), so it isn't one.
_CHANGES = re.compile(
    r"\binstead\b(?!\s+of\b)|"
    r"\b(?:make|change|switch|turn|move|push|bump)\s+(?:it|that|this|them|those)\b",
    re.IGNORECASE,
)
# The state block's lines that give such words something to point at: a job,
# a question, what this conversation touched or said, an earlier lead.
_SOMETHING = (
    _RUNNING_HEADER,
    _WAITING_HEADER,
    "Recent entities",
    "Last plan",
    "Matches for words",
    "Before this turn",
)
# What acts on, reads or searches for a thing: everything but a control step,
# chat or self.
_INVENTS = frozenset({"ask", "look", "open", "ask_user", "recall", "think"})


def points_at_nothing(plan: Plan, words: str, checks: Checks) -> list[str]:
    """Words that change something under way ("actually make it the first ten
    instead") when State shows nothing they could change: no running job,
    nothing waiting, nothing this conversation did or said (#176; #140's
    guard). Since #151 the 12B filed these as a new ask to tasks, inventing
    the work; an amend has code say there is nothing to change. A control
    step, chat or self passes."""
    if not _CHANGES.search(words) or _waits(checks):
        return []
    if any(line.startswith(_SOMETHING) for line in checks.context.splitlines()):
        return []
    if isinstance(plan, Direct):
        if plan.answer in ("chat", "self"):
            return []
    elif isinstance(plan, Resume) or not any(step.do in _INVENTS for step in plan.steps):
        return []
    return [
        "their words change something already under way, but State shows nothing running, "
        "waiting or done in this conversation for them to change, so never invent it with a "
        'new ask, a search or a think: plan one amend step (target: "last", change: their '
        "words), and code tells them there is nothing to change"
    ]


def looked_up_fact(plan: Plan, words: str, context: str) -> list[str]:
    """A plan that first looks up a saved fact State already lists, to feed
    another app's ask ("add a task for my music sorting day": recall or ask
    memory "when do I sort music?", then tasks) (#134). Code sends that fact
    with the ask itself (:func:`app.ai.facts.related`), and a recall names no
    fact to pass on, so the repair drops the lookup and keeps their words."""
    if not isinstance(plan, StepPlan) or len(plan.steps) < 2:
        return []
    first = plan.steps[0]
    lookup = isinstance(first, Recall) or (
        isinstance(first, Ask)
        and first.app == MEMORY_APP
        and parse_memory(first.request).kind == "list"
    )
    then = next((s for s in plan.steps[1:] if isinstance(s, Ask) and s.app != MEMORY_APP), None)
    if not lookup or then is None:
        return []
    saved = related(words, saved_in(context))
    if not saved:
        return []
    return [
        f'{first.id} looks up "{saved[0]}", which State already lists; code sends that fact '
        f"with the ask to {then.app} itself, so leave {first.id} out and ask {then.app} in the "
        "person's own words"
    ]


def saved_in(context: str) -> list[str]:
    """The saved facts the state block listed (#81), in its order."""
    found: list[str] = []
    inside = False
    for line in context.splitlines():
        if line.startswith(_FACTS_HEADER):
            inside = True
            continue
        if not inside:
            continue
        if not line.startswith('- "'):
            break
        text, sep, _ = line[3:].rpartition('" (saved')
        if sep:
            found.append(text)
    return found


# The step kinds whose receipt can give a later step a name for ``{sN}``: an
# ``ask`` (its receipt's title), the ``ask_user`` (the person's answer) and a
# ``recall`` (the one entity it resolved, #80). A ``look`` is several things,
# not a name (#73).
NAMERS: frozenset[str] = frozenset({"ask", "ask_user", "recall"})


def unrunnable(plan: Plan) -> list[str]:
    """What the interpreter can't run in this valid plan, in words the
    planner can repair from (#75): a ``{sN}`` on a step that gives no name.
    Empty when every step can run."""
    if not isinstance(plan, StepPlan):
        return []
    kinds = {step.id: step.do for step in plan.steps}
    found: list[str] = []
    for step in plan.steps:
        # A think reads its inputs whole (#180): its problem's ``{s2}`` for a
        # look it was given is a pointer, not a name to fill.
        given = set(step.inputs) if isinstance(step, Think) else set()
        for ref in dict.fromkeys(REF.findall(step_text(step))):
            if ref in given:
                continue
            kind = kinds.get(ref)
            if kind is not None and kind not in NAMERS:
                a = "an" if kind[0] in "aeiou" else "a"
                found.append(
                    f"{step.id} uses {{{ref}}}, but {ref} is {a} {kind} and gives no name to "
                    "pass on; write the name State gives, or ask the app that knows it"
                )
    return found


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


# ``proposal``: a think's proposal (#78), posted as its job's note; a yes runs it.
PauseReason = Literal["ask_user", "gate", "app_confirm", "proposal"]
# ``invalid``: no plan code could run, even after the repairs; code answered
# (#75). Old rows may also say ``fell_back`` (the loop took the turn).
PlanStatus = Literal["ran", "answered", "paused", "invalid"]


class TurnPlan(BaseModel):
    """What a turn did with its compiled plan, for the ledger and the trace."""

    body: dict[str, Any] | None = None
    status: PlanStatus
    # Why it was invalid, or where it stopped.
    note: str | None = None
    # Whether this plan is what acted. Always since #75; ``False`` only on old
    # rows, where the loop took the turn.
    acted: bool = True
    # A paused plan: why, the 0-based index of the step it resumes at, and
    # the app a gate or confirm is about.
    pause: PauseReason | None = None
    resume_at: int | None = None
    pause_app: str | None = None
    ack: str | None = None
    repaired: bool = False
    # A control step that ended the waiting plan (#74): its ledger id, and
    # whether it was ``cancelled`` or ``amended`` (the amended words run on);
    # a think's proposal (#78) is ``accepted`` by a yes or ``set_aside``.
    ends: int | None = None
    ends_as: Literal["cancelled", "amended", "accepted", "set_aside"] | None = None
    # A control step split off this plan (#75): it ran first, as a plan of its
    # own, and the turn's first ``first_steps`` records are its steps.
    first: TurnPlan | None = None
    first_steps: int = 0


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
        # Memory (#81) answers at once, from code: nothing to wait for.
        if isinstance(step, Ask) and step.app not in apps and step.app != MEMORY_APP:
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
