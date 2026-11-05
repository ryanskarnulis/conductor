"""D1: the plan schema, the planner prompt and the code validator (spike only)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

StepId = Literal["s1", "s2", "s3", "s4", "s5"]


class Ask(BaseModel):
    id: StepId
    do: Literal["ask"]
    app: Literal["tasks", "music"]
    request: str
    destructive: bool


class Look(BaseModel):
    id: StepId
    do: Literal["look"]
    app: Literal["tasks", "music", "chess"]


class Open(BaseModel):
    id: StepId
    do: Literal["open"]
    app: Literal["chess"]
    intent: str


class AskUser(BaseModel):
    id: StepId
    do: Literal["ask_user"]
    question: str
    options: list[str]


class Recall(BaseModel):
    id: StepId
    do: Literal["recall"]
    query: str


class Cancel(BaseModel):
    id: StepId
    do: Literal["cancel"]
    target: str


class Amend(BaseModel):
    id: StepId
    do: Literal["amend"]
    target: str
    change: str


class Confirm(BaseModel):
    id: StepId
    do: Literal["confirm"]
    answer: Literal["yes", "no"]


class Undo(BaseModel):
    id: StepId
    do: Literal["undo"]
    target: str


class Think(BaseModel):
    id: StepId
    do: Literal["think"]
    problem: str
    inputs: list[StepId]


Step = Annotated[
    Union[Ask, Look, Open, AskUser, Recall, Cancel, Amend, Confirm, Undo, Think],
    Field(discriminator="do"),
]


class Plan(BaseModel):
    steps: list[Step] = Field(max_length=5)
    say: Literal["result", "knowledge", "decline", "capabilities", "status", "chat"]


SCHEMA = Plan.model_json_schema()

_REF = re.compile(r"\{(s[1-5])(?:\.[a-z_]+)?\}")


def step_text(step: BaseModel) -> str:
    return " ".join(
        str(getattr(step, field))
        for field in ("request", "intent", "query", "question", "target", "change", "problem")
        if hasattr(step, field)
    )


def validate(plan: Plan, state_refs: set[str]) -> list[str]:
    """The code half of the compiler: what a real interpreter would refuse to run."""
    problems: list[str] = []
    seen: set[str] = set()
    for index, step in enumerate(plan.steps):
        if step.id in seen:
            problems.append(f"duplicate id {step.id}")
        for ref in _REF.findall(step_text(step)):
            if ref not in seen:
                problems.append(f"{step.id} refers to {ref} before it runs")
        if isinstance(step, Think):
            for ref in step.inputs:
                if ref not in seen:
                    problems.append(f"{step.id} thinks over {ref} before it runs")
        if isinstance(step, AskUser) and len(plan.steps) > 1:
            problems.append("ask_user must be the only step")
        if isinstance(step, Open) and index != len(plan.steps) - 1:
            problems.append("open must be the last step")
        if isinstance(step, (Cancel, Amend, Undo)):
            if step.target != "last" and not any(ref in step.target for ref in state_refs):
                problems.append(f"{step.do} target {step.target!r} is not in State")
        if isinstance(step, Confirm) and "pending" not in state_refs:
            problems.append("confirm with nothing pending")
        seen.add(step.id)
    if plan.steps and plan.say != "result":
        problems.append("steps with a non-result say")
    if not plan.steps and plan.say == "result":
        problems.append("no steps but say=result")
    return problems


def _fleet_lines() -> str:
    """App lines from the real manifests, reshaped for the plan vocabulary."""
    sys.path.insert(0, str(Path("/home/rskarnulis/Documents/workspace/conductor/backend")))
    from app.fleet.manifests import discover_fleet  # noqa: PLC0415

    fleet = discover_fleet(Path("/home/rskarnulis/Documents/workspace"))
    seen: set[str] = set()
    lines: list[str] = []
    glance = {
        "tasks": "today's and overdue tasks, and today's Focus",
        "music": "running and recent download jobs with progress, and how many songs wait to be sorted",
        "chess": "the current game (whose move, who's ahead) and the last finished game's result",
    }
    for app in fleet.agent_apps():
        if app.name in seen or app.agent is None:
            continue
        seen.add(app.name)
        examples = "; ".join(f'"{e}"' for e in app.agent.examples)
        lines.append(
            f"- {app.name} (ask, look): {app.agent.description}\n"
            f"  Glance: {glance[app.name]}.\n  Asks that go here: {examples}"
        )
    for app in fleet.open_apps():
        if app.name in seen or app.open is None:
            continue
        seen.add(app.name)
        examples = "; ".join(f'"{e}"' for e in app.open.examples)
        lines.append(
            f"- {app.name} (open, look): {app.open.description}\n"
            f"  Glance: {glance[app.name]}.\n  Asks that go here: {examples}"
        )
    return "\n".join(lines)


PLANNER_PROMPT = f"""\
You are the planner inside the household's conductor, a local assistant. You \
never talk to the person. You turn what they just said into a plan; code checks \
the plan and runs it, and a separate narrator speaks to them afterwards from \
what actually happened. Plan only what the person asked for.

A plan is 0 to 5 steps, ids s1, s2, ... in order. The step kinds:
- ask: hand a request to an app's agent, as one self-contained message in the \
person's words. The app's agent knows its own domain (dates, matching names, \
its own confirmations), so pass the request through; don't do its job. Set \
"destructive" to true when the request deletes, wipes, trashes, resets or \
abandons something.
- look: read an app's glance, a fixed summary listed below, instead of asking \
its agent. Use it only when the glance answers the question as asked.
- open: send the person to an app they use themselves (chess), carrying their \
words as the intent. Anything to do with playing or changing a game is an \
open. An open is always the last step.
- ask_user: ask the person one short question when a wrong guess would act on \
the wrong thing; list the candidates as options. It is always the only step.
- recall: search earlier conversations and results, when the question is about \
what was said or done before.
- cancel: stop a running job or a pending plan listed under State. target: its ref.
- amend: change a running job or pending plan listed under State. target: its \
ref; change: what to change.
- confirm: the person's answer (yes or no) to the question under "Waiting on \
the person".
- undo: reverse something done before. target: "last" for the last plan, or a \
ref from State.
- think: hand a hard multi-part problem (planning a week, weighing many \
options) to a slower, stronger model. inputs: the ids of earlier steps whose \
results it needs. It comes back as a proposal the person confirms.

Passing data between steps: write {{s1}} inside a later step's text to insert \
the main thing step s1 touched once it has run (the album it queued, the task \
it found). Use it whenever a later step needs something only an earlier step \
will find out. If State already names the thing, write its name instead.

"say": with steps, "result". With no steps, how the narrator answers:
- knowledge: general knowledge that doesn't change (cooking times, \
definitions, capitals). Never for weather, news, prices, scores or anything \
current.
- decline: nothing in the fleet can do it (weather, news, lights, playing \
music out loud).
- capabilities: they ask what you can do.
- status: they ask what you are doing or just did; the narrator answers from State.
- chat: thanks, greetings, small talk.

The apps:
{_fleet_lines()}

Reply with the plan as JSON only."""


def user_message(text: str, state: str) -> str:
    return f"State:\n{state}\n\nThe person said: {text}"
