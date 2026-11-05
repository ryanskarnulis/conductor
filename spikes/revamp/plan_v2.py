"""D1 v2: one schema change and four prompt lines, from v1's misses.

- The top level is either steps (1-5) or a direct answer; v1's redundant
  "say" beside steps was the largest single source of invalid plans.
- ask_user may be followed by steps that use its answer ({s1}); the plan pauses
  there and resumes, like a confirmation. v1's model kept writing this shape.
- "capabilities" and "status" merge into "self".
- Prompt: a worked data-flow example, amend vs a new ask, State before recall.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field, RootModel

from plan_schema import (
    _REF,
    PLANNER_PROMPT,
    Amend,
    AskUser,
    Cancel,
    Confirm,
    Open,
    Step,
    Think,
    Undo,
    step_text,
)


class StepPlan(BaseModel):
    steps: list[Step] = Field(min_length=1, max_length=5)


class Direct(BaseModel):
    answer: Literal["knowledge", "decline", "self", "chat"]


class Plan2(RootModel[Union[StepPlan, Direct]]):
    pass


SCHEMA2 = Plan2.model_json_schema()


def validate2(plan: Plan2, state_refs: set[str]) -> list[str]:
    root = plan.root
    if isinstance(root, Direct):
        return []
    problems: list[str] = []
    seen: set[str] = set()
    for index, step in enumerate(root.steps):
        if step.id in seen:
            problems.append(f"duplicate id {step.id}")
        for ref in _REF.findall(step_text(step)):
            if ref not in seen:
                problems.append(f"{step.id} refers to {ref} before it runs")
        if isinstance(step, Think):
            for ref in step.inputs:
                if ref not in seen:
                    problems.append(f"{step.id} thinks over {ref} before it runs")
        if isinstance(step, AskUser) and index != 0:
            problems.append("ask_user must come first")
        if isinstance(step, Open) and index != len(root.steps) - 1:
            problems.append("open must be the last step")
        if isinstance(step, (Cancel, Amend, Undo)):
            if step.target != "last" and not any(ref in step.target for ref in state_refs):
                problems.append(f"{step.do} target {step.target!r} is not in State")
        if isinstance(step, Confirm) and "pending" not in state_refs:
            problems.append("confirm with nothing pending")
        seen.add(step.id)
    return problems


_HEAD, _, _TAIL = PLANNER_PROMPT.partition('"say": with steps')
_APPS = _TAIL[_TAIL.index("The apps:"):]

PROMPT2 = (
    _HEAD.replace(
        "It is always the only step.",
        "It comes first; later steps can use the answer as {s1}, and the plan "
        "waits for it.",
    ).replace(
        "- amend: change a running job",
        "- amend: change a running job (never a new ask for a job already running)",
    ).replace(
        "If State already names the thing, write its name instead.",
        "If State already names the thing, write its name; use recall only for "
        "things State doesn't name.\nExample: \"download the new Bonobo album and "
        "remind me Sunday to listen to it\" -> "
        '{"steps": [{"id": "s1", "do": "ask", "app": "music", "request": '
        '"download the new Bonobo album", "destructive": false}, {"id": "s2", '
        '"do": "ask", "app": "tasks", "request": "add a task for Sunday to listen '
        'to {s1}", "destructive": false}]}. Never write "the new album" where {s1} '
        "belongs: only the earlier step will know its real title.",
    )
    + """When nothing needs to run, reply {"answer": ...} instead of steps:
- knowledge: general knowledge that doesn't change (cooking times, \
definitions, capitals). Never for weather, news, prices, scores or anything \
current.
- decline: nothing in the fleet can do it (weather, news, lights, playing \
music out loud).
- self: they ask what you can do, what you're doing or what you just did; the \
narrator answers from State and the app list.
- chat: thanks, greetings, small talk.

"""
    + _APPS
)
