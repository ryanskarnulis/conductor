"""The plan compiler (#70): the 12B writes a typed plan, code checks it.

One planner call under ``json_schema`` decoding (:func:`app.ai.plan.schema`),
then the validator (:func:`app.ai.plan.problems`). A plan with problems gets
one repair: the planner sees its plan and code's objections and writes it
again. A plan that is still invalid, or JSON that isn't a plan, is no plan
(:class:`Compiled` says why); the interpreter gives it one more round
(:func:`repair_turn`, #75), then answers in code's words, never a guess.

The prompt is spike D1's v2 (#56: 88% on the 12B, 99.6% valid first try),
built from the fleet's manifests and byte-identical turn to turn, so the
server's prompt cache keeps it; what changes per turn rides in the user
message (the state block and recall, then the person's words). Plans have no
designed length; the schema's ceiling stops a runaway.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from app.ai import facts, turn_context
from app.ai import plan as plans
from app.ai.provider import ChatResult
from app.fleet.manifests import Fleet

logger = structlog.get_logger(__name__)

# Room for a dozen steps of JSON with long requests.
MAX_TOKENS = 1500

# A model call: messages, a completion cap and a response format in, one turn out.
PlannerChat = Callable[[list[dict[str, Any]], int, dict[str, Any]], ChatResult]


def planner_prompt(fleet: Fleet) -> str:
    """The planner's system prompt, from the fleet's manifests."""
    return _PROMPT.replace("{apps}", _app_lines(fleet))


def _app_lines(fleet: Fleet) -> str:
    lines: list[str] = []
    for app in fleet.apps:
        if app.agent is not None:
            spec_description, examples, verbs = app.agent.description, app.agent.examples, "ask"
        elif app.open is not None:
            spec_description, examples, verbs = app.open.description, app.open.examples, "open"
        elif app.glance is not None:
            # Readable and nothing else: a look is all the planner can do here.
            lines.append(f"- {app.name} (look): {app.title}\n  Glance: {_holds(app.glance.holds)}.")
            continue
        else:
            continue
        quoted = "; ".join(f'"{example}"' for example in examples)
        if app.glance is None:
            # No glance, no `look`: the planner is neither offered nor told of one.
            lines.append(
                f"- {app.name} ({verbs}): {spec_description}\n  Asks that go here: {quoted}"
            )
            continue
        lines.append(
            f"- {app.name} ({verbs}, look): {spec_description}\n"
            f"  Glance: {_holds(app.glance.holds)}.\n"
            f"  Asks that go here: {quoted}"
        )
    # Memory (#81): conductor's own app, after the fleet's.
    lines.append(facts.PROMPT_LINE)
    return "\n".join(lines)


def _holds(holds: str) -> str:
    """The manifest's ``glance.holds`` as a clause the prompt can close with a period."""
    return holds.rstrip(". ")


_PROMPT = """\
You are the planner inside the household's conductor, a local assistant. You \
never talk to the person. You turn what they just said into a plan; code checks \
the plan and runs it, and a separate narrator speaks to them afterwards from \
what actually happened. Plan only what the person asked for.

A plan is steps with ids s1, s2, ... in order, as many as the request needs. \
The step kinds:
- ask: hand a request to an app's agent, as one self-contained message in the \
person's words. The app's agent knows its own domain (dates, matching names, \
its own confirmations), so pass the request through in their words; don't do \
its job. Never turn it into a search ("find the task related to...") or add \
dates: the app matches names and works out dates itself. Several things for \
one app go in one request. Set "destructive" to true when the \
request deletes, wipes or trashes something the app keeps. Only apps marked \
(ask) take an ask.
- look: read an app's glance, a fixed summary listed below, instead of asking \
its agent. Use it only when the glance answers the question as asked.
- open: send the person to an app they use themselves (chess), carrying their \
words as the intent. Anything to do with a game (playing, moving, analyzing, \
resetting or resigning it) is an open; "the game" always means chess. An open \
is always the last step.
- ask_user: ask the person one short question when a wrong guess would act on \
the wrong thing; list the candidates State names as options. When State names \
two or more that fit their words, ask_user first, never an ask that picks one. \
When State doesn't name the candidates, ask the app instead: it knows its own \
names. It comes first; later steps can use the answer as {s1}, and the plan waits for it.
- recall: search earlier conversations and results, when the question is about \
what was said or done before ("when did I last...", "what did we decide..."). \
Never a look for these: a glance only shows now.
- cancel: stop a running job or a pending plan listed under State. target: its ref.
- amend: change a running job (never a new ask for a job already running) or \
a pending plan listed under State. target: its ref; change: what to change.
- confirm: the person's answer (yes or no) to the question under "Waiting on \
the person". It is the first step.
- undo: reverse something done before. target: "last" for the last plan, or a \
ref from State.
- think: hand a hard multi-part problem (planning a week, weighing many \
options) to a slower, stronger model. inputs: the ids of earlier steps whose \
results it needs. It comes back as a proposal the person decides on later, so \
think is always the last step: never add a confirm, an ask_user or an ask to \
act on its result.

Passing data between steps: write {s1} inside a later step's text to insert \
the main thing step s1 touched once it has run (the album it queued, the task \
it found). Use it whenever a later step needs something only an earlier step \
will find out. If State already names the thing, write its name; use recall \
only for things State doesn't name.
Example: "download the new Bonobo album and remind me Sunday to listen to it" \
-> {"steps": [{"id": "s1", "do": "ask", "app": "music", "request": "download \
the new Bonobo album", "destructive": false}, {"id": "s2", "do": "ask", "app": \
"tasks", "request": "add a task for Sunday to listen to {s1}", "destructive": \
false}]}. Never write "the new album" where {s1} belongs: only the earlier step \
will know its real title.

When State shows something "Waiting on the person" and their message says yes \
or no to it, the plan is {"steps": [{"id": "s1", "do": "confirm", "answer": \
"yes" or "no"}]}, never an answer.

When nothing needs to run, reply {"answer": ...} instead of steps:
- knowledge: general knowledge that doesn't change (cooking times, \
definitions, capitals). Never for weather, news, prices, scores or anything \
current.
- decline: nothing in the fleet can do it (weather, news, lights, playing \
music out loud).
- self: they ask what you can do, what you're doing or what you just did; the \
narrator answers from State and the app list.
- chat: thanks, greetings, small talk.

When State says your last reply asked them a question and their message \
answers it, reply {"resume": "<their answer, in their words>"} and the waiting \
plan carries on.

The apps:
{apps}

Reply with the plan as JSON only."""


@dataclass
class Compiled:
    """What the compiler made of one turn."""

    plan: plans.Plan | None
    # Each planner reply, raw: the first, and the repair's if there was one.
    raw: list[str] = field(default_factory=list)
    # The validator's objections to the last plan (or why there is no plan).
    problems: list[str] = field(default_factory=list)
    repaired: bool = False


def user_message(context: str | None, words: str) -> str:
    """The turn's one planner message: State (and recall), then the words.
    An earlier request in the same words is left out of what the planner
    reads (#141, :func:`app.ai.turn_context.for_planner`); the validator's
    ``checks.context`` keeps it."""
    said = f"The person said: {words}"
    shown = turn_context.for_planner(context, words) if context else context
    return f"{shown}\n\n{said}" if shown else said


def _problems(plan: plans.Plan, words: str, checks: plans.Checks) -> list[str]:
    """The validator's objections, and a recalled answer (#127): a plan with
    any of these doesn't run."""
    return plans.problems(plan, checks) + plans.recalled_answer(plan, words, checks.context)


def _nudges(plan: plans.Plan, words: str, checks: plans.Checks) -> list[str]:
    """Repairs that read the person's words against State (#134): a
    remembered topic nothing saved is about, a lookup of a fact State lists,
    a question about you filed as general knowledge. The first plan gets
    them once; a repaired plan that keeps its choice still runs, since each
    of these plans can run and a turn that answers beats code's "couldn't
    put that together"."""
    return (
        plans.remembered_topic(plan, words, checks.context)
        + plans.looked_up_fact(plan, words, checks.context)
        + plans.about_you(plan, words)
    )


def compile_turn(
    chat: PlannerChat,
    *,
    system_prompt: str,
    context: str | None,
    words: str,
    fleet: Fleet,
    checks: plans.Checks,
) -> Compiled:
    """Compile one turn: a plan, at most one repair, or no plan with reasons.

    Provider failures propagate: the caller decides what a dead model means.
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "plan",
            "strict": True,
            "schema": plans.schema(fleet, resumable=checks.resumable),
        },
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message(context, words)},
    ]
    compiled = Compiled(plan=None)
    for attempt in range(2):
        result = chat(messages, MAX_TOKENS, response_format)
        raw = result.content or ""
        compiled.raw.append(raw)
        plan: plans.Plan | None
        try:
            plan = plans.parse(raw)
        except ValueError as exc:
            found = [f"not a plan: {str(exc).splitlines()[0][:200]}"]
            plan = None
        if plan is not None:
            found = _problems(plan, words, checks)
            if attempt == 0:
                found += _nudges(plan, words, checks)
        compiled.problems = found
        if plan is not None and not found:
            compiled.plan = plan
            compiled.repaired = attempt == 1
            return compiled
        if attempt == 0:
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Code can't run that plan: "
                        + "; ".join(found)
                        + ". Reply with the corrected plan as JSON only."
                    ),
                },
            ]
    logger.warning("plan_invalid", problems=compiled.problems)
    return compiled


def repair_turn(
    chat: PlannerChat,
    *,
    system_prompt: str,
    context: str | None,
    words: str,
    fleet: Fleet,
    checks: plans.Checks,
    previous: Compiled,
    found: list[str],
) -> Compiled:
    """One more repair round (#75), after :func:`compile_turn` gave no plan the
    interpreter can run: the last reply and code's objections (the
    validator's, or :func:`app.ai.plan.unrunnable`'s) go back once more. A
    turn with no reply to show (the server's answer wasn't a completion)
    asks again from the top. Returns ``previous`` with this round's reply and
    problems added, and the plan if this one can run.

    Provider failures propagate, as in :func:`compile_turn`.
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "plan",
            "strict": True,
            "schema": plans.schema(fleet, resumable=checks.resumable),
        },
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message(context, words)},
    ]
    if previous.raw:
        messages += [
            {"role": "assistant", "content": previous.raw[-1]},
            {
                "role": "user",
                "content": (
                    "Code still can't run that plan: "
                    + "; ".join(found)
                    + ". Reply with the corrected plan as JSON only."
                ),
            },
        ]
    result = chat(messages, MAX_TOKENS, response_format)
    raw = result.content or ""
    previous.raw.append(raw)
    previous.repaired = True
    previous.plan = None
    plan: plans.Plan | None
    try:
        plan = plans.parse(raw)
    except ValueError as exc:
        previous.problems = [f"not a plan: {str(exc).splitlines()[0][:200]}"]
        return previous
    problems = _problems(plan, words, checks) or plans.unrunnable(plan)
    previous.problems = problems
    if not problems:
        previous.plan = plan
    else:
        logger.warning("plan_unrepaired", problems=problems)
    return previous
