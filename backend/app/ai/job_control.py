"""``cancel`` and ``amend`` on a running job (#76), through the jobs contract.

A control step (#74, :mod:`app.ai.control`) aims at a running job when this
conversation has one and nothing is waiting on the person, or when its
target names the job (its ref, its title, or the word "job"). The jobs it can
mean are the ledger's (:attr:`app.ai.plan.Ledgered.running`); with several
running and none named, code asks which, and nothing is stopped.

- **cancel** calls the owning app's ``POST …/jobs/{ref}/cancel`` (idempotent,
  never blocks) and replies in code's words with the job's ``done`` count at
  that moment, from the cancel's answer: "Asked music to stop “Road trip”:
  164 of 1,480 done when I asked." The call is on the turn's record as
  ``cancel_job_<app>`` with a receipt built from the answer, so the ledger's
  job row takes the new report and the write has its receipt. The job then
  ends at the app's next safe boundary, and the watcher posts its one note.
- **amend** can't be done: the contract has no way to change a running job.
  Cancelling and asking again would redo work the job already did, and doing
  that unasked is not ours to choose, so the reply says so plainly, with how
  far the job has got, and how to stop it. Nothing is sent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import structlog

from app.ai import narrator, run_record, think, turn_events
from app.ai import plan as plans
from app.ai.plan import RunningJob, TurnPlan
from app.fleet.jobs import FINISHED, JobError, JobGone, JobRead

if TYPE_CHECKING:
    from app.ai.interpreter import Interpreter, _Run
    from app.ai.loop import AgentRunResult

logger = structlog.get_logger(__name__)

NO_UNDO = "a stopped job can't be resumed; ask for it again to restart it"
# How many jobs a "which one?" lists before counting the rest.
MAX_LISTED = 3


def aims_at_job(
    step: plans.Cancel | plans.Amend,
    running: Sequence[RunningJob],
    *,
    waiting: bool,
    words: str = "",
) -> bool:
    """Whether a cancel or amend is about a running job rather than what's
    waiting: a job runs and nothing waits, or the step names a job."""
    if not running:
        return False
    if not waiting:
        return True
    refs = [f"{job.app}:{job.ref}" for job in running]
    return plans.names_job(step.target, refs) or bool(_named(running, step.target, words))


def cancel(
    interp: Interpreter,
    state: _Run,
    turn: TurnPlan,
    step: plans.Cancel,
    running: Sequence[RunningJob],
) -> AgentRunResult:
    jobs = _chosen(running, step.target, state.words)
    if not jobs:
        turn.note = "cancel: which job?"
        return _say(interp, state, f"{_listing(running)} Which one should I stop?")
    turn_events.plan({"steps": [step.model_dump()]}, 0)
    lines: list[str] = []
    for job in jobs:
        lines.append(_cancel_one(interp, state, job))
    turn.note = "cancel: " + ", ".join(f"{job.app}:{job.ref}" for job in jobs)
    return _say(interp, state, " ".join(lines))


def amend(
    interp: Interpreter,
    state: _Run,
    turn: TurnPlan,
    step: plans.Amend,
    running: Sequence[RunningJob],
) -> AgentRunResult:
    jobs = _chosen(running, step.target, state.words)
    if not jobs:
        turn.note = "amend: which job?"
        return _say(interp, state, f"{_listing(running)} Which one did you mean?")
    job = jobs[0]
    subject = narrator.job_subject(job.app, job.title, job.untrusted)
    turn.note = f"amend: {job.app}:{job.ref} can't be changed while it runs"
    logger.info("job_amend_refused", app=job.app, ref=job.ref)
    return _say(
        interp,
        state,
        f"I can't change {subject} while it runs: {job.app} has no way to edit a job "
        f"in progress. It's at {narrator.job_progress(job.done, job.total)}. Say stop to "
        "cancel it, then ask again with the change.",
    )


# --- one cancel ------------------------------------------------------------------------------


def _cancel_one(interp: Interpreter, state: _Run, job: RunningJob) -> str:
    """Ask the app to stop one job; code's line for what came of it."""
    from app.ai.loop import ToolCallRecord  # noqa: PLC0415

    subject = narrator.job_subject(job.app, job.title, job.untrusted)
    tool = run_record.JOB_CANCEL_PREFIX + job.app
    arguments: dict[str, Any] = {"ref": job.ref}
    seq = len(state.records) + 1
    turn_events.step_started(seq, tool, arguments)
    app = interp._fleet.get(job.app)
    record = ToolCallRecord(tool=tool, arguments=arguments)
    line: str
    if job.app == think.APP:
        # Conductor's own think (#78): no app to ask; the thinker stops it.
        think.request_cancel(job.ref)
        answer = JobRead(
            ref=job.ref,
            status="canceled",
            title=job.title,
            untrusted=job.untrusted,
            cancel_requested=True,
        )
        line = f"Stopped thinking over {think.subject(job.title)}. Nothing was changed."
        record.result = line
        record.app_record = [_receipt(job, answer)]
    elif app is None or app.agent is None:
        record.error = f"{run_record.NOT_RUN_PREFIX}conductor can't reach {job.app}"
        line = f"I can't reach {job.app} to stop {subject}."
    else:
        try:
            answer = interp._cancel_job(app, job.ref)
        except JobGone:
            record.error = f"{job.app} doesn't know that job"
            line = f"{job.app} doesn't know {subject} any more, so there was nothing to stop."
        except JobError as exc:
            record.error = f"couldn't stop it: {exc}"
            line = (
                f"I couldn't stop {subject}: {exc}. As far as I know it's still running "
                f"({narrator.job_progress(job.done, job.total)} at the last report)."
            )
        else:
            line = _cancelled_line(job.app, subject, answer)
            record.result = line
            record.app_record = [_receipt(job, answer)]
    logger.info("job_cancel_step", app=job.app, ref=job.ref, ok=record.error is None)
    state.records.append(record)
    turn_events.receipt(seq, record)
    return line


def _cancelled_line(app: str, subject: str, answer: JobRead) -> str:
    progress = narrator.job_progress(answer.done, answer.total)
    if answer.cancel_requested is not False and answer.status == "canceled":
        return f"Cancelled {subject}: {progress}."
    if answer.cancel_requested is False or answer.status in FINISHED:
        ended = {"completed": "finished", "failed": "failed", "canceled": "been cancelled"}
        lead = subject[:1].upper() + subject[1:]
        return f"{lead} had already {ended.get(answer.status, 'ended')}: {progress}."
    return f"Asked {app} to stop {subject}: {progress} when I asked. I'll post a note when it ends."


def _receipt(job: RunningJob, answer: JobRead) -> dict[str, Any]:
    """The cancel as a v1.1 record entry, built by code from the app's answer:
    a write when the job was still running, else only a read of how it ended."""
    title = answer.title or job.title
    untrusted = answer.untrusted if answer.title else job.untrusted
    live = answer.cancel_requested is not False and not (
        answer.status in FINISHED and answer.status != "canceled"
    )
    entry: dict[str, Any] = {
        "tool": "cancel_job",
        "outcome": "done" if live else "looked_up",
        "touched": [
            {"app": job.app, "kind": "job", "id": job.ref, "title": title, "untrusted": untrusted}
        ],
        "job": {
            "ref": job.ref,
            "status": answer.status,
            "done": answer.done,
            "total": answer.total,
        },
    }
    if live:
        entry["undo"] = {"none": NO_UNDO}
    return entry


# --- which job -------------------------------------------------------------------------------


def _chosen(running: Sequence[RunningJob], target: str, words: str) -> list[RunningJob]:
    """The jobs a target means: those it (or the person's words) names, else
    the only one running. Several running and none named: none."""
    named = _named(running, target, words)
    if named:
        return named
    return list(running[:1]) if len(running) == 1 else []


def _named(running: Sequence[RunningJob], *texts: str) -> list[RunningJob]:
    found: list[RunningJob] = []
    for job in running:
        if any(_names(text, job) for text in texts if text):
            found.append(job)
    if found:
        return found
    # "stop the music download" with one music job among others.
    for text in texts:
        words = set(_words(text))
        by_app = [job for job in running if job.app.lower() in words]
        if len(by_app) == 1:
            return by_app
    return []


def _names(text: str, job: RunningJob) -> bool:
    words = f" {' '.join(_words(text))} "
    ref = job.ref.lower()
    if re.search(rf"(?<![\w]){re.escape(ref)}(?![\w])", text.lower()):
        return True
    title = " ".join(_words(job.title or ""))
    return len(title) >= 4 and f" {title} " in words


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("’", "'").replace("'", ""))


def _listing(running: Sequence[RunningJob]) -> str:
    named = [
        f"{narrator.job_subject(job.app, job.title, job.untrusted)} in {job.app} "
        f"({narrator.job_progress(job.done, job.total)})"
        for job in running[:MAX_LISTED]
    ]
    more = len(running) - len(named)
    listed = "; ".join(named) + (f"; and {more} more" if more > 0 else "")
    return f"{len(running)} jobs are running: {listed}."


def _say(interp: Interpreter, state: _Run, reply: str) -> AgentRunResult:
    """Code's own reply: plain facts, no model call."""
    return interp._loop._finish(
        "completed", narrator.fit(reply, state.trace.surface), state.records, 1, state.messages
    )
