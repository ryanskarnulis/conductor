"""The thinker (#78): runs each ``think`` job on the 26B and posts its one note.

A think detached in its turn (:mod:`app.ai.think`); its ``ledger_jobs`` row
(``app = "conductor"``) is followed by the job watcher like any job, which
hands it here (:meth:`Thinker.tick`) instead of polling an app. One try:

1. the row says ``working``;
2. the 26B (its own provider and sampling, thinking on, ``json_schema`` on
   the answer) reads the problem, the person's words, the calendar as of the
   ask, and what the think's inputs found, as data;
3. code validates the proposal like any plan, with one repair;
4. :func:`ledger.note_think` posts the note: the proposal with the question
   a yes answers, the 26B's answer when nothing needs to change, or code's
   honest "couldn't turn it into a plan".

**It yields.** The GPU is exclusive (``../llama-swap/config.yaml``), and the
person comes first. A try never starts while a conversation turn runs
(:mod:`app.ai.foreground`), and one in flight is cut off (the provider's
``abort``) the moment a turn starts. The call streams, so the cut-off closes
it between chunks: llama-swap finishes an in-flight request before it swaps,
and a blocked non-streamed read can't be closed from another thread, so
without streaming the turn waited out the whole thought (~2 min, measured in
``docs/agent-evals.md``). Another app's model call is not seen here and waits
behind a think. A call that fails (the 26B stopped, a timeout) counts the
same. Either way the try is ``interrupted`` (a ``not_run`` receipt on the think step) and tried again
only after :data:`app.ai.think.QUIET` with no turn and no try, at most
:data:`app.ai.think.MAX_TRIES` times in all; then code's note says it
couldn't finish. A proposal is never claimed unless one was made.

**Restarts.** A row left ``working`` by a process that died is a try cut off
(conductor restarted) and goes the same way; the note's ``noted_at`` claim
means it posts once. **Cancel**: ``cancel`` on a running think sets its flag
(:func:`app.ai.think.request_cancel`); a try in flight stops and the note says
nothing was changed.

In production a try runs in its own thread (one at a time), so the watcher
keeps polling the apps' jobs; :meth:`tick` with ``force`` (the watcher's
``sweep(force=True)``, the tests' and evals' seam) runs it to its end now,
retrying at once as if the quiet had passed.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

import structlog
from sqlalchemy.orm import Session, sessionmaker

from app import clock as household
from app.ai import foreground, think, turn_events
from app.ai.provider import ChatProvider, ChatResult, ProviderError
from app.ai.providers.llamacpp import LlamaCppProvider
from app.ai.context_capture import context_capture_from_settings
from app.config import get_settings
from app.fleet.manifests import Fleet
from app.services import ledger
from app.services.ledger import ThinkJob, WatchedJob

logger = structlog.get_logger(__name__)

Clock = Callable[[], datetime]
ProviderFactory = Callable[[], ChatProvider | None]
# How often a try in flight looks for a turn that needs the GPU, or a cancel.
WATCH_SECONDS = 0.25

TryEnd = Literal["posted", "interrupted", "cancelled", "skipped"]


def think_provider_from_settings() -> LlamaCppProvider:
    """The 26B, with its own sampling and a longer read timeout."""
    settings = get_settings()
    return LlamaCppProvider(
        settings.llamacpp_base_url,
        settings.conductor_think_model,
        timeout_seconds=settings.conductor_think_timeout_seconds,
        capture=context_capture_from_settings(),
        sampling=think.PROFILE,
        stream=True,
    )


class _Cut(Exception):
    """A try stopped from outside: a turn needs the GPU, or a cancel."""

    def __init__(self, why: Literal["turn", "cancel"]) -> None:
        super().__init__(why)
        self.why = why


class Thinker:
    """Runs think jobs; :meth:`tick` is called by the job watcher per job.

    ``provider`` makes the 26B's provider for one try (``None``: no model, so
    nothing can be tried). ``clock`` is UTC now.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        fleet: Fleet,
        *,
        provider: ProviderFactory,
        clock: Clock | None = None,
        watch_seconds: float = WATCH_SECONDS,
    ) -> None:
        self._sessions = session_factory
        self._fleet = fleet
        self._provider = provider
        self._clock: Clock = clock or (lambda: datetime.now(UTC))
        self._watch = watch_seconds
        self._lock = threading.Lock()
        # The job a try is running for, in this process (one at a time).
        self._running: int | None = None
        self._thread: threading.Thread | None = None

    def running(self) -> int | None:
        with self._lock:
            return self._running

    def join(self, timeout: float | None = None) -> None:
        """Wait for the try in flight (tests, shutdown)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # --- the watcher's hook --------------------------------------------------------------

    def tick(self, watched: WatchedJob, now: datetime, *, force: bool = False) -> int:
        """One look at a think job; how many notes were posted (0 or 1)."""
        with self._lock:
            if self._running is not None:
                return 0  # one try at a time: this one, or the next, waits its turn
        job = self._load(watched.id)
        if job is None:
            return 0
        if think.cancel_requested(job.ref) or job.status == "canceled":
            return self._post(job, think.cancelled_text(job.problem, job.surface), "cancelled")
        if job.status == "working":
            # No try of this process is running it: the process that was died.
            job = self._interrupt(job, "conductor restarted during the try")
        if job.interruptions >= think.MAX_TRIES:
            return self._give_up(job)
        if not force and not self._may_try(job, now):
            return 0
        if force:
            return self._run_to_end(job)
        with self._lock:
            if self._running is not None:
                return 0
            self._running = job.id
        self._thread = threading.Thread(
            target=self._in_thread, args=(job,), name="thinker", daemon=True
        )
        self._thread.start()
        return 0

    def _may_try(self, job: ThinkJob, now: datetime) -> bool:
        """The first try starts once no turn runs; a retry waits for the quiet."""
        if foreground.busy():
            return False
        if job.status != "interrupted":
            return True
        quiet = think.QUIET.total_seconds()
        waited = job.polled_at is None or now - job.polled_at >= think.QUIET
        return waited and foreground.quiet_for(quiet)

    def _in_thread(self, job: ThinkJob) -> None:
        try:
            self._try(job)
        except Exception:
            logger.exception("think_try_crashed", ref=job.ref)
        finally:
            with self._lock:
                self._running = None

    def _run_to_end(self, job: ThinkJob) -> int:
        """Tries until a note posts or the tries run out (the force seam)."""
        with self._lock:
            self._running = job.id
        try:
            while True:
                end = self._try(job)
                if end != "interrupted":
                    return 1 if end in ("posted", "cancelled") else 0
                reloaded = self._load(job.id)
                if reloaded is None:
                    return 0
                job = reloaded
                if job.interruptions >= think.MAX_TRIES:
                    return self._give_up(job)
        finally:
            with self._lock:
                self._running = None

    # --- one try --------------------------------------------------------------------------

    def _try(self, job: ThinkJob) -> TryEnd:
        with self._sessions() as db:
            started = ledger.think_started(db, job.id, self._clock())
            db.commit()
        if not started:
            return "skipped"
        self._publish(job, "working", ended=False)
        provider = self._provider()
        if provider is None:
            self._interrupt(job, "no model is configured for thinking")
            return "interrupted"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": think.system_prompt(self._fleet)},
            {
                "role": "user",
                "content": think.opening(
                    job.problem, job.asked, job.given, household.local_day(job.asked_at)
                ),
            },
        ]
        try:
            for attempt in range(2):
                result = self._call(provider, messages, job.ref)
                raw = result.content
                try:
                    proposal = think.parse(raw)
                except ValueError as exc:
                    found = [f"not a proposal: {exc}"]
                else:
                    found = think.problems(proposal, self._fleet)
                    if not found:
                        return self._propose(job, proposal)
                logger.warning("think_invalid", ref=job.ref, problems=found, attempt=attempt)
                messages += think.repair(raw, found)
            text = think.failed_text(job.problem, job.surface)
            return "posted" if self._post(job, text, "failed") else "skipped"
        except _Cut as cut:
            if cut.why == "cancel":
                self._post(job, think.cancelled_text(job.problem, job.surface), "cancelled")
                return "cancelled"
            self._interrupt(job, "a conversation turn needed the GPU")
            return "interrupted"
        except ProviderError as exc:
            # The 26B was stopped under it (another model call took the GPU),
            # or never answered: cut off, not failed.
            self._interrupt(job, str(exc))
            return "interrupted"
        finally:
            close = getattr(provider, "close", None)
            if callable(close):
                close()

    def _call(self, provider: ChatProvider, messages: list[dict[str, Any]], ref: str) -> ChatResult:
        """One 26B call, watched: a turn starting or a cancel cuts it off."""
        box: dict[str, Any] = {}
        done = threading.Event()

        def call() -> None:
            try:
                box["result"] = provider.chat(
                    messages,
                    enable_thinking=True,
                    max_tokens=think.MAX_TOKENS,
                    response_format=think.response_format(self._fleet),
                )
            except BaseException as exc:
                box["error"] = exc
            finally:
                done.set()

        worker = threading.Thread(target=call, name="thinker-call", daemon=True)
        worker.start()
        while not done.wait(self._watch):
            cut: Literal["turn", "cancel"] | None = None
            if think.cancel_requested(ref):
                cut = "cancel"
            elif foreground.busy():
                cut = "turn"
            if cut is not None:
                abort = getattr(provider, "abort", None)
                if callable(abort):
                    abort()
                worker.join(timeout=5)
                logger.info("think_cut_off", ref=ref, why=cut)
                raise _Cut(cut)
        if "error" in box:
            raise box["error"]
        result: ChatResult = box["result"]
        return result

    # --- endings --------------------------------------------------------------------------

    def _propose(self, job: ThinkJob, proposal: think.Proposal) -> TryEnd:
        text = think.proposal_text(job.problem, proposal, job.surface)
        if not proposal.steps:
            return "posted" if self._post(job, text, "answered") else "skipped"
        posted = self._post(
            job, text, "proposed", proposal=think.body(proposal), summary=proposal.summary
        )
        logger.info("think_proposed", ref=job.ref, steps=len(proposal.steps), posted=posted)
        return "posted" if posted else "skipped"

    def _give_up(self, job: ThinkJob) -> int:
        text = think.gave_up_text(job.problem, job.interruptions, job.surface)
        return self._post(job, text, "gave_up")

    def _interrupt(self, job: ThinkJob, reason: str) -> ThinkJob:
        with self._sessions() as db:
            cut = ledger.think_interrupted(db, job.id, reason, self._clock())
            db.commit()
        logger.info("think_interrupted", ref=job.ref, reason=reason, tries_cut=cut)
        self._publish(job, "interrupted", ended=False)
        return self._load(job.id) or job

    def _post(
        self,
        job: ThinkJob,
        text: str,
        how: ledger.ThinkEnd,
        *,
        proposal: dict[str, Any] | None = None,
        summary: str | None = None,
    ) -> int:
        with self._sessions() as db:
            try:
                message_id = ledger.note_think(
                    db,
                    job.id,
                    text,
                    how=how,
                    proposal=proposal,
                    summary=summary,
                    now=self._clock(),
                )
                db.commit()
            except Exception:
                db.rollback()
                raise
        think.forget(job.ref)
        if message_id is None:
            return 0
        status = {"proposed": "completed", "answered": "completed", "cancelled": "canceled"}
        self._publish(job, status.get(how, "failed"), ended=True, how=how, message_id=message_id)
        return 1

    def _load(self, job_id: int) -> ThinkJob | None:
        with self._sessions() as db:
            return ledger.think_job(db, job_id)

    def _publish(
        self,
        job: ThinkJob,
        status: str,
        *,
        ended: bool,
        how: str | None = None,
        message_id: int | None = None,
    ) -> None:
        if job.conversation_id is None:
            return
        payload: dict[str, Any] = {
            "app": think.APP,
            "ref": job.ref,
            "status": status,
            "done": 0,
            "total": None,
            "title": job.problem,
            "untrusted": job.untrusted,
            "ended": ended,
        }
        if ended:
            payload |= {"how": how, "note_message_id": message_id}
        turn_events.job(job.conversation_id, payload)
