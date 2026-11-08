"""The job watcher (#76): follow each running job and post one note when it ends.

A step whose receipt carries a job ref detached: the turn answered at once
("queued 40") and the ledger's ``ledger_jobs`` row tracks the job. This
watcher polls every job not yet noted through the jobs contract
(:mod:`app.fleet.jobs`):

- **every 5 s** while the job is under 2 minutes old, then **every 30 s**;
- each poll files its report on the row and, when status or ``done``/``total``
  moved, pushes a ``job`` event to the conversation's live channel
  (:func:`app.ai.turn_events.job`), so it can be followed after its turn ended;
- when the job ends (``completed``, ``failed``, ``canceled``) the narrator
  writes the note from the app's final report, code checks its counts and
  names (:func:`app.ai.narrator.job_note`), and :func:`ledger.note_job` posts
  it, **once**: the ``noted_at`` claim, the message and the job's receipts are
  one commit, so a restart or a second sweep never posts it twice;
- a job with no end after **6 hours**, or one its app no longer knows (404),
  gets code's "lost track of it" note: unknown, never failed.

Polls never use the model, have short timeouts and forward no headers. Only
the note is narrated, outside any database transaction.

A ``think`` (#78) is a job conductor owns (``app = "conductor"``): there is
no app to poll, so each sweep hands it to the thinker
(:mod:`app.services.thinker`), which runs it on the 26B and posts its note.

In-process on purpose: conductor is single-worker (CLAUDE.md), so one daemon
thread started and stopped with the app's lifespan is enough, and a restart
resumes from the ledger's rows. :meth:`JobWatcher.sweep` is also the test and
eval seam: one pass, synchronously (``force`` polls every job now, as if the
intervals had passed).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy.orm import Session, sessionmaker

from app.ai import narrator, think, turn_events
from app.ai.provider import ChatProvider, ChatResult
from app.fleet.jobs import JobError, JobGone, JobRead, JobReader, read_job
from app.fleet.manifests import Fleet
from app.services import ledger
from app.services.ledger import WatchedJob
from app.services.thinker import Thinker

logger = structlog.get_logger(__name__)

FAST_POLL = timedelta(seconds=5)
SLOW_POLL = timedelta(seconds=30)
# How long a job is polled fast before backing off.
FAST_FOR = timedelta(minutes=2)
# How long a job may run with no end before the watcher gives up on it.
GIVE_UP_AFTER = timedelta(hours=narrator.LOST_AFTER_HOURS)
# How often the thread looks for due jobs.
TICK_SECONDS = 1.0

Clock = Callable[[], datetime]
ProviderFactory = Callable[[], ChatProvider | None]


def poll_interval(age: timedelta) -> timedelta:
    """How long to wait between polls of a job this old."""
    return FAST_POLL if age < FAST_FOR else SLOW_POLL


def due(job: WatchedJob, now: datetime) -> bool:
    """Whether a job should be polled now: never polled, or its interval passed."""
    if job.polled_at is None:
        return True
    return now - job.polled_at >= poll_interval(now - job.created_at)


class JobWatcher:
    """Polls the ledger's running jobs and posts each one's note when it ends.

    ``session_factory`` opens the watcher's own sessions (never a request's);
    ``fleet`` resolves an app by name; ``reader`` reads a job (the contract's
    ``GET``; a fake in tests); ``provider`` makes the narrator's model for a
    note (``None``: code's words only). ``clock`` is UTC now.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        fleet: Fleet,
        *,
        reader: JobReader | None = None,
        provider: ProviderFactory | None = None,
        clock: Clock | None = None,
        tick: float = TICK_SECONDS,
        thinker: Thinker | None = None,
    ) -> None:
        self._sessions = session_factory
        self._fleet = fleet
        self._reader: JobReader = reader or read_job
        self._provider = provider
        self._clock: Clock = clock or (lambda: datetime.now(UTC))
        self._tick = tick
        # Runs conductor's own think jobs (#78); none: they wait.
        self._thinker = thinker
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # One sweep at a time in this process; the ledger's claim covers the rest.
        self._sweeping = threading.Lock()

    # --- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="job-watcher", daemon=True)
        self._thread.start()
        logger.info("job_watcher_started")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._thinker is not None:
            self._thinker.join(timeout=timeout)
        logger.info("job_watcher_stopped")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sweep()
            except Exception:
                logger.exception("job_watcher_sweep_failed")
            self._stop.wait(self._tick)

    # --- one pass ----------------------------------------------------------------------

    def sweep(self, *, force: bool = False) -> int:
        """Poll every due job once (every job, with ``force``); how many notes
        were posted. A job that fails is logged and left for the next pass."""
        with self._sweeping:
            now = self._clock()
            with self._sessions() as db:
                jobs = ledger.watched_jobs(db)
            posted = 0
            for job in jobs:
                if self._stop.is_set() and self._thread is not None:
                    break
                try:
                    posted += self._watch(job, now, force=force)
                except Exception:
                    logger.exception("job_watch_failed", app=job.app, ref=job.ref)
            return posted

    def _watch(self, job: WatchedJob, now: datetime, *, force: bool) -> int:
        if job.app == think.APP:
            return 0 if self._thinker is None else self._thinker.tick(job, now, force=force)
        age = now - job.created_at
        lost = age >= GIVE_UP_AFTER
        app = self._fleet.get(job.app)
        if app is None or app.agent is None:
            # Not in the fleet now (a manifest gone): nothing to poll.
            return self._lost(job, now, gone=False) if lost else 0
        if not (force or due(job, now)):
            return self._lost(job, now, gone=False) if lost else 0
        try:
            read = self._reader(app, job.ref)
        except JobGone:
            return self._lost(job, now, gone=True)
        except JobError as exc:
            logger.warning("job_poll_failed", app=job.app, ref=job.ref, error=str(exc))
            with self._sessions() as db:
                ledger.job_poll_failed(db, job.id, now)
                db.commit()
            return self._lost(job, now, gone=False) if lost else 0
        with self._sessions() as db:
            moved = ledger.job_polled(db, job.id, read, now)
            db.commit()
        if read.finished:
            return self._ended(job, read, now)
        if moved:
            self._publish(job, read, ended=False)
        if not lost:
            return 0
        latest = replace(job, status=read.status, done=read.done, total=read.total)
        return self._lost(latest, now, gone=False)

    # --- notes -------------------------------------------------------------------------

    def _ended(self, job: WatchedJob, read: JobRead, now: datetime) -> int:
        end = narrator.JobEnd(
            app=job.app,
            ref=job.ref,
            status=read.status,
            done=read.done,
            total=read.total,
            title=read.title or job.title,
            untrusted=read.untrusted if read.title else job.untrusted,
            reason=read.reason,
            record=read.record,
            asked=job.asked,
        )
        text = self._narrate(end, job.surface)
        return self._post(job, text, read=read, how="ended", now=now)

    def _lost(self, job: WatchedJob, now: datetime, *, gone: bool) -> int:
        end = narrator.JobEnd(
            app=job.app,
            ref=job.ref,
            status=job.status,
            done=job.done,
            total=job.total,
            title=job.title,
            untrusted=job.untrusted,
        )
        text = narrator.job_lost(end, gone=gone, surface=job.surface)
        return self._post(job, text, read=None, how="gone" if gone else "lost", now=now)

    def _post(
        self, job: WatchedJob, text: str, *, read: JobRead | None, how: Any, now: datetime
    ) -> int:
        with self._sessions() as db:
            try:
                message_id = ledger.note_job(db, job.id, text, read=read, how=how, now=now)
                db.commit()
            except Exception:
                db.rollback()
                raise
        if message_id is None:
            logger.info("job_note_not_posted", app=job.app, ref=job.ref, how=how)
            return 0
        self._publish(job, read, ended=True, how=how, message_id=message_id)
        return 1

    def _narrate(self, end: narrator.JobEnd, surface: str | None) -> str:
        provider = self._provider() if self._provider is not None else None
        try:
            chat = _chat(provider) if provider is not None else None
            note = narrator.job_note(end, surface=surface, chat=chat)
        finally:
            close = getattr(provider, "close", None)
            if callable(close):
                close()
        logger.info("job_note_narrated", app=end.app, ref=end.ref, path=note.path)
        return note.reply or narrator.job_fallback(end, surface)

    def _publish(
        self,
        job: WatchedJob,
        read: JobRead | None,
        *,
        ended: bool,
        how: str | None = None,
        message_id: int | None = None,
    ) -> None:
        if job.conversation_id is None:
            return
        payload: dict[str, Any] = {
            "app": job.app,
            "ref": job.ref,
            "status": read.status if read is not None else job.status,
            "done": read.done if read is not None else job.done,
            "total": read.total if read is not None else job.total,
            "title": (read.title if read is not None and read.title else job.title),
            "untrusted": read.untrusted if read is not None and read.title else job.untrusted,
            "ended": ended,
        }
        if ended:
            payload |= {"how": how, "note_message_id": message_id}
        turn_events.job(job.conversation_id, payload)


def _chat(provider: ChatProvider) -> narrator.Chat:
    def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        return provider.chat(messages, max_tokens=max_tokens)

    return chat
