"""Agent chat API: the timeline, its history, and the message → turn round trip.

There is one timeline (#82, ``GET /agent/timeline``): every surface writes to
it, ``POST /agent/conversations`` answers it (there is no new conversation)
and it can't be deleted ("forget that" is how something goes, #81). Legacy
conversations are gone from the API; recall still reads them. A message may
be ``incognito``: its words and reply are never written (:mod:`app.ai.incognito`).

``POST /agent/conversations/{id}/messages`` is the one model-calling endpoint:
it stores the user turn, binds the delegation context (DB-backed thread store,
per-turn budget, audit driver), runs the turn (compiled plan, #75)
synchronously, then
stores and returns the assistant turn. While it blocks,
``GET /agent/conversations/{id}/events`` streams what the turn is doing as
Server-Sent Events (#38) and ``GET …/activity`` answers the same question
for a poll (``app/api/turn_activity.py``), so the UI can show "asking chess…"
and each step's receipt as it lands; ``POST …/cancel`` stops it (#66,
``app/ai/stop.py``).

Deltas from PCC's routes_agent: no ``X-Agent-Actor`` handling — conductor is
the delegation root (STANDARD.md depth-1), so no other agent may drive this
API and every run carries conductor's own identity; plus the delegation-context
binding and the activity endpoint, which PCC's local-tool agent doesn't need.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Generator, Sequence
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response
from fastapi.sse import EventSourceResponse, format_sse_event
from sqlalchemy.orm import Session, sessionmaker

from app import clock
from app.ai import (
    confirmation,
    foreground,
    incognito,
    job_control,
    narrator,
    think,
    thread_note,
    turn_context,
    turn_events,
)
from app.ai.plan import RunningJob
from app.ai.interpreter import Interpreter, conductor_from_settings
from app.ai.runtime import Activity, AgentRunFailed
from app.ai.plan import PausedPlan
from app.ai.providers.llamacpp import provider_from_settings
from app.ai.state_block import Now
from app.api import turn_activity
from app.api.rate_limit import rate_limit
from app.config import get_settings
from app.db.models import Conversation, ConversationMessage
from app.db.session import SessionLocal, get_db
from app.fleet.context import DelegationContext, ThreadStore, use_delegation_context
from app.fleet import jobs as fleet_jobs
from app.fleet.jobs import JobCanceller, JobError, JobGone
from app.fleet.manifests import Fleet
from app.fleet.thread_cleanup import SubagentThreadCleanup
from app.fleet.thread_store import DbThreadStore
from app.schemas.conversations import (
    ConversationCreate,
    ConversationDetail,
    ConversationRead,
    JobCancelCreate,
    JobCancelRead,
    MessageCreate,
    MessageExchange,
    MessageRead,
    NoteThreadRead,
    RunCancelRead,
    TimelineDetail,
    TimelineHit,
    ToolCallRead,
    TurnActivityRead,
)
from app.services import conversations as conversations_service
from app.services import facts as facts_service
from app.services import ledger
from app.services import memory as memory_service
from app.services import threads as threads_service
from app.services import timeline as timeline_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


def get_conductor(request: Request) -> Generator[Interpreter, None, None]:
    """The turn runtime over the configured provider and the fleet the
    lifespan discovered (503 before discovery); tests override this
    dependency."""
    fleet: Fleet | None = getattr(request.app.state, "fleet", None)
    if fleet is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the fleet has not been discovered yet",
        )
    provider = provider_from_settings()
    try:
        yield conductor_from_settings(provider, fleet)
    finally:
        provider.close()


def get_thread_store() -> ThreadStore:
    """The DB-backed subagent-thread map; tests override this dependency."""
    return DbThreadStore(SessionLocal)


def get_thread_cleanup(request: Request) -> SubagentThreadCleanup | None:
    """Best-effort subagent-thread cleanup over the discovered fleet (``None``
    before discovery); tests override this dependency."""
    fleet = getattr(request.app.state, "fleet", None)
    if fleet is None:
        return None
    return SubagentThreadCleanup(fleet, SessionLocal)


def get_session_factory() -> sessionmaker[Session]:
    """Where the live channel reads the running jobs a connecting page missed
    (#90); tests override this dependency."""
    return SessionLocal


def get_job_canceller() -> JobCanceller:
    """The jobs contract's cancel (#76), by name at call time so a test may
    swap it; tests override this dependency."""
    return fleet_jobs.cancel_job


def get_job_fleet(request: Request) -> Fleet:
    """The fleet discovered at startup; tests override this dependency."""
    fleet: Fleet | None = getattr(request.app.state, "fleet", None)
    if fleet is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the fleet has not been discovered yet",
        )
    return fleet


def _stored_turns(db: Session, conversation_id: int, since: Any) -> list[dict[str, Any]]:
    """The turns before this one since the quiet gap; a failed read is no
    history, never no turn."""
    try:
        return conversations_service.stored_turns(db, conversation_id, since=since)
    except Exception as exc:
        logger.warning("stored_turns_failed", error=str(exc), error_type=type(exc).__name__)
        return []


def _window(db: Session, conversation_id: int) -> Any:
    """Where this turn's "this conversation" starts (#82); a failed read is
    the whole conversation, as before the timeline."""
    try:
        return conversations_service.window_since(db, conversation_id)
    except Exception as exc:
        logger.warning("window_failed", error=str(exc), error_type=type(exc).__name__)
        return None


def _paused_plan(db: Session, conversation_id: int, since: Any) -> PausedPlan | None:
    """Last turn's compiled plan waiting on the person; a failed read is none."""
    try:
        return ledger.paused_plan(db, conversation_id, since=since)
    except Exception as exc:
        logger.warning("paused_plan_failed", error=str(exc), error_type=type(exc).__name__)
        return None


def _messages(db: Session, conversation: Conversation) -> list[MessageRead]:
    """A conversation's messages as the page reads them, each tagged with its
    turn's surface and incognito (#82): the timeline's newest."""
    rows = conversations_service.list_messages(
        db, conversation.id, newest=conversations_service.TIMELINE_MESSAGES
    )
    tags = timeline_service.message_tags(db, (row.id for row in rows))
    # A side thread's result note (#139) names the thread it came from.
    noted = threads_service.note_threads(
        db, [row.id for row in rows if row.stop_reason == thread_note.THREAD_NOTE]
    )
    reads = [_read(row, tags.get(row.id)) for row in rows]
    for read in reads:
        if read.id in noted:
            read.thread = NoteThreadRead(id=noted[read.id].id, title=noted[read.id].title)
    return reads


def _read(message: ConversationMessage, tag: timeline_service.Tag | None) -> MessageRead:
    read = MessageRead.model_validate(message)
    if tag is not None:
        read.surface, read.incognito = tag.surface, tag.incognito
    return read


def _live(
    message: ConversationMessage, *, surface: str | None, content: str | None, calls: Any
) -> MessageRead:
    """An incognito turn's message as this response carries it: the words and
    reply the person is reading now, which were never written (#82)."""
    read = MessageRead.model_validate(message)
    read.surface, read.incognito = surface, True
    read.content = content
    if calls is not None:
        read.tool_calls = [ToolCallRead.model_validate(record.model_dump()) for record in calls]
    return read


def _get_or_404(db: Session, conversation_id: int) -> Conversation:
    conversation = conversations_service.get_conversation(db, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


def _writable_or_409(conversation: Conversation) -> None:
    """A closed side thread (#139) takes no more turns or notes."""
    if threads_service.is_thread(conversation) and threads_service.is_closed(conversation):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="this side thread is closed; its result is in the timeline",
        )


@router.get("/conversations", response_model=list[ConversationRead])
def list_conversations(db: Session = Depends(get_db)) -> Sequence[Conversation]:
    """The timeline, once it exists (#82); legacy conversations never list."""
    return conversations_service.list_conversations(db)


@router.post("/conversations", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
def create_conversation(data: ConversationCreate, db: Session = Depends(get_db)) -> Conversation:
    """There is no new conversation (#82): the timeline, created on first use."""
    conversation = conversations_service.create_conversation(db, title=data.title)
    db.commit()
    db.refresh(conversation)
    return conversation


@router.get("/timeline", response_model=TimelineDetail)
def get_timeline(db: Session = Depends(get_db)) -> TimelineDetail:
    """The one timeline every surface writes to (#82), created clean on first
    use, with its newest messages, each tagged with its surface."""
    conversation = conversations_service.timeline(db)
    db.commit()
    db.refresh(conversation)
    return TimelineDetail(
        **ConversationRead.model_validate(conversation).model_dump(),
        messages=_messages(db, conversation),
        quiet_gap_seconds=int(conversations_service.QUIET_GAP.total_seconds()),
        incognito_notice=incognito.NOTICE,
    )


@router.get("/timeline/search", response_model=list[TimelineHit])
def search_timeline(
    q: str = Query(
        min_length=timeline_service.MIN_QUERY_CHARS, max_length=timeline_service.MAX_QUERY_CHARS
    ),
    db: Session = Depends(get_db),
) -> list[TimelineHit]:
    """The timeline's turns whose words, replies or receipt titles match ``q``,
    newest first. **No model runs.** Incognito and forgotten turns are never hits."""
    conversation = conversations_service.timeline(db)
    db.commit()
    hits = timeline_service.search(db, conversation.id, q.strip())
    logger.info("timeline_search", hits=len(hits))
    return [
        TimelineHit(
            turn_id=hit.turn_id,
            message_id=hit.message_id,
            at=hit.at,
            surface=hit.surface,
            kind=hit.kind,
            said=hit.said,
            reply=hit.reply,
            titles=hit.titles,
            matched=hit.matched,  # type: ignore[arg-type]
        )
        for hit in hits
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: int, db: Session = Depends(get_db)) -> ConversationDetail:
    conversation = _get_or_404(db, conversation_id)
    return ConversationDetail(
        **ConversationRead.model_validate(conversation).model_dump(),
        messages=_messages(db, conversation),
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    thread_cleanup: SubagentThreadCleanup | None = Depends(get_thread_cleanup),
) -> None:
    conversation = _get_or_404(db, conversation_id)
    if conversations_service.is_timeline(conversation):
        # The timeline is never deleted (#82): "forget that" (#81) is how a turn goes.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="the timeline can't be deleted; say “forget that” to take something out",
        )
    conversations_service.soft_delete_conversation(db, conversation)
    db.commit()
    logger.info("conversation_deleted", conversation_id=conversation_id)
    if thread_cleanup is not None:
        # After the response: the person's delete never waits on the fleet.
        background_tasks.add_task(thread_cleanup, str(conversation_id))


@router.get("/conversations/{conversation_id}/activity", response_model=TurnActivityRead)
def get_turn_activity(conversation_id: int, db: Session = Depends(get_db)) -> TurnActivityRead:
    """What the conversation's in-flight turn is doing right now, if any.

    The poll target while ``POST …/messages`` blocks — v1's no-SSE progress
    channel. 404s for an unknown conversation so a stale tab learns the thread
    is gone rather than polling forever.
    """
    _get_or_404(db, conversation_id)
    activity = turn_activity.get(conversation_id)
    if activity is None:
        return TurnActivityRead(active=False)
    return TurnActivityRead(
        active=True,
        kind=activity.kind,
        tool=activity.tool,
        iteration=activity.iteration,
        elapsed_seconds=round(time.monotonic() - activity.started_at, 1),
        ack=activity.ack,
    )


@router.get(
    "/conversations/{conversation_id}/events",
    response_class=Response,
    responses={200: {"content": {"text/event-stream": {}}, "description": "The turn's events"}},
)
def stream_turn_events(
    conversation_id: int,
    request: Request,
    # Closed before the stream starts: a stream never holds a database connection.
    db: Session = Depends(get_db, scope="function"),
    sessions: sessionmaker[Session] = Depends(get_session_factory),
) -> EventSourceResponse:
    """The conversation's turn as it happens, over Server-Sent Events (#38).

    First a ``snapshot`` of the turn in flight (``active: false`` when there is
    none), then each event as it is published: ``turn_started``, ``activity``
    (the poll's beat), ``plan`` (the validated plan's steps and where this turn
    starts in it), ``step_started``, ``receipt`` (the step as the ledger
    journals it), ``paused``, ``resumed`` (a turn picking up a plan that was
    waiting on the person) and ``turn_ended`` (after the ledger committed
    the turn), each with a rising ``id``. A quiet stream gets a heartbeat
    comment. A job the conversation started (#76) streams ``job`` events
    (progress, then its end with the note's message id), and the stream stays
    open past ``turn_ended`` until the last running job ends; the first
    snapshot also lists the jobs the ledger says are still running (#90), so a
    page that connects later finds them. The stream ends
    after ``turn_ended``, or when no turn begins in time; a reconnect starts
    over from a snapshot. 404s for an unknown
    conversation. Runs on the event loop and never holds a worker thread.
    """
    _get_or_404(db, conversation_id)

    def seed() -> list[dict[str, Any]]:
        # Its own session, read after the stream subscribed: never the request's.
        with sessions() as session:
            return ledger.running_job_events(session, conversation_id, limit=turn_activity.MAX_JOBS)

    return EventSourceResponse(
        _sse(request, conversation_id, seed),
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _sse(
    request: Request, conversation_id: int, seed: turn_activity.JobSeed
) -> AsyncIterator[bytes]:
    events = turn_activity.follow(conversation_id, seed=seed)
    try:
        async for event in events:
            if isinstance(event, turn_activity.Heartbeat):
                if await request.is_disconnected():
                    return
                yield b": heartbeat\n\n"
                continue
            yield format_sse_event(
                data_str=json.dumps(event.data, default=str),
                event=event.kind,
                id=str(event.id),
                retry=turn_activity.RETRY_MS if event.kind == "snapshot" else None,
            )
    finally:
        await events.aclose()


@router.post("/conversations/{conversation_id}/cancel", response_model=RunCancelRead)
def cancel_turn(
    conversation_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
) -> RunCancelRead:
    """Stop the conversation's in-flight turn (#66); **404** for an unknown thread.

    Idempotent and never blocks: it sets the turn's Stop and returns. The turn
    stops at its next boundary (a model call or a step) and the
    turn's own ``POST …/messages`` answers 200 ``cancelled``, its reply the
    code-kept record of what ran. A delegate call in flight gets the app's
    cancel after the response, best-effort. With nothing running the answer
    is ``cancel_requested: false`` and nothing is remembered.
    """
    _get_or_404(db, conversation_id)
    stop = turn_activity.request_stop(conversation_id)
    if stop is not None:
        background_tasks.add_task(stop.pass_on)
    logger.info("agent_turn_cancel", conversation_id=conversation_id, live=stop is not None)
    return RunCancelRead(cancel_requested=stop is not None)


@router.post(
    "/conversations/{conversation_id}/jobs/cancel",
    response_model=JobCancelRead,
    dependencies=[Depends(rate_limit("agent_notes", per_min_attr="fleet_actions_per_min"))],
)
def cancel_job(
    conversation_id: int,
    data: JobCancelCreate,
    db: Session = Depends(get_db),
    fleet: Fleet = Depends(get_job_fleet),
    canceller: JobCanceller = Depends(get_job_canceller),
) -> JobCancelRead:
    """Stop a job from its card (#90). **No model runs.**

    The same cancel as saying "stop" (#76, :mod:`app.ai.job_control`): the
    owning app's ``POST …/jobs/{ref}/cancel``, whose answer carries the job's
    ``done`` count at that moment. The tap is the person's own act, so it is
    recorded as their note turn (as ``POST …/notes`` does) with the cancel
    and its receipt on the ledger; nothing is put in conductor's mouth. The
    job ends at the app's next safe boundary and the watcher posts its one
    note then.

    **404** for a job this conversation didn't start; **502** when the app
    can't be reached or won't answer (nothing is recorded: nothing happened);
    **410** when the app no longer knows the job. A job that had already
    ended answers 200 with ``cancel_requested: false`` and how far it got.
    """
    conversation = _get_or_404(db, conversation_id)
    job = ledger.conversation_job(db, conversation_id, data.app, data.ref)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    app = fleet.get(job.app)
    if job.app == think.APP:
        # Conductor's own think (#78): the thinker stops it, no app to ask.
        answer = job_control.cancel_think(job.ref, job.title, job.untrusted)
    elif app is None or app.agent is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"conductor can't reach {job.app}"
        )
    else:
        try:
            answer = canceller(app, job.ref)
        except JobGone as exc:
            raise HTTPException(status_code=status.HTTP_410_GONE, detail=str(exc)) from exc
        except JobError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    running = RunningJob(
        app=job.app,
        ref=job.ref,
        status=job.status,
        done=job.done,
        total=job.total,
        title=job.title,
        untrusted=job.untrusted,
    )
    named = answer.title or job.title
    outside = answer.untrusted if answer.title else job.untrusted
    subject = narrator.job_subject(job.app, named, outside)
    note = conversations_service.append_user_message(
        db, conversation, job_control.card_note(job.app, subject, answer)
    )
    ledger.note_job_cancel(db, conversation, note, job, job_control.cancel_receipt(running, answer))
    db.commit()
    db.refresh(note)
    db.refresh(job)
    live = answer.cancel_requested is not False
    if live and job.noted_at is None:
        # The card shows the answer now and the watcher's one note ends it; a
        # job that had already ended is the watcher's to close, never reopened.
        turn_events.job(
            conversation_id,
            {
                "app": job.app,
                "ref": job.ref,
                "status": job.status,
                "done": job.done,
                "total": job.total,
                "title": job.title,
                "untrusted": job.untrusted,
                "ended": False,
                "cancel_requested": True,
            },
        )
    logger.info(
        "agent_job_cancel",
        conversation_id=conversation_id,
        app=job.app,
        ref=job.ref,
        status=answer.status,
        done=answer.done,
        live=live,
    )
    return JobCancelRead(
        app=job.app,
        ref=job.ref,
        status=answer.status,
        done=answer.done,
        total=answer.total,
        cancel_requested=live,
        message=MessageRead.model_validate(note),
    )


@router.post(
    "/conversations/{conversation_id}/notes",
    response_model=MessageRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit("agent_notes", per_min_attr="fleet_actions_per_min"))],
)
def post_note(
    conversation_id: int,
    data: MessageCreate,
    db: Session = Depends(get_db),
) -> Conversation | Any:
    """Record something the person did in the UI. **No model runs.**

    Some answers are given by clicking rather than by talking — filing a song
    from the sort panel, which reaches the app through
    ``app/api/routes_fleet.py`` and never passes through a conductor turn. The
    act happened, so the thread has to show it: a transcript with a question in
    it and no answer under it is a worse record than none, and the next turn's
    model context would contradict what is on screen.

    Stored as the person's own turn, because the person is who acted. Nothing
    is put in conductor's mouth — inventing an assistant reply it never made is
    the one thing a truthful transcript cannot do.
    """
    conversation = _get_or_404(db, conversation_id)
    _writable_or_409(conversation)
    note = conversations_service.append_user_message(
        db, conversation, data.content, hidden=data.incognito
    )
    ledger.note_turn(db, conversation, note, surface=data.surface, hidden=data.incognito)
    db.commit()
    db.refresh(note)
    logger.info("agent_note", conversation_id=conversation_id, message_id=note.id)
    if data.incognito:
        return _live(note, surface=data.surface, content=data.content, calls=None)
    return _read(note, timeline_service.Tag(data.surface, False))


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageExchange,
    dependencies=[Depends(rate_limit("agent_messages", per_min_attr="agent_messages_per_min"))],
)
def post_message(
    conversation_id: int,
    data: MessageCreate,
    db: Session = Depends(get_db),
    conductor: Interpreter = Depends(get_conductor),
    thread_store: ThreadStore = Depends(get_thread_store),
) -> MessageExchange:
    """Store the user turn, run it, store and return the assistant turn.

    The user message is committed *before* the turn runs: a turn can take
    minutes (each step may wrap an app agent's full loop) and a provider
    failure — surfaced as 502 — must not swallow what the user said. A failed
    run (:class:`AgentRunFailed`: 502 for the provider, 500 for a crash) is
    stored as a truthful partial turn before the error goes out, so a write an
    app made before the failure stays in the transcript and the next turn's
    recall (#52). No chat turns are replayed (#64): the turn opens with the
    state block and recall (:mod:`app.ai.turn_context`), built before the user
    turn is stored, so it is "before this turn". The
    delegation context is bound for exactly the run: the DB-backed thread
    store keyed by this conversation, the per-turn per-app call budget, and
    the ``agent:loop`` audit driver. The ledger (#69) opens the turn with the
    user message and closes it with the assistant turn, in the same commits.

    An ``incognito`` turn (#82) runs the same, but its words and reply are
    never written, its log lines carry none, and neither the trace nor the
    capture records it; this response carries the reply the person reads now.
    """
    with incognito.turn(data.incognito):
        return _post_message(conversation_id, data, db, conductor, thread_store)


def _post_message(
    conversation_id: int,
    data: MessageCreate,
    db: Session,
    conductor: Interpreter,
    thread_store: ThreadStore,
) -> MessageExchange:
    conversation = _get_or_404(db, conversation_id)
    _writable_or_409(conversation)
    hidden = data.incognito
    # A side thread (#139) is read whole, with what it carries in from the
    # timeline; the timeline is "this conversation" since its quiet gap
    # (#82), and older turns reach the planner only as cross-day recall.
    thread = conversation if threads_service.is_thread(conversation) else None
    since = None if thread is not None else _window(db, conversation.id)
    turns = _stored_turns(db, conversation.id, since)
    if data.heard is not None and not hidden:
        # Barge-in (#77): the person spoke over the last reply. Kept on its
        # turn before the opening is built, so this turn's State says it.
        ledger.record_heard(db, conversation.id, data.heard.model_dump(), since=since)
    turn_opening = turn_context.build(
        clock.local_today(),
        Now(surface=data.surface, panel=data.panel),
        data.content,
        history=lambda: turns,
        earlier=lambda: conversations_service.earlier_calls(db, conversation.id, since=since),
        entities=lambda: (
            threads_service.turn_entities(db, thread)
            if thread is not None
            else ledger.recent_entities(db, conversation.id, since=since)
        ),
        jobs=lambda: ledger.running_job_lines(db, conversation.id),
        matches=lambda: memory_service.matches(db, data.content, conversation.id, since=since),
        facts=lambda: (
            threads_service.turn_facts(db, thread, data.content)
            if thread is not None
            else facts_service.for_state(db, data.content)
        ),
        # Incognito shows nothing of it (#82).
        heard=None if hidden else lambda: ledger.last_heard(db, conversation.id, since=since),
    )
    user_message = conversations_service.append_user_message(
        db, conversation, data.content, hidden=hidden
    )
    turn = ledger.open_turn(
        db, conversation, user_message, surface=data.surface, panel=data.panel, hidden=hidden
    )
    db.commit()

    context = DelegationContext(
        # The apps' own threads follow the window: after a quiet gap each app
        # starts a fresh one, as it did for a new conversation (#82).
        master_conversation_id=conversations_service.thread_key(db, conversation.id, since),
        thread_store=thread_store,
        calls_per_turn_per_app=get_settings().conductor_delegate_calls_per_turn_per_app,
        # Only a question the person saw last turn can take a yes (#68).
        pending_confirm=confirmation.pending(turns),
    )

    def report(activity: Activity) -> None:
        turn_activity.update(
            conversation_id, kind=activity.kind, tool=activity.tool, ack=activity.ack
        )

    stop = turn_activity.begin(conversation_id)
    # The GPU is the person's while their turn runs: a think yields (#78).
    foreground.entered()

    def publish(kind: str, payload: dict[str, Any]) -> None:
        turn_activity.publish(conversation_id, stop, kind, payload)

    # How the turn ended, once the ledger has it: the stream's last word,
    # sent only after the commit, so nothing says the turn ended before its
    # record exists. An unexpected failure ends it with no stop reason.
    ended: dict[str, Any] = {}
    try:
        try:
            with use_delegation_context(context), turn_events.publishing(publish):
                run = conductor.run(
                    data.content,
                    context=turn_opening,
                    surface=data.surface,
                    on_activity=report,
                    conversation_id=str(conversation_id),
                    stop=stop,
                    paused=_paused_plan(db, conversation.id, since),
                    self_facts=lambda: "\n\n".join(
                        (
                            ledger.self_report(db, conversation.id, since=since),
                            facts_service.self_lines(db),
                        )
                    ),
                    control_view=lambda: ledger.control_view(db, conversation.id, since=since),
                    recall=lambda query: memory_service.recall(db, query),
                    memory_app=lambda request, words: facts_service.handle(
                        db,
                        request,
                        words,
                        conversation_id=conversation.id,
                        turn_id=turn.id if turn is not None else None,
                        hidden=hidden,
                        since=since,
                    ),
                    # Incognito: no saved fact rides an app's ask (#134's attach).
                    saved_facts=None if hidden else lambda: facts_service.texts(db),
                )
        except AgentRunFailed as exc:
            logger.error(
                "agent_run_failed",
                conversation_id=conversation_id,
                stop_reason=exc.result.stop_reason,
                tool_calls=len(exc.result.tool_calls),
                error=str(exc),
            )
            failed_message = None
            if exc.result.tool_calls:
                # Something ran: keep the record. A failure before any call leaves
                # the user turn alone, as before.
                failed_message = conversations_service.append_assistant_message(
                    db, conversation, exc.result, hidden=hidden
                )
            # The ledger keeps the turn either way: its words, how it ended, what ran.
            ledger.close_turn(db, turn, exc.result, failed_message)
            db.commit()
            ended.update(
                stop_reason=exc.result.stop_reason,
                message_id=failed_message.id if failed_message else None,
            )
            raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc

        assistant_message = conversations_service.append_assistant_message(
            db, conversation, run, hidden=hidden
        )
        ledger.close_turn(db, turn, run, assistant_message)
        db.commit()
        ended.update(stop_reason=run.stop_reason, message_id=assistant_message.id)
    finally:
        foreground.left()
        turn_activity.end(conversation_id, stop, **ended)
    if hidden:
        # The person reads the reply now; it was never written (#82).
        return MessageExchange(
            user_message=_live(
                user_message, surface=data.surface, content=data.content, calls=None
            ),
            assistant_message=_live(
                assistant_message, surface=data.surface, content=run.reply, calls=run.tool_calls
            ),
        )
    tag = timeline_service.Tag(data.surface, False)
    return MessageExchange(
        user_message=_read(user_message, tag),
        assistant_message=_read(assistant_message, tag),
    )
