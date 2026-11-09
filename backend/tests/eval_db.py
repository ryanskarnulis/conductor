"""A clean database for every live-eval sample (#103, #122), and time passing (#82).

Samples share one app and one database, and a sample's "new conversation"
must not find another sample's conversations: #64's cross-conversation recall
reads them, so the second ask of an identical golden utterance saw the first's
answered call, and the compiled planner answered from it instead of routing.

Since the timeline (#82) there is no new conversation: "another day" is the
same timeline after a quiet gap. :func:`age` makes one by moving the record
back a day.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

from sqlalchemy import DateTime, select, update
from sqlalchemy.orm import Session

from app.api import turn_activity
from app.db.models import Base


def database_wiper(db_session: Session) -> Callable[[], None]:
    """Empty every table (and the in-memory turn registry, keyed by the
    conversation ids the wipe frees) so the next sample starts clean."""

    def wipe() -> None:
        db_session.rollback()
        for table in reversed(Base.metadata.sorted_tables):
            db_session.execute(table.delete())
        db_session.commit()
        db_session.expunge_all()
        turn_activity._reset()

    return wipe


# How far back :func:`age` moves the record: past the timeline's quiet gap
# and onto another day, so a later turn reads it as earlier.
AGED_BY = timedelta(days=1)

_STAMPS = frozenset({"created_at", "updated_at", "last_seen_at", "ended_at", "polled_at"})


def age(db_session: Session, by: timedelta = AGED_BY) -> None:
    """Time passes (#82): every timestamp on record moves ``by`` into the
    past, so the timeline's next turn starts after a quiet gap and finds what
    came before only through cross-day recall. Moving the record back, not
    the clock forward, keeps "today" the fake world's today."""
    db_session.flush()
    for table in Base.metadata.sorted_tables:
        stamps = [
            column
            for column in table.columns
            if column.name in _STAMPS and isinstance(column.type, DateTime)
        ]
        if not stamps or "id" not in table.columns:
            continue
        key = table.columns["id"]
        # In Python, row by row: SQLite has no arithmetic on stored datetimes.
        for row in db_session.execute(select(key, *stamps)).all():
            moved = {
                column.name: value - by
                for column, value in zip(stamps, row[1:], strict=True)
                if value is not None
            }
            if moved:
                db_session.execute(update(table).where(key == row[0]).values(moved))
    db_session.commit()
    db_session.expire_all()


def database_ager(db_session: Session) -> Callable[[], None]:
    """:func:`age` as the session driver's "another day" (``NewConversation``)."""
    return lambda: age(db_session)
