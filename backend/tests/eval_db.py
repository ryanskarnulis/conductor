"""A clean database for every live-eval sample (#103, #122).

Samples share one app and one database, and a sample's "new conversation"
must not find another sample's conversations: #64's cross-conversation recall
reads them, so the second ask of an identical golden utterance saw the first's
answered call, and the compiled planner answered from it instead of routing.
"""

from __future__ import annotations

from collections.abc import Callable

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
