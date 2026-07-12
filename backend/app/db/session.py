from collections.abc import Generator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.config import get_settings


def enable_sqlite_fk_enforcement(engine: Engine) -> None:
    """Set per-connection SQLite pragmas (no-op for other dialects).

    - ``foreign_keys = ON``: SQLite ignores foreign keys unless this is issued on
      each connection, so orphaned FKs persist silently without it.
    - ``journal_mode = WAL``: the default rollback journal blocks readers during
      any write; WAL lets concurrent clients read while a write is in flight
      without "database is locked" errors. Persistent, but cheap to re-issue.
    - ``busy_timeout``: when two writers do collide, wait for the lock instead of
      failing immediately.

    Gated on the dialect so it stays correct if ``database_url`` is ever
    non-SQLite. NullPool (below) opens a connection per checkout, so these run
    per request — all three are fast no-ops once set.
    """
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn: object, _record: object) -> None:
        cursor = dbapi_conn.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA busy_timeout = 5000")
        cursor.close()


def _build_engine() -> Engine:
    settings = get_settings()
    kwargs: dict[str, object] = {}
    if settings.database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        # NullPool: open a fresh connection per checkout and discard it on close,
        # rather than parking it in the default QueuePool. A pooled SQLite
        # connection whose implicit read transaction was never committed (a
        # read-only request just closes its session) pins a stale read snapshot,
        # so later requests reusing it never see another connection's committed
        # writes. A per-request connection always reads the latest committed
        # disk state. Cost is negligible for a local single-file SQLite app.
        kwargs["poolclass"] = NullPool
    return create_engine(settings.database_url, **kwargs)


engine: Engine = _build_engine()
enable_sqlite_fk_enforcement(engine)

SessionLocal: sessionmaker[Session] = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
