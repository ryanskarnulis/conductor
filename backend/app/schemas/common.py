"""Shared schema field types. Trimmed from PCC's to the two conductor needs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator, StringConstraints


def _assume_utc(value: datetime) -> datetime:
    """Stamp naive timestamps as UTC so serialized JSON always carries an offset.

    Every timestamp the app writes is UTC, but a row written by SQLite's
    ``CURRENT_TIMESTAMP`` fallback reads back naive. Serializing those without
    an offset makes JS ``new Date(...)`` parse them as *local* time, skewing
    displayed times by the UTC offset.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# The one datetime type read schemas should use for DB-sourced timestamps.
UTCDateTime = Annotated[datetime, AfterValidator(_assume_utc)]
