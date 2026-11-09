"""The household's clock (#130): conductor stores UTC and speaks local.

One zone for the household, ``CONDUCTOR_TIMEZONE`` (an IANA name, default
``America/Denver``, env only). Every place conductor says "today" or turns a
timestamp into a day goes through here, so a Sunday evening in Denver is never
already Monday; what is stored stays UTC. :func:`utc_now` is the one real
clock read (``models.utcnow`` writes with it too), so tests pin it.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from app.config import get_settings


def zone() -> ZoneInfo:
    return ZoneInfo(get_settings().conductor_timezone)


def utc_now() -> datetime:
    return datetime.now(UTC)


def local_now() -> datetime:
    return utc_now().astimezone(zone())


def local_today() -> date:
    return local_now().date()


def local_day(when: datetime) -> date:
    """The household's day for a timestamp; a naive one is UTC (how SQLite
    hands conductor's own timestamps back)."""
    aware = when if when.tzinfo is not None else when.replace(tzinfo=UTC)
    return aware.astimezone(zone()).date()
