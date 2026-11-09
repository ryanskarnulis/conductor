"""The household's clock (#130): conductor stores UTC and speaks local.

The deployed container runs on UTC, so from 17:00 (18:00 in summer) to midnight
in Denver the UTC date is already tomorrow. Every "today", calendar and
recalled day goes through :mod:`app.clock`; these pin its one real clock read
and the zone, over that evening window and both DST changes.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from app import clock
from app.ai import facts, memory, recall
from app.ai.recall import EarlierCall
from app.ai.runtime import ToolCallRecord
from app.config import Settings, get_settings
from app.db import models

Pin = Callable[..., None]

# Sunday 2026-11-08, 17:03 MST: the moment #130 was found (00:03 UTC Monday).
SUNDAY_EVENING = datetime(2026, 11, 9, 0, 3, tzinfo=UTC)
SUNDAY = date(2026, 11, 8)


@pytest.fixture
def pin(monkeypatch: pytest.MonkeyPatch) -> Pin:
    def pin(at: datetime, zone: str = "America/Denver") -> None:
        monkeypatch.setattr(clock, "utc_now", lambda: at)
        monkeypatch.setattr(get_settings(), "conductor_timezone", zone)

    return pin


# --- the setting ----------------------------------------------------------------------------


def test_the_household_zone_defaults_to_denver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONDUCTOR_TIMEZONE", raising=False)
    assert Settings().conductor_timezone == "America/Denver"
    monkeypatch.setenv("CONDUCTOR_TIMEZONE", "Europe/London")
    assert Settings().conductor_timezone == "Europe/London"


@pytest.mark.parametrize("name", ["Mountain Time", "America/Nowhere", "../../etc/passwd"])
def test_an_unknown_zone_fails_loudly_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONDUCTOR_TIMEZONE", name)
    with pytest.raises(ValidationError, match="CONDUCTOR_TIMEZONE"):
        Settings()


# --- now, today and a timestamp's day ---------------------------------------------------------


def test_the_evening_is_still_today_at_home(pin: Pin) -> None:
    pin(SUNDAY_EVENING)
    assert clock.utc_now().date() == date(2026, 11, 9)  # UTC says Monday
    assert clock.local_today() == SUNDAY
    assert clock.local_now().hour == 17 and clock.local_now().utcoffset() is not None
    # The whole window, to the last minute before local midnight.
    pin(datetime(2026, 11, 9, 6, 59, tzinfo=UTC))
    assert clock.local_today() == SUNDAY
    pin(datetime(2026, 11, 9, 7, 0, tzinfo=UTC))
    assert clock.local_today() == date(2026, 11, 9)


def test_a_naive_timestamp_is_utc_and_an_aware_one_keeps_its_offset() -> None:
    naive = datetime(2026, 11, 9, 1, 30)  # how SQLite hands conductor's UTC back
    assert clock.local_day(naive) == SUNDAY
    assert clock.local_day(naive.replace(tzinfo=UTC)) == SUNDAY
    tokyo = datetime.fromisoformat("2026-11-09T10:30:00+09:00")  # the same instant
    assert clock.local_day(tokyo) == SUNDAY


def test_the_zone_is_the_households_not_the_machines(pin: Pin) -> None:
    pin(SUNDAY_EVENING, zone="Europe/London")
    assert clock.local_today() == date(2026, 11, 9)
    assert clock.local_day(SUNDAY_EVENING) == date(2026, 11, 9)
    pin(SUNDAY_EVENING, zone="UTC")
    assert clock.local_today() == date(2026, 11, 9)


def test_storage_stays_utc_on_the_same_clock(pin: Pin) -> None:
    pin(SUNDAY_EVENING)
    assert models.utcnow() == SUNDAY_EVENING
    assert models.utcnow().tzinfo is UTC


@pytest.mark.parametrize(
    ("at", "day"),
    [
        # Spring forward, 2026-03-08 02:00 MST → 03:00 MDT.
        (datetime(2026, 3, 8, 6, 30, tzinfo=UTC), date(2026, 3, 7)),  # 23:30 MST
        (datetime(2026, 3, 9, 5, 30, tzinfo=UTC), date(2026, 3, 8)),  # 23:30 MDT
        (datetime(2026, 3, 9, 6, 30, tzinfo=UTC), date(2026, 3, 9)),  # 00:30 MDT (MST: 23:30)
        # Fall back, 2026-11-01 02:00 MDT → 01:00 MST.
        (datetime(2026, 11, 1, 5, 30, tzinfo=UTC), date(2026, 10, 31)),  # 23:30 MDT
        (datetime(2026, 11, 1, 6, 30, tzinfo=UTC), date(2026, 11, 1)),  # 00:30 MDT (MST: 23:30)
        (datetime(2026, 11, 2, 6, 30, tzinfo=UTC), date(2026, 11, 1)),  # 23:30 MST (MDT: 00:30)
    ],
)
def test_the_day_follows_daylight_saving(pin: Pin, at: datetime, day: date) -> None:
    pin(at)
    assert clock.local_today() == day
    assert clock.local_day(at) == day
    assert clock.local_day(at.replace(tzinfo=None)) == day


# --- what conductor says ----------------------------------------------------------------------


def test_recall_labels_tonight_as_today_not_tomorrow(pin: Pin) -> None:
    pin(SUNDAY_EVENING)
    tonight = datetime(2026, 11, 9, 1, 30, tzinfo=UTC)  # Sunday 18:30 MST
    assert memory.day(tonight) == "Sunday 2026-11-08"
    assert memory.day(tonight.replace(tzinfo=None)) == "Sunday 2026-11-08"
    assert memory.cues(tonight, clock.local_today())[:2] == ["today", "Sunday"]
    # Saturday 22:00 MST is Sunday in UTC: still yesterday at home.
    saturday_night = datetime(2026, 11, 8, 5, 0, tzinfo=UTC)
    assert memory.cues(saturday_night, clock.local_today())[:3] == [
        "yesterday",
        "Saturday",
        "2026-11-07",
    ]
    # A date is already a day: it passes through.
    assert memory.day(SUNDAY) == "Sunday 2026-11-08"
    hit = memory.Hit(1, 1, tonight, "go with the steel rack")
    result, (entry,) = memory.record(memory.Recalled("rack", (hit,)), clock.local_today())
    assert entry["day"] == "Sunday 2026-11-08" and entry["cues"][0] == "today"
    assert entry["when"] == tonight.isoformat()  # the stored instant stays UTC
    assert result.startswith("Sunday 2026-11-08, your words:")


def test_earlier_calls_and_saved_facts_say_the_local_day(pin: Pin) -> None:
    pin(SUNDAY_EVENING)
    tonight = datetime(2026, 11, 9, 1, 30)  # naive, as stored
    call = EarlierCall(
        when=tonight,
        asked="grab Odessa",
        record=ToolCallRecord(tool="ask_music", arguments={"message": "x"}, result="Saved."),
    )
    assert recall._hit_line(call).startswith("Sunday 2026-11-08 (as of then)")
    assert facts.source_line(tonight) == "you said it Sunday 2026-11-08"
