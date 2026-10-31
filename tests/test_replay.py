from datetime import date

import pytest

from driveguard import replay


def test_cursor_advances_one_data_day_per_real_day():
    assert replay.cursor(today=date(2026, 11, 1), anchor="2025-01-01@2026-11-01") == date(2025, 1, 1)
    assert replay.cursor(today=date(2026, 11, 15), anchor="2025-01-01@2026-11-01") == date(2025, 1, 15)
    assert replay.cursor(today=date(2026, 11, 1), anchor="") is None


def test_resolve_as_of_precedence(monkeypatch):
    latest = date(2025, 3, 31)
    assert replay.resolve_as_of("2025-02-02", latest, anchor="2025-01-01@2026-11-01") == date(2025, 2, 2)
    monkeypatch.setenv("DG_REPLAY_ANCHOR", "")
    assert replay.resolve_as_of(None, latest) == latest
    # anchor far in the past relative to the data -> clamped to the latest day available
    assert replay.resolve_as_of(None, latest, anchor="2020-01-01@2020-01-01") == latest


def test_bad_anchor():
    with pytest.raises(ValueError):
        replay.parse_anchor("2025-01-01")
