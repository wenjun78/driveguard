"""Replay clock.

Backblaze publishes history, not a live feed. To operate the system as if it were live - and to let the console
show a fleet that moves - the daily job can advance through the historical subset one data-day per real day.

``DG_REPLAY_ANCHOR="<data_date>@<real_date>"`` pins a data day to a calendar day; "today" in data terms is then
``data_date + (today_utc - real_date)``, clamped to what the gold table holds. Scoring, monitoring and the console
all treat that cursor as *now*: nothing after it is scored, and labels are only considered matured once the
cursor has passed ``t + horizon``. Without an anchor, "now" is simply the latest day in gold.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone


def parse_anchor(value: str | None) -> tuple[date, date] | None:
    if not value:
        return None
    data_s, _, real_s = value.partition("@")
    if not real_s:
        raise ValueError("DG_REPLAY_ANCHOR must look like 2025-03-01@2026-11-01 (data_date@real_date)")
    return date.fromisoformat(data_s.strip()), date.fromisoformat(real_s.strip())


def cursor(today: date | None = None, anchor: str | None = None) -> date | None:
    """The data day that corresponds to today, or None when no anchor is configured."""
    parsed = parse_anchor(anchor if anchor is not None else os.environ.get("DG_REPLAY_ANCHOR"))
    if parsed is None:
        return None
    data_date, real_date = parsed
    today = today or datetime.now(timezone.utc).date()
    return data_date + (today - real_date)


def resolve_as_of(explicit: str | None, latest_in_gold: date, anchor: str | None = None) -> date:
    """Explicit date wins; else the replay cursor (clamped to the data); else the latest day in gold."""
    if explicit:
        return date.fromisoformat(explicit)
    c = cursor(anchor=anchor)
    if c is None:
        return latest_in_gold
    return min(c, latest_in_gold)
