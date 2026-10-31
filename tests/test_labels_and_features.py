"""Label semantics (horizon, censoring, failure row) and feature time safety."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from pyspark.sql import functions as F

from driveguard import gold, silver

START = date(2024, 1, 1)


def _rows(serial: str, days: range, fail_day: int | None, model="M1", smart5=None):
    out = []
    for d in days:
        out.append(
            {
                "date": (START + timedelta(days=d)).isoformat(),
                "serial_number": serial,
                "model": model,
                "capacity_bytes": 12_000_000_000_000,
                "failure": 1 if fail_day is not None and d == fail_day else 0,
                "smart_5_raw": (smart5(d) if smart5 else 0),
                "smart_9_raw": 1000 + 24 * d,
                "smart_194_raw": 30,
                "smart_197_raw": 0,
                "smart_198_raw": 0,
                "smart_187_raw": 0,
                "smart_188_raw": 0,
                "source_file": "f",
            }
        )
    return out


@pytest.fixture
def tiny_gold(spark):
    rows = []
    rows += _rows("A", range(0, 61), fail_day=60)  # fails on day 60
    rows += _rows("B", range(0, 41), fail_day=None)  # leaves the fleet on day 40, no failure
    rows += _rows("C", range(0, 101), fail_day=100)  # fails on day 100
    bronze = spark.createDataFrame(pd.DataFrame(rows)).withColumn("date", F.to_date("date"))
    sil = silver.clean(bronze)
    lc = silver.lifecycle(sil)
    g = gold.attach_labels(gold.build_features(sil, (7, 30)), lc, horizon=30)
    return g.toPandas().set_index(["serial_number", "day"]).sort_index()


def _day(d: int) -> int:
    return (START + timedelta(days=d) - date(1970, 1, 1)).days


def test_label_positive_inside_horizon(tiny_gold):
    assert tiny_gold.loc[("A", _day(30)), "label"] == 1  # failure at 60 is in (30, 60]
    assert tiny_gold.loc[("A", _day(59)), "label"] == 1
    assert tiny_gold.loc[("C", _day(70)), "label"] == 1


def test_label_negative_only_when_survival_is_known(tiny_gold):
    assert tiny_gold.loc[("A", _day(29)), "label"] == 0  # 60 > 29 + 30 and the drive is observed through day 59
    assert tiny_gold.loc[("C", _day(69)), "label"] == 0
    assert tiny_gold.loc[("B", _day(10)), "label"] == 0  # observed up to 40 >= 10 + 30


def test_censoring_when_drive_leaves_before_horizon(tiny_gold):
    row = tiny_gold.loc[("B", _day(11))]
    assert pd.isna(row["label"]) and row["censored"] == 1
    assert tiny_gold.loc[("B", _day(40)), "censored"] == 1
    assert tiny_gold.loc[("B", _day(10)), "censored"] == 0


def test_failure_row_is_excluded_not_labeled(tiny_gold):
    row = tiny_gold.loc[("A", _day(60))]
    assert row["is_failure_row"] == 1 and pd.isna(row["label"]) and row["censored"] == 0
    assert row["days_to_failure"] == 0


def test_days_to_failure(tiny_gold):
    assert tiny_gold.loc[("A", _day(45)), "days_to_failure"] == 15
    assert pd.isna(tiny_gold.loc[("B", _day(5)), "days_to_failure"])


def test_window_delta_uses_calendar_days(spark):
    # drive with a gap: days 0-9, then 20-30; smart_5 increases by 1 per observed day
    rows = _rows("G", list(range(0, 10)) + list(range(20, 31)), fail_day=None, smart5=lambda d: d)
    bronze = spark.createDataFrame(pd.DataFrame(rows)).withColumn("date", F.to_date("date"))
    sil = silver.clean(bronze)
    g = gold.build_features(sil, (7,)).toPandas().set_index("day").sort_index()
    # day 25: window covers days 18..25 -> only days 20..25 observed -> first value is 20 -> delta 5
    assert g.loc[_day(25), "smart_5_delta_7d"] == 5
    # day 20: nothing in the previous 7 calendar days (gap) -> delta 0
    assert g.loc[_day(20), "smart_5_delta_7d"] == 0
    assert g.loc[_day(20), "gap_since_prev_obs"] == 11


def test_features_do_not_look_into_the_future(spark):
    rows = _rows("T", range(0, 80), fail_day=79, smart5=lambda d: 0 if d < 50 else (d - 49) * 10)
    full = spark.createDataFrame(pd.DataFrame(rows)).withColumn("date", F.to_date("date"))
    truncated = full.filter(F.col("date") <= F.lit(START + timedelta(days=49)))
    f_full = gold.build_features(silver.clean(full), (7, 30)).toPandas().set_index("day").sort_index()
    f_trunc = gold.build_features(silver.clean(truncated), (7, 30)).toPandas().set_index("day").sort_index()
    cols = [c for c in f_trunc.columns if c not in ("date", "month", "serial_number", "model")]
    common = f_trunc.index
    pd.testing.assert_frame_equal(f_full.loc[common, cols], f_trunc.loc[common, cols], check_dtype=False)
