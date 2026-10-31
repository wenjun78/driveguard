"""Split protocol invariants and alert-budget metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driveguard import metrics as M
from driveguard import splits
from driveguard.config import Settings


def test_time_split_has_label_disjoint_gaps(spark):
    s = Settings(horizon_days=30, val_days=40, test_days=60, gap_days=30)
    days = list(range(19000, 19400))
    df = spark.createDataFrame(pd.DataFrame({"day": days, "model": "M"}))
    out, b = splits.assign(df, s)
    pdf = out.toPandas()
    assert b.end_day == 19399 and b.test_start == 19340
    assert b.val_end == 19309 and b.val_start == 19270 and b.train_end == 19239
    assert pdf[pdf.split == "train"].day.max() == b.train_end
    assert pdf[pdf.split == "val"].day.min() == b.val_start and pdf[pdf.split == "val"].day.max() == b.val_end
    assert pdf[pdf.split == "test"].day.min() == b.test_start
    # the gaps are exactly horizon-long so training labels never depend on validation-period outcomes
    assert b.val_start - b.train_end - 1 == s.gap_days and b.test_start - b.val_end - 1 == s.gap_days
    assert (pdf.split == "gap").sum() == 2 * s.gap_days


def test_split_refuses_short_history(spark):
    s = Settings(horizon_days=30, val_days=60, test_days=90, gap_days=30)
    df = spark.createDataFrame(pd.DataFrame({"day": list(range(100, 300)), "model": "M"}))
    with pytest.raises(ValueError):
        splits.assign(df, s)


def test_group_holdout_routes_unseen_models(spark):
    s = Settings(horizon_days=10, val_days=20, test_days=30, gap_days=10, group_holdout_frac=0.5)
    pdf = pd.DataFrame({"day": list(range(0, 200)) * 4, "model": ["A"] * 200 + ["B"] * 200 + ["C"] * 200 + ["D"] * 200})
    out = splits.assign(spark.createDataFrame(pdf), s)[0].toPandas()
    held = set(out[out.holdout_model == 1].model)
    assert 0 < len(held) < 4
    assert set(out[out.model.isin(held)].split) <= {"test_unseen", "gap"}
    assert "test" in set(out[~out.model.isin(held)].split)


def _frame():
    # 2 days x 10 drives; drive d9 fails (label 1 on both days), everyone else healthy
    rows = []
    for day in (1, 2):
        for i in range(10):
            rows.append(
                {
                    "serial_number": f"d{i}",
                    "day": day,
                    "label": 1 if i == 9 else 0,
                    "days_to_failure": (12 - day) if i == 9 else np.nan,
                    "score": i / 10,
                }
            )
    return pd.DataFrame(rows)


def test_flag_at_budget_top_k_per_day():
    df = _frame()
    flags = M.flag_at_budget(df, "score", budget=0.2)
    assert flags.sum() == 4  # ceil(0.2 * 10) = 2 per day
    assert set(df[flags].serial_number) == {"d8", "d9"}


def test_evaluate_metrics_at_budget():
    df = _frame()
    m = M.evaluate(df, "score", budget=0.1, horizon=30)  # 1 alert per day -> d9 both days
    assert m["recall_at_budget"] == 1.0 and m["precision_at_budget"] == 1.0
    assert m["drive_recall_at_budget"] == 1.0 and m["failing_drives"] == 1
    assert m["lead_time_median_days"] == 11  # first alert on day 1, failure on day 12
    assert m["false_alerts_per_1000_drive_days"] == 0
    assert m["pr_auc"] == 1.0
    # inverted scores -> the failing drive is never flagged
    df["score"] = 1 - df["score"]
    m = M.evaluate(df, "score", budget=0.1, horizon=30)
    assert m["recall_at_budget"] == 0.0 and m["drive_recall_at_budget"] == 0.0
    assert m["false_alerts_per_1000_drive_days"] > 0


def test_evaluate_ignores_unlabeled_rows_but_counts_them_in_budget():
    df = _frame()
    df.loc[df.serial_number == "d0", "label"] = np.nan  # censored
    m = M.evaluate(df, "score", budget=0.1, horizon=30)
    assert m["rows_scored"] == 20 and m["rows_labeled"] == 18


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert M.psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert M.psi(a, rng.normal(1.5, 1, 5000)) > 0.5
    zeros = np.zeros(5000)
    assert M.psi(zeros, zeros) < 1e-6
    mostly_zero_then_more = np.concatenate([np.zeros(4500), np.ones(500)])
    assert M.psi(zeros, mostly_zero_then_more) > 0.1
