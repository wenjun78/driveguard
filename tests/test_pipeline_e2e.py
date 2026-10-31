"""The whole pipeline on a small synthetic fleet: bronze -> silver -> gold -> train -> backfill -> monitor."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date

import pytest

from driveguard import bronze, gold, monitor, score, silver, train
from driveguard.config import TABLES
from driveguard.spark import TableStore

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("raw")
    subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "..", "scripts", "make_fixture.py"),
            "--out",
            str(out),
            "--drives",
            "250",
            "--days",
            "150",
            "--annual-failure-rate",
            "0.12",
            "--seed",
            "3",
        ],
        check=True,
    )
    return str(out)


def test_end_to_end(spark, settings, fixture_dir):
    settings.raw_path = fixture_dir
    settings.test_days, settings.val_days, settings.gap_days = 40, 30, 10
    settings.xgb_rounds = 120
    settings.experiment = "/Shared/driveguard-test"
    store = TableStore(spark, settings)

    assert bronze.run(spark, settings) == 150
    sil = silver.run(spark, settings)
    assert sil["drives"] > 200 and sil["failed_drives"] > 0
    g = gold.run(spark, settings)
    assert g["positives"] > 0 and g["censored"] > 0 and g["failure_rows"] == sil["failed_drives"]

    res = train.run(spark, settings)
    rep = res["report"]
    assert "val" in rep and "test" in rep and "baseline_test" in rep
    assert 0 <= rep["test"]["recall_at_budget"] <= 1
    assert rep["val"]["pr_auc"] > rep["val"]["positives"] / rep["val"]["rows_labeled"]  # better than random
    assert res["model_uri"]

    sc = score.run(spark, settings, model_uri=res["model_uri"])
    assert sc["alerts"] >= 1 and sc["drives_scored"] > 100
    alerts = store.read(TABLES["alerts"]).toPandas()
    assert alerts.flagged.sum() == sc["alerts"]
    assert alerts[alerts.flagged == 1].reason_codes.map(len).gt(0).all()
    assert (alerts.rank_in_day >= 1).all()

    bf = score.backfill(spark, settings, days=40, model_uri=res["model_uri"])
    assert bf["dates_scored"] == 40
    alerts = store.read(TABLES["alerts"])
    assert alerts.select("date").distinct().count() == 40  # the earlier single-day run was replaced, not duplicated

    mon = monitor.run(spark, settings, model_uri=res["model_uri"])
    assert mon["matured_snapshots"] >= 5
    assert "retrain_recommended" in mon and isinstance(mon["retrain_recommended"], bool)
    assert isinstance(mon["psi_by_feature"], dict) and len(mon["psi_by_feature"]) > 5
    assert store.read(TABLES["monitor"]).count() == 1
    assert date.fromisoformat(mon["snapshot_date"])
