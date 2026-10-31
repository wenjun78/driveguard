"""Monitoring: input drift, score drift, matured-label performance, and a retraining decision.

Labels for a snapshot mature ``horizon_days`` later: only then do we know which flagged drives actually failed.
So each monitoring run does two things - checks *today's* inputs and scores against the training reference (PSI),
and back-fills *real* precision/recall for every past snapshot whose horizon has now elapsed. The retraining flag
combines both: material drift on several key features, or matured recall at the alert budget falling below the
floor. The flag is written to ``ops_monitoring`` and exposed as a job task value so Databricks Workflows can run
the training task conditionally.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from driveguard import metrics as M
from driveguard import replay
from driveguard.config import KEY_SMART_RAW, TABLES, Settings
from driveguard.model import DriveGuardModel
from driveguard.score import resolve_model_uri
from driveguard.spark import TableStore
from driveguard.train import setup_mlflow

KEY_FEATURES = (
    [f"smart_{n}_raw" for n in KEY_SMART_RAW]
    + [f"smart_{n}_delta_30d" for n in KEY_SMART_RAW]
    + ["smart_9_raw", "smart_194_raw"]
)


def load_reference(model_uri: str) -> tuple[DriveGuardModel, pd.DataFrame]:
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri}/artifacts/model_dir")
    model = DriveGuardModel.load(local)
    ref_path = os.path.join(local, "reference_sample.parquet")
    ref = pd.read_parquet(ref_path) if os.path.exists(ref_path) else pd.DataFrame()
    return model, ref


def feature_psi(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str]) -> dict[str, float]:
    out = {}
    for f in features:
        if f in ref.columns and f in cur.columns:
            out[f] = M.psi(
                pd.to_numeric(ref[f], errors="coerce").to_numpy(float),
                pd.to_numeric(cur[f], errors="coerce").to_numpy(float),
            )
    return out


def matured_performance(spark: SparkSession, store: TableStore, s: Settings, now: date) -> list[dict[str, Any]]:
    """Real outcomes for every scored snapshot whose horizon has elapsed by ``now``."""
    if not store.exists(TABLES["alerts"]):
        return []
    gold = store.read(TABLES["gold"])
    end_day = (now - date(1970, 1, 1)).days
    alerts = store.read(TABLES["alerts"]).select("serial_number", "date", "day", "score", "flagged")
    matured = alerts.filter(F.col("day") + F.lit(s.horizon_days) <= F.lit(end_day))
    if matured.limit(1).count() == 0:
        return []
    labels = gold.select("serial_number", "date", "label", "days_to_failure", "is_failure_row")
    joined = matured.join(labels, ["serial_number", "date"], "left").filter(
        "is_failure_row = 0 OR is_failure_row IS NULL"
    )
    pdf = joined.select("serial_number", "date", "day", "label", "days_to_failure", "score", "flagged").toPandas()
    out = []
    for d, part in pdf.groupby("date"):
        m = M.evaluate(part, "score", s.alert_budget, s.horizon_days, flag_col="flagged")
        m["snapshot_date"] = str(d)
        out.append(m)
    return sorted(out, key=lambda r: r["snapshot_date"])


def run(spark: SparkSession, s: Settings, as_of: str | None = None, model_uri: str | None = None) -> dict[str, Any]:
    setup_mlflow(s)
    store = TableStore(spark, s)
    gold = store.read(TABLES["gold"])
    snapshot_day: date = replay.resolve_as_of(as_of, gold.agg(F.max("date")).first()[0])
    cur_df = gold.filter(F.col("date") == F.lit(snapshot_day)).filter("is_failure_row = 0")

    uri = resolve_model_uri(s, model_uri)
    model, ref = load_reference(uri)
    cur = cur_df.select(*[c for c in KEY_FEATURES if c in cur_df.columns], "model").toPandas()
    if len(cur) > 50000:
        cur = cur.sample(n=50000, random_state=s.seed)

    psi_by_feature = feature_psi(ref, cur, KEY_FEATURES) if len(ref) else {}
    drifted = sorted([f for f, v in psi_by_feature.items() if not np.isnan(v) and v >= s.psi_alert])
    warned = sorted([f for f, v in psi_by_feature.items() if not np.isnan(v) and s.psi_warn <= v < s.psi_alert])

    score_psi = float("nan")
    if len(ref) and set(model.spec.numeric).issubset(ref.columns):
        ref_scores = model.predict_proba(ref)
        present = [c for c in model.spec.numeric if c in cur_df.columns]
        cur_full = cur_df.select(*present, "model").toPandas()
        cur_scores = model.predict_proba(cur_full)
        score_psi = M.psi(ref_scores.astype(float), cur_scores.astype(float))

    matured = matured_performance(spark, store, s, now=snapshot_day)
    recent = [r for r in matured if "recall_at_budget" in r][-7:]
    recent_recall = float(np.mean([r["recall_at_budget"] for r in recent])) if recent else float("nan")

    reasons = []
    if len(drifted) >= s.psi_features_for_retrain:
        reasons.append(f"{len(drifted)} key features drifted (PSI >= {s.psi_alert}): {', '.join(drifted)}")
    if recent and recent_recall < s.matured_recall_floor:
        reasons.append(
            f"matured recall@budget over last {len(recent)} snapshots = {recent_recall:.3f} < {s.matured_recall_floor}"
        )
    retrain = bool(reasons)

    row = {
        "monitored_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_date": snapshot_day.isoformat(),
        "model_uri": uri,
        "drives_scored": int(cur_df.count()),
        "psi_by_feature": json.dumps({k: (None if np.isnan(v) else round(v, 5)) for k, v in psi_by_feature.items()}),
        "features_drifted": json.dumps(drifted),
        "features_warned": json.dumps(warned),
        "score_psi": None if np.isnan(score_psi) else float(score_psi),
        "matured_snapshots": len(matured),
        "matured_recent_recall_at_budget": None if np.isnan(recent_recall) else recent_recall,
        "matured_latest": json.dumps(matured[-1], default=float) if matured else None,
        "matured_all": json.dumps(matured, default=float),
        "retrain_recommended": retrain,
        "retrain_reasons": json.dumps(reasons),
    }
    schema = T.StructType(
        [
            T.StructField("monitored_at", T.StringType()),
            T.StructField("snapshot_date", T.StringType()),
            T.StructField("model_uri", T.StringType()),
            T.StructField("drives_scored", T.IntegerType()),
            T.StructField("psi_by_feature", T.StringType()),
            T.StructField("features_drifted", T.StringType()),
            T.StructField("features_warned", T.StringType()),
            T.StructField("score_psi", T.DoubleType()),
            T.StructField("matured_snapshots", T.IntegerType()),
            T.StructField("matured_recent_recall_at_budget", T.DoubleType()),
            T.StructField("matured_latest", T.StringType()),
            T.StructField("matured_all", T.StringType()),
            T.StructField("retrain_recommended", T.BooleanType()),
            T.StructField("retrain_reasons", T.StringType()),
        ]
    )
    store.write(
        spark.createDataFrame([row], schema),
        TABLES["monitor"],
        mode="append" if store.exists(TABLES["monitor"]) else "overwrite",
    )
    return {
        **{k: v for k, v in row.items() if k != "matured_all"},
        "psi_by_feature": psi_by_feature,
        "matured": matured[-3:],
    }
