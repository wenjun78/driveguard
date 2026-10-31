"""Batch scoring: rank the fleet for one or more snapshot days and write the alert list with reason codes."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

import mlflow
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from driveguard import scoring
from driveguard.config import TABLES, Settings
from driveguard.model import DriveGuardModel
from driveguard.spark import TableStore
from driveguard.train import setup_mlflow

log = logging.getLogger(__name__)
KEEP = ["serial_number", "date", "day", "model", "capacity_tb"]


def resolve_model_uri(s: Settings, model_uri: str | None = None) -> str:
    """Champion alias first; otherwise the most recent training run (so a fresh workspace still scores)."""
    if model_uri:
        return model_uri
    name = s.registered_model_name()
    try:
        from mlflow.tracking import MlflowClient

        mv = MlflowClient().get_model_version_by_alias(name, s.champion_alias)
        return f"models:/{name}@{s.champion_alias}" if s.catalog else f"models:/{name}/{mv.version}"
    except Exception:
        pass
    exp = mlflow.get_experiment_by_name(s.experiment if s.on_databricks else s.experiment.strip("/").replace("/", "_"))
    if exp is None:
        raise RuntimeError("no registered champion and no training runs found")
    runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        raise RuntimeError("no training runs found")
    return f"runs:/{runs.iloc[0]['run_id']}/model"


def load_model(model_uri: str) -> DriveGuardModel:
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri}/artifacts/model_dir")
    return DriveGuardModel.load(local)


def _write_for_dates(store: TableStore, df: DataFrame, name: str, dates: list[date]) -> None:
    """Replace the rows of the given snapshot dates (idempotent re-runs)."""
    if not store.exists(name):
        store.write(df, name, mode="overwrite", partition_by=["date"])
        return
    if store.s.table_format == "delta":
        target = store.s.table(name)
        where = "date IN (" + ", ".join(f"'{d.isoformat()}'" for d in dates) + ")"
        writer = df.write.format("delta").mode("overwrite").option("replaceWhere", where)
        writer.saveAsTable(target) if store.s.catalog else writer.save(target)
        return
    existing = store.read(name).filter(~F.col("date").isin(dates))
    merged = existing.unionByName(df, allowMissingColumns=True)
    merged.cache().count()
    store.write(merged, name, mode="overwrite", partition_by=["date"])
    merged.unpersist()


def score_dates(
    spark: SparkSession,
    store: TableStore,
    s: Settings,
    gold: DataFrame,
    dates: list[date],
    uri: str,
    model: DriveGuardModel,
) -> dict[str, Any]:
    snap = gold.filter(F.col("date").isin(dates)).filter("is_failure_row = 0")
    n = snap.count()
    if n == 0:
        raise ValueError(f"no drive-days for {dates}")
    scored = scoring.flag_at_budget(scoring.score_spark(snap, model, KEEP), s.alert_budget)

    # reason codes for the flagged rows only (a budget's worth per day, so small)
    flagged = scored.filter("flagged = 1").select("serial_number", "date")
    feats = snap.join(flagged, ["serial_number", "date"], "inner").toPandas()
    reasons = model.reason_codes(feats, top=3) if len(feats) else []
    reason_df = spark.createDataFrame(
        [(sn, d, r) for sn, d, r in zip(feats["serial_number"], feats["date"], reasons, strict=True)],
        T.StructType(
            [
                T.StructField("serial_number", T.StringType()),
                T.StructField("date", T.DateType()),
                T.StructField("reason_codes", T.ArrayType(T.StringType())),
            ]
        ),
    )
    out = (
        scored.join(reason_df, ["serial_number", "date"], "left")
        .withColumn("model_uri", F.lit(uri))
        .withColumn("scored_at", F.lit(datetime.now(timezone.utc).isoformat()))
        .withColumn("horizon_days", F.lit(s.horizon_days))
        .withColumn("alert_budget", F.lit(s.alert_budget))
    )
    _write_for_dates(store, out, TABLES["alerts"], dates)
    last = max(dates)
    written = store.read(TABLES["alerts"]).filter(F.col("date").isin(dates))
    top = (
        written.filter((F.col("flagged") == 1) & (F.col("date") == F.lit(last)))
        .orderBy("rank_in_day")
        .select("rank_in_day", "serial_number", "model", F.round("score", 4).alias("score"), "reason_codes")
        .limit(10)
        .collect()
    )
    per_day = {
        str(r["date"]): {"drives_scored": int(r["n"]), "alerts": int(r["alerts"])}
        for r in written.groupBy("date").agg(F.count("*").alias("n"), F.sum("flagged").alias("alerts")).collect()
    }
    return {
        "snapshot_date": last.isoformat(),
        "dates_scored": len(dates),
        "drives_scored": per_day[last.isoformat()]["drives_scored"],
        "alerts": per_day[last.isoformat()]["alerts"],
        "model_uri": uri,
        "top_alerts": [r.asDict() for r in top],
        "per_day": per_day,
    }


def run(spark: SparkSession, s: Settings, as_of: str | None = None, model_uri: str | None = None) -> dict[str, Any]:
    setup_mlflow(s)
    store = TableStore(spark, s)
    gold = store.read(TABLES["gold"])
    snapshot_day: date = date.fromisoformat(as_of) if as_of else gold.agg(F.max("date")).first()[0]
    uri = resolve_model_uri(s, model_uri)
    model = load_model(uri)
    out = score_dates(spark, store, s, gold, [snapshot_day], uri, model)
    out.pop("per_day")
    return out


def backfill(spark: SparkSession, s: Settings, days: int, model_uri: str | None = None) -> dict[str, Any]:
    """Replay daily scoring over the last ``days`` snapshot dates - how matured-label monitoring gets its history."""
    setup_mlflow(s)
    store = TableStore(spark, s)
    gold = store.read(TABLES["gold"])
    dates = sorted(r[0] for r in gold.select("date").distinct().orderBy(F.col("date").desc()).limit(days).collect())
    uri = resolve_model_uri(s, model_uri)
    model = load_model(uri)
    out = score_dates(spark, store, s, gold, dates, uri, model)
    out.pop("top_alerts")
    return out
