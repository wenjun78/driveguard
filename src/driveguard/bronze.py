"""Bronze: land Backblaze daily SMART files as one typed table, idempotently.

Backblaze publishes one CSV per day (``YYYY-MM-DD.csv``) inside quarterly zips. The column set drifts over the
years - attributes get added, later files carry datacenter/pod columns - so files are grouped by header
signature, read per group, and unioned by name. Everything is kept; typing is the only transformation.

Idempotency: every ingested file is recorded in ``ops_ingest_log``; re-running skips files already landed, so a
scheduled job can point at the same Volume as quarters are added.
"""

from __future__ import annotations

import glob
import os
import re
from datetime import datetime, timezone

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from driveguard.config import TABLES, Settings
from driveguard.spark import TableStore

BASE_COLUMNS = ("date", "serial_number", "model", "capacity_bytes", "failure")
CONTEXT_COLUMNS = ("datacenter", "cluster_id", "vault_id", "pod_id", "pod_slot_num", "is_legacy_format")
_SMART = re.compile(r"^smart_(\d+)_(raw|normalized)$")


def discover_files(raw_path: str) -> dict[str, list[str]]:
    """Return {"csv": [...], "parquet": [...]} found under ``raw_path`` (recursively), sorted."""
    csvs = sorted(glob.glob(os.path.join(raw_path, "**", "*.csv"), recursive=True))
    parquet = sorted(glob.glob(os.path.join(raw_path, "**", "*.parquet"), recursive=True))
    return {"csv": csvs, "parquet": parquet}


def _header(path: str) -> tuple[str, ...]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        line = fh.readline().strip().lstrip("﻿")
    return tuple(c.strip().strip('"') for c in line.split(","))


def group_by_header(paths: list[str]) -> dict[tuple[str, ...], list[str]]:
    groups: dict[tuple[str, ...], list[str]] = {}
    for p in paths:
        groups.setdefault(_header(p), []).append(p)
    return groups


def _type_columns(df: DataFrame) -> DataFrame:
    """Cast known columns; keep unknown ones as strings."""
    exprs = []
    for c in df.columns:
        if c == "date":
            exprs.append(F.to_date(F.col("date")).alias("date"))
        elif c in ("capacity_bytes",):
            exprs.append(F.col(c).cast("double").cast("long").alias(c))
        elif c in ("failure", "pod_slot_num", "is_legacy_format"):
            exprs.append(F.col(c).cast("double").cast("int").alias(c))
        elif _SMART.match(c):
            exprs.append(F.col(c).cast("double").cast("long").alias(c))
        else:
            exprs.append(F.col(c).cast("string").alias(c))
    return df.select(*exprs)


def read_csv_group(spark: SparkSession, paths: list[str]) -> DataFrame:
    df = spark.read.option("header", True).option("inferSchema", False).csv(paths)
    df = df.toDF(*[c.strip().lstrip("﻿") for c in df.columns])
    return _type_columns(df).withColumn("source_file", F.input_file_name())


def read_parquet_subset(spark: SparkSession, paths: list[str]) -> DataFrame:
    df = spark.read.option("mergeSchema", True).parquet(*paths)
    if "source_file" not in df.columns:
        df = df.withColumn("source_file", F.input_file_name())
    return _type_columns(df)


def _union_all(frames: list[DataFrame]) -> DataFrame:
    out = frames[0]
    for f in frames[1:]:
        out = out.unionByName(f, allowMissingColumns=True)
    return out


def already_ingested(store: TableStore) -> set[str]:
    if not store.exists(TABLES["ingest_log"]):
        return set()
    return {r[0] for r in store.read(TABLES["ingest_log"]).select("source_file").distinct().collect()}


def run(spark: SparkSession, settings: Settings, raw_path: str | None = None) -> int:
    """Land new raw files into bronze. Returns the number of files ingested."""
    store = TableStore(spark, settings)
    store.ensure_schema()
    raw_path = raw_path or settings.raw_path
    found = discover_files(raw_path)
    done = already_ingested(store)
    # Spark reports file names as URIs (file:///...); normalise both sides to absolute paths for the comparison.
    done_abs = {os.path.abspath(p.replace("file:", "")) for p in done}
    csv_new = [p for p in found["csv"] if os.path.abspath(p) not in done_abs]
    pq_new = [p for p in found["parquet"] if os.path.abspath(p) not in done_abs]
    if not csv_new and not pq_new:
        return 0

    frames: list[DataFrame] = []
    for _, paths in group_by_header(csv_new).items():
        frames.append(read_csv_group(spark, paths))
    if pq_new:
        frames.append(read_parquet_subset(spark, pq_new))
    bronze = _union_all(frames)
    for c in BASE_COLUMNS:
        if c not in bronze.columns:
            raise ValueError(f"raw files lack required column {c!r}; found {bronze.columns[:10]}...")

    bronze = (
        bronze.withColumn("ingest_ts", F.lit(datetime.now(timezone.utc).isoformat()))
        .withColumn("month", F.date_format("date", "yyyy-MM"))
        .withColumn("source_file", F.regexp_replace("source_file", "^file:/*", "/"))
    )
    mode = "append" if store.exists(TABLES["bronze"]) else "overwrite"
    store.write(bronze, TABLES["bronze"], mode=mode, partition_by=["month"])

    log = spark.createDataFrame(
        [(os.path.abspath(p), datetime.now(timezone.utc).isoformat()) for p in csv_new + pq_new],
        T.StructType([T.StructField("source_file", T.StringType()), T.StructField("ingest_ts", T.StringType())]),
    )
    store.write(log, TABLES["ingest_log"], mode="append" if store.exists(TABLES["ingest_log"]) else "overwrite")
    return len(csv_new) + len(pq_new)
