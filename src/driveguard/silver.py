"""Silver: clean, typed, deduplicated drive-days plus a per-drive lifecycle table.

Rules (each one is a known quirk of the Backblaze files):
* one row per (serial_number, date) - duplicates keep the row that records a failure, else the last one landed;
* ``failure`` is 0/1, never null;
* ``capacity_bytes <= 0`` (Backblaze uses -1 when unknown) becomes the drive's maximum reported capacity;
* a handful of serials report rows *after* their failure day (re-inserted drives) - those rows are dropped, so a
  failure is always the drive's final observation and labels downstream are unambiguous;
* ``day`` is the date as an integer day number, which lets feature windows use range frames over calendar days
  rather than row counts (drives have gaps).

The lifecycle table gives first/last observation and failure date per drive; it is what the label logic and the
right-censoring rule read.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from driveguard.config import (
    CONTEXT_SMART_RAW,
    KEY_SMART_NORMALIZED,
    KEY_SMART_RAW,
    TABLES,
    Settings,
)
from driveguard.spark import TableStore


def smart_columns() -> list[str]:
    cols = [f"smart_{n}_raw" for n in KEY_SMART_RAW + CONTEXT_SMART_RAW]
    cols += [f"smart_{n}_normalized" for n in KEY_SMART_NORMALIZED]
    return cols


def clean(bronze: DataFrame) -> DataFrame:
    wanted = ["date", "serial_number", "model", "capacity_bytes", "failure", "source_file"] + smart_columns()
    df = bronze
    for c in wanted:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("long" if c.startswith("smart_") else "string"))
    df = df.select(*wanted)
    df = (
        df.filter(F.col("date").isNotNull() & F.col("serial_number").isNotNull() & (F.trim("serial_number") != ""))
        .withColumn("serial_number", F.trim("serial_number"))
        .withColumn("model", F.trim(F.coalesce(F.col("model"), F.lit("UNKNOWN"))))
        .withColumn("failure", F.coalesce(F.col("failure").cast("int"), F.lit(0)))
        .withColumn("failure", F.when(F.col("failure") > 0, 1).otherwise(0))
    )
    # one row per drive-day
    w = Window.partitionBy("serial_number", "date").orderBy(F.col("failure").desc(), F.col("source_file").desc())
    df = df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn", "source_file")

    # capacity: unknown -> drive's max known capacity
    df = df.withColumn("capacity_bytes", F.when(F.col("capacity_bytes") <= 0, None).otherwise(F.col("capacity_bytes")))
    df = df.withColumn("capacity_bytes", F.max("capacity_bytes").over(Window.partitionBy("serial_number")))

    # drop observations after a drive's (first) failure day
    first_fail = F.min(F.when(F.col("failure") == 1, F.col("date"))).over(Window.partitionBy("serial_number"))
    df = df.withColumn("_fail_date", first_fail).filter(
        F.col("_fail_date").isNull() | (F.col("date") <= F.col("_fail_date"))
    )
    df = df.drop("_fail_date")

    return df.withColumn("day", F.datediff(F.col("date"), F.lit("1970-01-01")).cast("int")).withColumn(
        "month", F.date_format("date", "yyyy-MM")
    )


def lifecycle(silver: DataFrame) -> DataFrame:
    w_last = Window.partitionBy("serial_number").orderBy(F.col("date").desc())
    last_model = F.first("model").over(w_last)
    return (
        silver.withColumn("_last_model", last_model)
        .groupBy("serial_number")
        .agg(
            F.first("_last_model").alias("model"),
            F.max("capacity_bytes").alias("capacity_bytes"),
            F.min("date").alias("first_seen"),
            F.max("date").alias("last_seen"),
            F.min("day").alias("first_day"),
            F.max("day").alias("last_day"),
            F.count("*").alias("n_days_observed"),
            F.max("failure").alias("failed"),
            F.min(F.when(F.col("failure") == 1, F.col("date"))).alias("failure_date"),
            F.min(F.when(F.col("failure") == 1, F.col("day"))).alias("failure_day"),
        )
    )


def run(spark: SparkSession, settings: Settings) -> dict[str, int]:
    store = TableStore(spark, settings)
    bronze = store.read(TABLES["bronze"])
    silver = clean(bronze)
    store.write(silver, TABLES["silver"], mode="overwrite", partition_by=["month"])
    silver = store.read(TABLES["silver"])
    lc = lifecycle(silver)
    store.write(lc, TABLES["lifecycle"], mode="overwrite")
    return {"silver_rows": silver.count(), "drives": lc.count(), "failed_drives": lc.filter("failed = 1").count()}
