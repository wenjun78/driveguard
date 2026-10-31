"""Gold: one row per drive-day with time-safe features and a right-censoring-aware 30-day failure label.

Time safety. Every feature for drive *d* on day *t* is a function of rows of *d* with ``day <= t`` only. Window
features use *range* frames over the integer day number, so a window of 30 means "the last 30 calendar days",
not "the last 30 rows" - drives have gaps and the two differ.

Label. ``label = 1`` when the drive's recorded failure day lies in ``(t, t + horizon]``. ``label = 0`` only when we
*know* the drive survived the whole horizon: it has an observation on or after ``t + horizon`` (or it failed later
than that). Rows where neither is known - the drive left the fleet or the data ends before ``t + horizon`` without a
failure - are **censored** and get ``label = NULL``; they are kept in the table (they still get scored in production)
but excluded from training and evaluation. The failure-day row itself is flagged ``is_failure_row`` and also excluded:
it is the outcome, not a decision point.
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

ID_COLUMNS = ("serial_number", "date", "day", "month")
LABEL_COLUMNS = ("label", "is_failure_row", "censored", "days_to_failure", "failure")
CATEGORICAL_FEATURES = ("model",)

# attributes that get change-over-window features
DELTA_ATTRS = KEY_SMART_RAW + (9, 12, 192, 193, 199, 240, 241, 242)
NONZERO_DAYS_ATTRS = KEY_SMART_RAW


def feature_columns(df: DataFrame) -> list[str]:
    skip = set(ID_COLUMNS) | set(LABEL_COLUMNS)
    return [c for c in df.columns if c not in skip]


def build_features(silver: DataFrame, windows: tuple[int, ...]) -> DataFrame:
    by_drive = Window.partitionBy("serial_number").orderBy("day")
    df = silver
    cols = [
        "serial_number",
        "date",
        "day",
        "month",
        "model",
        "failure",
        (F.col("capacity_bytes") / 1e12).alias("capacity_tb"),
    ]
    cols += [F.col(f"smart_{n}_raw") for n in KEY_SMART_RAW + CONTEXT_SMART_RAW]
    cols += [F.col(f"smart_{n}_normalized") for n in KEY_SMART_NORMALIZED]
    # observation pattern. Deliberately *no* "days since first seen": it grows with calendar time, so under a
    # time-ordered split every test row would carry values the model never saw in training. Power-on hours
    # (smart_9_raw) is the age signal, and it is a property of the drive, not of the dataset window.
    cols += [(F.col("day") - F.lag("day", 1).over(by_drive)).alias("gap_since_prev_obs")]
    df = df.select(*cols)

    for w in windows:
        frame = Window.partitionBy("serial_number").orderBy("day").rangeBetween(-w, 0)
        new = []
        for n in DELTA_ATTRS:
            c = F.col(f"smart_{n}_raw")
            new.append((c - F.first(c, ignorenulls=True).over(frame)).alias(f"smart_{n}_delta_{w}d"))
        for n in NONZERO_DAYS_ATTRS:
            c = F.col(f"smart_{n}_raw")
            new.append(F.sum(F.when(c > 0, 1).otherwise(0)).over(frame).alias(f"smart_{n}_nzdays_{w}d"))
        new.append(F.avg("smart_194_raw").over(frame).alias(f"smart_194_mean_{w}d"))
        new.append(F.max("smart_194_raw").over(frame).alias(f"smart_194_max_{w}d"))
        new.append(F.max("smart_5_raw").over(frame).alias(f"smart_5_max_{w}d"))
        new.append(F.max("smart_197_raw").over(frame).alias(f"smart_197_max_{w}d"))
        df = df.select("*", *new)

    # any of the five key counters non-zero right now (a simple, explainable risk flag used as a baseline too)
    any_key = None
    for n in KEY_SMART_RAW:
        cond = F.coalesce(F.col(f"smart_{n}_raw"), F.lit(0)) > 0
        any_key = cond if any_key is None else (any_key | cond)
    return df.withColumn("any_key_smart_nonzero", any_key.cast("int"))


def attach_labels(features: DataFrame, lifecycle: DataFrame, horizon: int) -> DataFrame:
    lc = lifecycle.select("serial_number", "last_day", "failure_day")
    df = features.join(lc, "serial_number", "left")
    horizon_end = F.col("day") + F.lit(horizon)
    fails_in_horizon = (
        F.col("failure_day").isNotNull() & (F.col("failure_day") > F.col("day")) & (F.col("failure_day") <= horizon_end)
    )
    survived_known = (F.col("failure_day").isNull() | (F.col("failure_day") > horizon_end)) & (
        F.col("last_day") >= horizon_end
    )
    df = (
        df.withColumn("is_failure_row", (F.col("failure") == 1).cast("int"))
        .withColumn(
            "label",
            F.when(F.col("is_failure_row") == 1, None)
            .when(fails_in_horizon, 1)
            .when(survived_known, 0)
            .otherwise(None),
        )
        .withColumn("censored", (F.col("label").isNull() & (F.col("is_failure_row") == 0)).cast("int"))
        .withColumn("days_to_failure", F.col("failure_day") - F.col("day"))
    )
    return df.drop("last_day", "failure_day")


def run(spark: SparkSession, settings: Settings) -> dict[str, int]:
    store = TableStore(spark, settings)
    silver = store.read(TABLES["silver"])
    lc = store.read(TABLES["lifecycle"])
    gold = attach_labels(build_features(silver, settings.feature_windows), lc, settings.horizon_days)
    store.write(gold, TABLES["gold"], mode="overwrite", partition_by=["month"])
    gold = store.read(TABLES["gold"])
    agg = gold.agg(
        F.count("*").alias("rows"),
        F.sum("label").alias("positives"),
        F.sum(F.when(F.col("label") == 0, 1).otherwise(0)).alias("negatives"),
        F.sum("censored").alias("censored"),
        F.sum("is_failure_row").alias("failure_rows"),
    ).first()
    return {k: int(agg[k] or 0) for k in ("rows", "positives", "negatives", "censored", "failure_rows")}
