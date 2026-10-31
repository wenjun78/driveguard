"""Distributed scoring and per-day alert ranking on Spark."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from driveguard.model import DriveGuardModel


def score_spark(df: DataFrame, model: DriveGuardModel, keep: Iterable[str]) -> DataFrame:
    """Add a ``score`` column (failure probability within the horizon) using mapInPandas."""
    keep = list(keep)
    # the model travels inside the function closure (a few hundred KB): no SparkContext / broadcast needed,
    # which keeps this working on serverless compute and Spark Connect
    payload = model.to_bytes()
    needed = list(dict.fromkeys(keep + model.spec.numeric + ["model"]))
    needed = [c for c in needed if c in df.columns]
    schema = T.StructType([f for f in df.schema.fields if f.name in keep] + [T.StructField("score", T.DoubleType())])

    def fn(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        m = None
        for pdf in batches:
            if m is None:
                m = DriveGuardModel.from_bytes(*payload)
            out = pdf[keep].copy()
            out["score"] = m.predict_proba(pdf).astype("float64") if len(pdf) else pd.Series(dtype="float64")
            yield out

    return df.select(*needed).mapInPandas(fn, schema=schema)


def flag_at_budget(scored: DataFrame, budget: float, score_col: str = "score") -> DataFrame:
    """Per day, flag the top ``ceil(budget * drives scored that day)`` drives (at least one)."""
    by_day = Window.partitionBy("day")
    ranked = Window.partitionBy("day").orderBy(F.col(score_col).desc(), F.col("serial_number"))
    k = F.greatest(F.lit(1), F.ceil(F.lit(budget) * F.count("*").over(by_day)))
    return scored.withColumn("rank_in_day", F.row_number().over(ranked)).withColumn(
        "flagged", (F.col("rank_in_day") <= k).cast("int")
    )


def rule_baseline_score(df: DataFrame) -> DataFrame:
    """The operator rule of thumb: any of the five key SMART counters non-zero, ranked by their sum.

    It is what a storage team can do without a model, so it is the bar the model has to clear.
    """
    total = sum(F.coalesce(F.col(f"smart_{n}_raw"), F.lit(0)) for n in (5, 187, 188, 197, 198))
    return df.withColumn("score", (F.col("any_key_smart_nonzero") * 1e6 + F.log1p(total)).cast("double"))
