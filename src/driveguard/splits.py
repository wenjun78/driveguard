"""Evaluation protocol: strictly time-ordered splits with unlabeled gaps, plus an optional unseen-population holdout.

::

    |------------- train -------------|  gap  |---- val ----|  gap  |-------- test --------|
                                     T1-gap  T1          T2-gap  T2                      end

* ``gap = horizon_days``: a training row at day *t* carries a label that depends on outcomes up to ``t + horizon``.
  Without the gap those outcomes fall inside the validation period, so the model would be fitted on the very
  failures it is later evaluated on. The gap makes the periods label-disjoint, not just feature-disjoint.
* The last ``horizon`` days of the test period are censored by construction (no future to look at), which is why
  ``test_days`` should be comfortably larger than ``horizon_days``.
* ``group_holdout_frac`` optionally removes a hashed share of drive *models* from train/val; their test-period rows
  are reported separately as ``test_unseen`` - the "does it transfer to a drive population it never saw" check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from driveguard.config import Settings


@dataclass(frozen=True)
class Boundaries:
    end_day: int
    test_start: int
    val_start: int
    val_end: int
    train_end: int
    horizon: int
    gap: int

    def as_dict(self) -> dict:
        return asdict(self)


def boundaries(gold: DataFrame, s: Settings) -> Boundaries:
    end_day = gold.agg(F.max("day")).first()[0]
    test_start = end_day - s.test_days + 1
    val_end = test_start - s.gap_days - 1
    val_start = val_end - s.val_days + 1
    train_end = val_start - s.gap_days - 1
    first_day = gold.agg(F.min("day")).first()[0]
    if train_end - first_day < s.horizon_days:
        raise ValueError(
            f"not enough history: {end_day - first_day + 1} days observed, need > "
            f"{s.test_days + s.val_days + 2 * s.gap_days + s.horizon_days} for this protocol"
        )
    return Boundaries(end_day, test_start, val_start, val_end, train_end, s.horizon_days, s.gap_days)


def assign(gold: DataFrame, s: Settings) -> tuple[DataFrame, Boundaries]:
    b = boundaries(gold, s)
    d = F.col("day")
    split = (
        F.when(d <= b.train_end, "train")
        .when((d >= b.val_start) & (d <= b.val_end), "val")
        .when(d >= b.test_start, "test")
        .otherwise("gap")
    )
    df = gold.withColumn("split", split)
    if s.group_holdout_frac > 0:
        bucket = F.pmod(F.xxhash64(F.col("model")), F.lit(1000))
        held = bucket < F.lit(int(round(s.group_holdout_frac * 1000)))
        df = df.withColumn("holdout_model", held.cast("int")).withColumn(
            "split",
            F.when(
                F.col("holdout_model") == 1, F.when(F.col("split") == "test", "test_unseen").otherwise("gap")
            ).otherwise(F.col("split")),
        )
    else:
        df = df.withColumn("holdout_model", F.lit(0))
    return df, b
