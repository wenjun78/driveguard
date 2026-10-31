"""Operational metrics for a ranked daily alert list.

The deployed artefact is not a probability, it is a *ranked list* an administrator works through with a fixed
budget (people, spare drives, maintenance windows). So besides threshold-free PR-AUC we report what happens at the
budget: of the drives that will fail within the horizon, how many were flagged (recall at budget), how clean the
list is (precision at budget), how early the first alert came (lead time), and how much noise a healthy fleet
generates (false alerts per 1,000 drive-days).

Row level treats every drive-day as a decision; drive level asks the question operators care about - was this
failing drive flagged at least once, in time.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def flag_at_budget(df: pd.DataFrame, score_col: str, budget: float) -> pd.Series:
    """Per day, flag the top ``ceil(budget * drives scored that day)`` rows (at least one)."""
    out = pd.Series(False, index=df.index)
    for _, idx in df.groupby("day").groups.items():
        sub = df.loc[idx, score_col]
        k = max(1, math.ceil(budget * len(sub)))
        top = sub.sort_values(ascending=False, kind="mergesort").index[:k]
        out.loc[top] = True
    return out


def evaluate(
    df: pd.DataFrame, score_col: str, budget: float, horizon: int, flag_col: str | None = None
) -> dict[str, float]:
    """``df`` holds one row per scored drive-day: serial_number, day, label (0/1/NaN), days_to_failure, score.

    Pass ``flag_col`` when the per-day ranking was already done (e.g. in Spark, see ``scoring.flag_at_budget``).
    """
    df = df.copy()
    df["_flag"] = df[flag_col].astype(bool) if flag_col else flag_at_budget(df, score_col, budget)
    lab = df[df["label"].notna()].copy()
    lab["label"] = lab["label"].astype(int)
    out: dict[str, float] = {
        "rows_scored": float(len(df)),
        "rows_labeled": float(len(lab)),
        "positives": float(lab["label"].sum()),
        "alerts_per_day": float(df.groupby("day")["_flag"].sum().mean()) if len(df) else float("nan"),
    }
    if lab["label"].nunique() < 2:
        return out
    y, s = lab["label"].to_numpy(), lab[score_col].to_numpy()
    out["pr_auc"] = float(average_precision_score(y, s))
    out["roc_auc"] = float(roc_auc_score(y, s))
    if np.nanmin(s) >= 0 and np.nanmax(s) <= 1:
        out["brier"] = float(brier_score_loss(y, s))

    flagged = lab["_flag"].to_numpy()
    tp = int(((y == 1) & flagged).sum())
    out["recall_at_budget"] = tp / max(1, int((y == 1).sum()))
    out["precision_at_budget"] = tp / max(1, int(flagged.sum()))
    out["false_alerts_per_1000_drive_days"] = 1000.0 * int(((y == 0) & flagged).sum()) / max(1, int((y == 0).sum()))

    # drive level
    failing = lab[lab["label"] == 1]
    n_failing = failing["serial_number"].nunique()
    caught = failing[failing["_flag"]]
    out["failing_drives"] = float(n_failing)
    out["drive_recall_at_budget"] = caught["serial_number"].nunique() / max(1, n_failing)
    if len(caught):
        lead = caught.groupby("serial_number")["days_to_failure"].max()
        out["lead_time_median_days"] = float(lead.median())
        out["lead_time_mean_days"] = float(lead.mean())
        out["share_caught_with_7d_lead"] = float((lead >= 7).mean())
    else:
        out["lead_time_median_days"] = float("nan")
        out["lead_time_mean_days"] = float("nan")
        out["share_caught_with_7d_lead"] = 0.0
    out["horizon_days"] = float(horizon)
    out["alert_budget"] = float(budget)
    return out


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index between two samples of one feature (quantile bins from the reference)."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) < 10 or len(cur) < 10:
        return float("nan")
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:  # near-constant feature: compare the share of non-zero / non-mode values instead
        mode = qs[0]
        r = np.array([(ref == mode).mean(), (ref != mode).mean()])
        c = np.array([(cur == mode).mean(), (cur != mode).mean()])
    else:
        qs[0], qs[-1] = -np.inf, np.inf
        r = np.histogram(ref, qs)[0] / len(ref)
        c = np.histogram(cur, qs)[0] / len(cur)
    eps = 1e-4
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))
