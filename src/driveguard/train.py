"""Train, evaluate against the operator baseline, log to MLflow, and promote through a champion/challenger gate."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from driveguard import gold as gold_mod
from driveguard import metrics as M
from driveguard import scoring, splits
from driveguard.config import TABLES, Settings
from driveguard.model import CATEGORICAL, DriveGuardModel, FeatureSpec, PyFuncWrapper, prepare
from driveguard.spark import TableStore

log = logging.getLogger(__name__)
EVAL_KEEP = ["serial_number", "day", "label", "days_to_failure"]


def _plain(pdf: pd.DataFrame) -> pd.DataFrame:
    """Drop DataFrame.attrs: Databricks serverless attaches Spark plan metrics there, which pandas then fails to
    JSON-serialize into Parquet metadata / MLflow examples."""
    pdf.attrs = {}
    return pdf


def setup_mlflow(s: Settings) -> None:
    if s.mlflow_tracking_uri:
        os.makedirs(s.data_root, exist_ok=True)
        mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    if s.catalog:
        mlflow.set_registry_uri("databricks-uc")
    exp = s.experiment if s.on_databricks else s.experiment.strip("/").replace("/", "_")
    if mlflow.get_experiment_by_name(exp) is None:
        if s.on_databricks:
            mlflow.create_experiment(exp)
        else:
            mlflow.create_experiment(exp, artifact_location=s.mlflow_artifact_root)
    mlflow.set_experiment(exp)


def sample_training(df: DataFrame, ratio: int, seed: int) -> tuple[DataFrame, float]:
    """All positives plus a negatives sample; the sampling fraction becomes the negatives' weight so the fitted
    probabilities keep the fleet's true base rate."""
    pos = df.filter("label = 1")
    neg = df.filter("label = 0")
    n_pos, n_neg = pos.count(), neg.count()
    if n_pos == 0:
        raise ValueError("no positive rows in the training period")
    frac = min(1.0, ratio * n_pos / max(1, n_neg))
    neg = neg.sample(withReplacement=False, fraction=frac, seed=seed)
    out = pos.withColumn("weight", F.lit(1.0)).unionByName(neg.withColumn("weight", F.lit(1.0 / frac)))
    return out, frac


def make_spec(train_pdf: pd.DataFrame, numeric: list[str], s: Settings) -> FeatureSpec:
    counts = train_pdf[CATEGORICAL].astype(str).value_counts()
    vocab = {m: i for i, m in enumerate(counts.index)}
    freq = (counts / counts.sum()).to_dict()
    return FeatureSpec(
        numeric=numeric, model_vocab=vocab, model_freq=freq, horizon_days=s.horizon_days, alert_budget=s.alert_budget
    )


def fit(train_pdf: pd.DataFrame, val_pdf: pd.DataFrame, spec: FeatureSpec, s: Settings) -> tuple[xgb.Booster, dict]:
    dtrain = xgb.DMatrix(
        prepare(train_pdf, spec), label=train_pdf["label"], weight=train_pdf["weight"], feature_names=spec.columns
    )
    dval = xgb.DMatrix(
        prepare(val_pdf, spec), label=val_pdf["label"], weight=val_pdf["weight"], feature_names=spec.columns
    )
    params = dict(s.xgb_params, seed=s.seed)
    history: dict = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=s.xgb_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        evals_result=history,
        verbose_eval=False,
    )
    info = {
        "best_iteration": int(booster.best_iteration),
        "best_val_aucpr": float(history["val"]["aucpr"][booster.best_iteration]),
    }
    return booster, info


def evaluate_split(df: DataFrame, model: DriveGuardModel, s: Settings) -> dict[str, float]:
    scored = scoring.flag_at_budget(scoring.score_spark(df, model, EVAL_KEEP), s.alert_budget)
    pdf = scored.select(*EVAL_KEEP, "score", "flagged").toPandas()
    return M.evaluate(pdf, "score", s.alert_budget, s.horizon_days, flag_col="flagged")


def evaluate_baseline(df: DataFrame, s: Settings) -> dict[str, float]:
    scored = scoring.flag_at_budget(scoring.rule_baseline_score(df), s.alert_budget)
    pdf = scored.select(*EVAL_KEEP, "score", "flagged").toPandas()
    return M.evaluate(pdf, "score", s.alert_budget, s.horizon_days, flag_col="flagged")


def promote_if_better(client: MlflowClient, name: str, version: str, alias: str, new_val_pr_auc: float) -> str:
    try:
        champ = client.get_model_version_by_alias(name, alias)
    except Exception:
        champ = None
    if champ is not None:
        prev = client.get_run(champ.run_id).data.metrics.get("val_pr_auc", float("-inf"))
        if new_val_pr_auc < prev:
            return f"kept champion v{champ.version} (val_pr_auc {prev:.4f} >= {new_val_pr_auc:.4f})"
    client.set_registered_model_alias(name, alias, version)
    return f"promoted v{version} to @{alias}"


def run(spark: SparkSession, s: Settings) -> dict[str, Any]:
    setup_mlflow(s)
    store = TableStore(spark, s)
    gold = store.read(TABLES["gold"])
    # Replay honesty: "now" may be a cursor inside the historical data. Train only on rows whose label window closed
    # before the cursor, so a retrain triggered mid-replay never sees outcomes that have not "happened" yet.
    from datetime import timedelta

    from driveguard import replay

    now = replay.resolve_as_of(None, gold.agg(F.max("date")).first()[0])
    cutoff = now - timedelta(days=s.horizon_days)
    gold = gold.filter(F.col("date") <= F.lit(cutoff))
    df, b = splits.assign(gold, s)
    df = df.filter("is_failure_row = 0")
    numeric = [c for c in gold_mod.feature_columns(gold) if c not in (CATEGORICAL,)]

    train_df, neg_frac = sample_training(
        df.filter("split = 'train' AND label IS NOT NULL"), s.negative_sample_ratio, s.seed
    )
    val_df, _ = sample_training(df.filter("split = 'val' AND label IS NOT NULL"), s.negative_sample_ratio, s.seed + 1)
    train_pdf = _plain(train_df.select(*numeric, CATEGORICAL, "label", "weight").toPandas())
    val_pdf = _plain(val_df.select(*numeric, CATEGORICAL, "label", "weight").toPandas())
    spec = make_spec(train_pdf, numeric, s)
    train_period = df.filter("split = 'train'")
    n_train_period = train_period.count()
    reference_pdf = _plain(
        train_period.sample(fraction=min(1.0, 25000 / max(1, n_train_period)), seed=s.seed)
        .select(*numeric, CATEGORICAL)
        .limit(20000)
        .toPandas()
    )

    with mlflow.start_run(run_name="driveguard-xgb") as run:
        mlflow.log_params({f"split_{k}": v for k, v in b.as_dict().items()})
        mlflow.log_params({"data_cutoff": cutoff.isoformat(), "replay_now": now.isoformat()})
        mlflow.log_params(
            {
                "horizon_days": s.horizon_days,
                "alert_budget": s.alert_budget,
                "negative_sample_ratio": s.negative_sample_ratio,
                "neg_sample_frac": round(neg_frac, 5),
                "n_features": len(spec.columns),
                **{f"xgb_{k}": v for k, v in s.xgb_params.items()},
            }
        )
        mlflow.log_metrics(
            {
                "train_rows": len(train_pdf),
                "train_positives": int(train_pdf["label"].sum()),
                "val_rows": len(val_pdf),
                "val_positives": int(val_pdf["label"].sum()),
            }
        )

        booster, info = fit(train_pdf, val_pdf, spec, s)
        spec.extras = info
        model = DriveGuardModel(booster, spec)
        mlflow.log_metrics(
            {"best_iteration": info["best_iteration"], "val_aucpr_early_stopping": info["best_val_aucpr"]}
        )

        report: dict[str, dict] = {}
        for split in ("val", "test", "test_unseen"):
            part = df.filter(F.col("split") == split)
            if part.limit(1).count() == 0:
                continue
            report[split] = evaluate_split(part, model, s)
            mlflow.log_metrics(
                {f"{split}_{k}": v for k, v in report[split].items() if isinstance(v, float) and not np.isnan(v)}
            )
        report["baseline_test"] = evaluate_baseline(df.filter("split = 'test'"), s)
        mlflow.log_metrics(
            {
                f"baseline_test_{k}": v
                for k, v in report["baseline_test"].items()
                if isinstance(v, float) and not np.isnan(v)
            }
        )

        gain = booster.get_score(importance_type="gain")
        importance = sorted(gain.items(), key=lambda kv: -kv[1])
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "metrics_report.json"), "w") as fh:
                json.dump(report, fh, indent=1, default=float)
            with open(os.path.join(tmp, "feature_importance.json"), "w") as fh:
                json.dump(importance, fh, indent=1)
            mlflow.log_artifact(os.path.join(tmp, "metrics_report.json"))
            mlflow.log_artifact(os.path.join(tmp, "feature_importance.json"))

            model_dir = os.path.join(tmp, "model_dir")
            model.save(model_dir)
            # drift reference travels with the model: an *unbiased* sample of the training-period fleet
            # (not the class-rebalanced training set, which would make every healthy day look like drift)
            reference_pdf.to_parquet(os.path.join(model_dir, "reference_sample.parquet"), index=False)
            example = train_pdf[numeric + [CATEGORICAL]].head(5).copy()
            example[numeric] = example[numeric].astype("float64")  # floats so NaNs pass schema enforcement at serving
            signature = infer_signature(example, pd.Series(model.predict_proba(example), name="p_fail_within_horizon"))
            code_path = os.path.dirname(os.path.abspath(__file__))
            kwargs = dict(
                python_model=PyFuncWrapper(),
                artifacts={"model_dir": model_dir},
                signature=signature,
                input_example=example,
                code_paths=[code_path],
                pip_requirements=[
                    f"xgboost=={xgb.__version__}",
                    f"pandas=={pd.__version__}",
                    f"numpy=={np.__version__}",
                    "scikit-learn",
                ],
            )
            try:
                logged = mlflow.pyfunc.log_model(name="model", **kwargs)
            except TypeError:  # MLflow < 3
                logged = mlflow.pyfunc.log_model(artifact_path="model", **kwargs)

        promotion = "registration disabled"
        if s.register_model:
            name = s.registered_model_name()
            try:
                client = MlflowClient()
                mv = mlflow.register_model(logged.model_uri, name)
                promotion = promote_if_better(
                    client, name, mv.version, s.champion_alias, report.get("val", {}).get("pr_auc", 0.0)
                )
            except Exception as exc:  # registry unavailable (e.g. plain file store without alias support)
                promotion = f"registration skipped: {type(exc).__name__}: {exc}"
        mlflow.set_tag("promotion", promotion)
        log.info("promotion: %s", promotion)

    return {
        "run_id": run.info.run_id,
        "model_uri": f"runs:/{run.info.run_id}/model",
        "promotion": promotion,
        "boundaries": b.as_dict(),
        "fit": info,
        "report": report,
        "importance_top10": importance[:10],
    }
