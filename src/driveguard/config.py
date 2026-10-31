"""Runtime settings.

One code path runs in two places:

* Databricks: tables live in Unity Catalog (``catalog.schema.table``, Delta), raw files in a UC Volume,
  MLflow tracking + model registry are the workspace's own.
* Local / CI: tables are Parquet directories under ``data_root/tables``, MLflow tracks to ``data_root/mlruns``.

Everything is driven by environment variables (prefixed ``DG_``) so the same wheel works unchanged in a
Databricks job, a laptop and GitHub Actions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

# SMART attributes with a documented relationship to impending failure. Backblaze's own "five SMART stats"
# are 5, 187, 188, 197, 198; the rest are commonly reported wear / error / thermal counters.
KEY_SMART_RAW: tuple[int, ...] = (5, 187, 188, 197, 198)
CONTEXT_SMART_RAW: tuple[int, ...] = (1, 3, 4, 7, 9, 10, 12, 192, 193, 194, 196, 199, 240, 241, 242)
KEY_SMART_NORMALIZED: tuple[int, ...] = (5, 187, 197, 198, 194)


def _env(name: str, default):
    raw = os.environ.get(f"DG_{name}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes", "y"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Settings:
    # Where tables live. On Databricks set DG_CATALOG and DG_SCHEMA; locally leave them empty and set DG_DATA_ROOT.
    catalog: str = ""
    schema: str = "driveguard"
    data_root: str = "./data"
    table_format: str = "parquet"  # "delta" on Databricks
    # Raw Backblaze files: a directory of daily CSVs (any nesting) and/or Parquet subsets made by scripts/download_backblaze.py
    raw_path: str = "./data/raw"

    # Problem definition
    horizon_days: int = 30  # predict failure within (t, t + horizon]
    alert_budget: float = 0.005  # share of the fleet that may be flagged per day (0.5 %)
    feature_windows: tuple[int, ...] = (7, 30)

    # Evaluation protocol
    val_days: int = 60  # length of the validation period
    test_days: int = 90  # length of the test period (labels in its last `horizon_days` are censored and excluded)
    gap_days: int = 30  # unlabeled gap between periods so training labels never peek into evaluation time
    group_holdout_frac: float = 0.0  # optional: hold out this share of drive models as an unseen-population test

    # Training
    negative_sample_ratio: int = 20  # keep all positives, sample negatives at this ratio (weights restore the prior)
    seed: int = 7
    xgb_rounds: int = 600
    xgb_params: dict = field(
        default_factory=lambda: {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.0,
            "tree_method": "hist",
        }
    )

    # MLflow
    experiment: str = "/Shared/driveguard"
    model_name: str = "driveguard_failure_30d"
    champion_alias: str = "champion"
    register_model: bool = True

    # Monitoring
    psi_warn: float = 0.1
    psi_alert: float = 0.2
    psi_features_for_retrain: int = 3  # retrain when at least this many key features exceed psi_alert
    matured_recall_floor: float = 0.3  # or when matured recall@budget (mean of recent snapshots) drops below this;
    # on the real fleet single-day matured recall runs ~0.3-0.6 (median ~0.5), so 0.3 marks real degradation

    @classmethod
    def from_env(cls) -> Settings:
        s = cls()
        for f in fields(cls):
            if f.name in {"xgb_params", "feature_windows"}:
                continue
            setattr(s, f.name, _env(f.name.upper(), getattr(s, f.name)))
        if (w := os.environ.get("DG_FEATURE_WINDOWS")) is not None:
            s.feature_windows = tuple(int(x) for x in w.split(","))
        return s

    # ---- naming -------------------------------------------------------------------------------------------------
    @property
    def on_databricks(self) -> bool:
        return bool(self.catalog) or "DATABRICKS_RUNTIME_VERSION" in os.environ

    def table(self, name: str) -> str:
        """Fully qualified table name (Databricks) or Parquet path (local)."""
        if self.catalog:
            return f"{self.catalog}.{self.schema}.{name}"
        return os.path.join(self.data_root, "tables", name)

    def registered_model_name(self) -> str:
        if self.catalog:
            return f"{self.catalog}.{self.schema}.{self.model_name}"
        return self.model_name

    @property
    def mlflow_tracking_uri(self) -> str | None:
        if self.on_databricks:
            return None  # use the workspace tracking server
        return "sqlite:///" + os.path.abspath(os.path.join(self.data_root, "mlflow.db"))

    @property
    def mlflow_artifact_root(self) -> str:
        return os.path.abspath(os.path.join(self.data_root, "mlartifacts"))


TABLES = {
    "bronze": "bronze_smart_daily",
    "silver": "silver_smart_daily",
    "lifecycle": "silver_drive_lifecycle",
    "gold": "gold_drive_day_features",
    "alerts": "gold_risk_alerts",
    "monitor": "ops_monitoring",
    "ingest_log": "ops_ingest_log",
}
