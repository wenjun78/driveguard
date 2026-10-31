"""Developer-only twin of ``DatabricksSource`` over the local Parquet tables and sqlite MLflow store.

It reuses every query in ``DatabricksSource`` by swapping the SQL engine for DuckDB (same SQL, two tiny dialect
rewrites), so the snapshot the console ships with is produced by the *same* code path as the live view.
"""

from __future__ import annotations

import glob
import os
import re

import duckdb

from driveguard.app.sources import DatabricksSource
from driveguard.config import TABLES, Settings
from driveguard.model import DriveGuardModel
from driveguard.train import setup_mlflow


class LocalSource(DatabricksSource):
    kind = "snapshot"

    def __init__(self, settings: Settings):
        self.s = settings
        self.endpoint = "driveguard-failure-30d"
        self.daily_job_name, self.build_job_name = "driveguard - daily", "driveguard - build"
        self._model = None
        self._model_version = None
        setup_mlflow(settings)

    def t(self, name: str) -> str:
        path = self.s.table(TABLES[name])
        files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
        if not files:
            raise FileNotFoundError(path)
        return (
            f"read_parquet('{os.path.join(path, '**', '*.parquet')}', hive_partitioning = true, union_by_name = true)"
        )

    def query(self, sql: str, params: dict | None = None) -> list[dict]:
        sql = re.sub(r"DATE_SUB\(:(\w+), (\d+)\)", r"(CAST($\1 AS DATE) - INTERVAL \2 DAY)", sql)
        sql = re.sub(r":(\w+)\b", r"$\1", sql)
        con = duckdb.connect()
        try:
            cur = con.execute(sql, params or {})
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
        finally:
            con.close()

    def _mlflow_client(self):
        from mlflow.tracking import MlflowClient

        return MlflowClient()

    def _ensure_model(self) -> DriveGuardModel:
        import mlflow

        _, name, mv = self._champion()
        if self._model is None or self._model_version != mv.version:
            local = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{mv.version}/artifacts/model_dir")
            self._model = DriveGuardModel.load(local)
            self._model_version = mv.version
        return self._model

    def status(self) -> dict:
        return {
            "source": self.kind,
            "as_of": self.now().isoformat(),
            "note": "demo snapshot exported from a local synthetic run",
        }

    def pipeline(self) -> dict:
        return {
            "jobs": [],
            "serving": None,
            "replay": {"anchor": None, "cursor": None},
            "workspace": None,
            "note": "Snapshot mode: job runs, the serving endpoint and the replay clock appear once the console is pointed at a Databricks workspace.",
        }

    def predict(self, features) -> dict:
        model = self._ensure_model()
        p = float(model.predict_proba(features)[0])
        return {
            "p_fail_within_horizon": p,
            "reason_codes": model.reason_codes(features, top=4)[0],
            "served_by": f"local copy of champion v{self._model_version} (snapshot)",
            "local_p": p,
        }
