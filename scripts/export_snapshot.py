#!/usr/bin/env python3
"""Export the console's snapshot - from a local pipeline run (``make demo``) or from the live Databricks workspace.

    DG_DATA_ROOT=./data python scripts/export_snapshot.py --out app_snapshot                 # from a local run
    DATABRICKS_HOST=... DATABRICKS_TOKEN=... DATABRICKS_HTTP_PATH=... DG_CATALOG=workspace \
        python scripts/export_snapshot.py --source databricks --out app_snapshot            # from the workspace

Writes overview/alerts/drives/monitoring/model/pipeline JSON plus the champion model directory, in exactly the
shapes the live Databricks source returns, so the console behaves identically in both modes. The GitHub daily
workflow runs the second form and commits the result, so the console's fallback is a recent copy of the real thing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone

import mlflow

from driveguard.config import Settings


def _compact(v):
    """Round floats and drop NaN so the JSON stays small; dates/strings pass through."""
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()[:10]
    if isinstance(v, float):
        if v != v:
            return None
        return int(v) if v.is_integer() and abs(v) < 1e15 else float(f"{v:.5g}")
    if hasattr(v, "item"):
        return _compact(v.item())
    return v


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="app_snapshot")
    p.add_argument("--source", choices=["local", "databricks"], default="local")
    p.add_argument("--note", default=None)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.source == "databricks":
        from driveguard.app.sources import DatabricksSource

        settings = Settings.from_env()
        settings.catalog = settings.catalog or os.environ.get("DG_CATALOG", "workspace")
        src = DatabricksSource(
            settings, os.environ["DATABRICKS_HOST"], os.environ["DATABRICKS_TOKEN"], os.environ["DATABRICKS_HTTP_PATH"]
        )
        note = a.note or (
            f"Snapshot of the live Databricks workspace ({settings.catalog}.{settings.schema}) exported "
            f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC; the console serves it whenever the workspace is unreachable."
        )
    else:
        from driveguard.app.local_source import LocalSource

        src = LocalSource(Settings.from_env())
        note = (
            a.note
            or "Demo snapshot from a synthetic 3,000-drive fleet (not real Backblaze results); the live console reads a Databricks workspace."
        )

    overview = src.overview()
    overview["note"] = note
    json.dump(overview, open(os.path.join(a.out, "overview.json"), "w"), indent=0, default=str)

    alerts = {}
    drives = {}
    for d in src.alert_dates():
        day = src.alerts(d)
        alerts[d] = day["rows"]
        for row in day["rows"]:
            sn = row["serial"]
            if sn not in drives:
                drives[sn] = src.drive(sn)
                feats = src.latest_features(sn)
                lf = feats.iloc[0].to_dict() if feats is not None else {}
                drives[sn]["latest_features"] = {
                    k: _compact(v) for k, v in lf.items() if _compact(v) is not None
                } or None
                for h in drives[sn]["history"]:
                    for k, v in h.items():
                        h[k] = _compact(v)
    json.dump(alerts, open(os.path.join(a.out, "alerts.json"), "w"), indent=0, default=str)
    json.dump(drives, open(os.path.join(a.out, "drives.json"), "w"), indent=0, default=str)
    json.dump(src.monitoring(), open(os.path.join(a.out, "monitoring.json"), "w"), indent=0, default=str)
    json.dump(src.model_info(), open(os.path.join(a.out, "model.json"), "w"), indent=0, default=str)
    json.dump(src.pipeline(), open(os.path.join(a.out, "pipeline.json"), "w"), indent=0, default=str)

    _, name, mv = src._champion()
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/{mv.version}/artifacts/model_dir", dst_path=tempfile.mkdtemp()
    )
    dest = os.path.join(a.out, "model_dir")
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(local, dest)
    for f in os.listdir(dest):  # the drift reference is not needed by the console
        if f.startswith("reference_sample"):
            os.remove(os.path.join(dest, f))
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "alert_dates": len(alerts),
        "drives": len(drives),
        "champion_version": mv.version,
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
