#!/usr/bin/env python3
"""Print the latest training run's evaluation as a Markdown table (paste into the README results section).

    python scripts/report.py                      # local MLflow store under DG_DATA_ROOT
    python scripts/report.py --run-id <id>        # a specific run
On Databricks: run inside a notebook after `04_train`; the workspace tracking server is used automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import mlflow

from driveguard.config import Settings
from driveguard.train import setup_mlflow

ROWS = [
    ("PR-AUC", "pr_auc", "{:.3f}"),
    ("ROC-AUC", "roc_auc", "{:.3f}"),
    ("Recall @ budget (drive-days)", "recall_at_budget", "{:.1%}"),
    ("Precision @ budget", "precision_at_budget", "{:.1%}"),
    ("Drive recall @ budget", "drive_recall_at_budget", "{:.1%}"),
    ("Median lead time (days)", "lead_time_median_days", "{:.0f}"),
    ("Caught with >= 7 d lead", "share_caught_with_7d_lead", "{:.0%}"),
    ("False alerts / 1,000 healthy drive-days", "false_alerts_per_1000_drive_days", "{:.2f}"),
    ("Failing drives in period", "failing_drives", "{:.0f}"),
    ("Labeled drive-days", "rows_labeled", "{:,.0f}"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None)
    a = p.parse_args()
    s = Settings.from_env()
    setup_mlflow(s)
    if a.run_id:
        run_id = a.run_id
    else:
        exp = mlflow.get_experiment_by_name(
            s.experiment if s.on_databricks else s.experiment.strip("/").replace("/", "_")
        )
        runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
        run_id = runs.iloc[0]["run_id"]
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="metrics_report.json", dst_path=tmp)
        with open(path) as fh:
            rep = json.load(fh)
    run = mlflow.get_run(run_id)
    cols = [c for c in ("val", "test", "test_unseen", "baseline_test") if c in rep]
    names = {
        "val": "Validation",
        "test": "Test",
        "test_unseen": "Test (unseen models)",
        "baseline_test": "Operator rule (test)",
    }
    print(f"Run `{run_id}` · horizon {s.horizon_days} d · alert budget {s.alert_budget:.2%} of the fleet per day\n")
    print("| metric | " + " | ".join(names[c] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for label, key, fmt in ROWS:
        vals = []
        for c in cols:
            v = rep[c].get(key)
            vals.append("–" if v is None or v != v else fmt.format(v))
        print(f"| {label} | " + " | ".join(vals) + " |")
    b = {k.replace("split_", ""): v for k, v in run.data.params.items() if k.startswith("split_")}
    print(f"\nSplit boundaries (day numbers): {b}")
    print(f"Promotion: {run.data.tags.get('promotion', '-')}")
    if os.environ.get("DG_DATA_ROOT"):
        print(f"Tracking: {s.mlflow_tracking_uri}")


if __name__ == "__main__":
    main()
