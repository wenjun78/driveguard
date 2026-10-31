# Databricks notebook source
# MAGIC %md
# MAGIC # DriveGuard · 04 · Train
# MAGIC Time-ordered split with horizon-long gaps, XGBoost on a negatives-sampled training set, operator-baseline comparison, MLflow logging, Unity Catalog registration and the champion/challenger gate.
# MAGIC
# MAGIC Runs the same `driveguard` package as the CLI and the tests; this notebook only wires widgets to settings.

# COMMAND ----------

import os

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "driveguard")
dbutils.widgets.text("raw_path", "/Volumes/workspace/driveguard/raw")
dbutils.widgets.text("experiment", "/Shared/driveguard")
dbutils.widgets.text("alert_budget", "0.005")
os.environ["DG_ALERT_BUDGET"] = dbutils.widgets.get("alert_budget")

os.environ["DG_CATALOG"] = dbutils.widgets.get("catalog")
os.environ["DG_SCHEMA"] = dbutils.widgets.get("schema")
os.environ["DG_RAW_PATH"] = dbutils.widgets.get("raw_path")
os.environ["DG_EXPERIMENT"] = dbutils.widgets.get("experiment")
os.environ["DG_TABLE_FORMAT"] = "delta"

from driveguard.config import Settings

settings = Settings.from_env()
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {settings.catalog}.{settings.schema}")

# COMMAND ----------

import json

from driveguard import train

result = train.run(spark, settings)
print(json.dumps({k: v for k, v in result.items() if k != "report"}, indent=1, default=str))
print(json.dumps(result["report"], indent=1, default=float))
dbutils.jobs.taskValues.set("run_id", result["run_id"])
dbutils.jobs.taskValues.set("promotion", result["promotion"])
