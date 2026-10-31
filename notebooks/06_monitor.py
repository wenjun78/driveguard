# Databricks notebook source
# MAGIC %md
# MAGIC # DriveGuard · 06 · Monitor
# MAGIC PSI drift on key features and on the score distribution, matured-label precision/recall for past snapshots, and the retraining decision exposed as a task value for the workflow's condition task.
# MAGIC
# MAGIC Runs the same `driveguard` package as the CLI and the tests; this notebook only wires widgets to settings.

# COMMAND ----------

import os

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "driveguard")
dbutils.widgets.text("raw_path", "/Volumes/workspace/driveguard/raw")
dbutils.widgets.text("experiment", "/Shared/driveguard")
dbutils.widgets.text("replay_anchor", "")

os.environ["DG_CATALOG"] = dbutils.widgets.get("catalog")
os.environ["DG_SCHEMA"] = dbutils.widgets.get("schema")
os.environ["DG_RAW_PATH"] = dbutils.widgets.get("raw_path")
os.environ["DG_EXPERIMENT"] = dbutils.widgets.get("experiment")
if dbutils.widgets.get("replay_anchor"):
    os.environ["DG_REPLAY_ANCHOR"] = dbutils.widgets.get("replay_anchor")
os.environ["DG_TABLE_FORMAT"] = "delta"

from driveguard.config import Settings

settings = Settings.from_env()
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {settings.catalog}.{settings.schema}")

# COMMAND ----------

import json

from driveguard import monitor

result = monitor.run(spark, settings)
print(json.dumps({k: v for k, v in result.items() if k not in ("matured",)}, indent=1, default=str))
dbutils.jobs.taskValues.set("retrain_recommended", "true" if result["retrain_recommended"] else "false")
display(spark.table(settings.table("ops_monitoring")).orderBy("monitored_at", ascending=False).limit(20))
