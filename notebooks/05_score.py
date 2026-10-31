# Databricks notebook source
# MAGIC %md
# MAGIC # DriveGuard · 05 · Score
# MAGIC Rank the latest snapshot with the champion model and write `gold_risk_alerts` (score, rank, flagged at the alert budget, reason codes).
# MAGIC
# MAGIC Runs the same `driveguard` package as the CLI and the tests; this notebook only wires widgets to settings.

# COMMAND ----------

import os

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "driveguard")
dbutils.widgets.text("raw_path", "/Volumes/workspace/driveguard/raw")
dbutils.widgets.text("experiment", "/Shared/driveguard")
dbutils.widgets.text("replay_anchor", "")
dbutils.widgets.text("as_of", "")
dbutils.widgets.text("backfill_days", "0")

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

from driveguard import score

as_of = dbutils.widgets.get("as_of") or None
backfill_days = int(dbutils.widgets.get("backfill_days") or 0)
if backfill_days > 0:
    result = score.backfill(spark, settings, backfill_days)
else:
    result = score.run(spark, settings, as_of)
print(json.dumps(result, indent=1, default=str))
display(spark.table(settings.table("gold_risk_alerts")).filter("flagged = 1").orderBy("date", "rank_in_day").limit(50))
