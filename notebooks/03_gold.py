# Databricks notebook source
# On serverless compute the `spark` global is created lazily; get the session explicitly so the notebook
# never depends on injection order (DatabricksSession on serverless, plain SparkSession elsewhere).
try:
    from databricks.connect import DatabricksSession

    spark = DatabricksSession.builder.getOrCreate()
except Exception:  # classic compute / local
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

import os

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "driveguard")
dbutils.widgets.text("raw_path", "/Volumes/workspace/driveguard/raw")
dbutils.widgets.text("experiment", "/Shared/driveguard")
dbutils.widgets.text("replay_anchor", "")
dbutils.widgets.text("horizon_days", "30")
os.environ["DG_HORIZON_DAYS"] = dbutils.widgets.get("horizon_days")

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

# MAGIC %md
# MAGIC # DriveGuard · 03 · Gold
# MAGIC Time-safe rolling SMART features and the right-censoring-aware 30-day failure label.
# MAGIC
# MAGIC Runs the same `driveguard` package as the CLI and the tests; this notebook only wires widgets to settings.

# COMMAND ----------

from driveguard import gold

print(gold.run(spark, settings))
g = spark.table(settings.table("gold_drive_day_features"))
display(g.groupBy("month").agg({"label": "sum", "censored": "sum", "*": "count"}).orderBy("month"))
