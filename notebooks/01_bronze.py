# Databricks notebook source
# MAGIC %md
# MAGIC # DriveGuard · 01 · Bronze
# MAGIC Land new Backblaze daily files from the raw Volume into `bronze_smart_daily` (idempotent: files already logged in `ops_ingest_log` are skipped).
# MAGIC
# MAGIC Runs the same `driveguard` package as the CLI and the tests; this notebook only wires widgets to settings.

# COMMAND ----------

import os

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "driveguard")
dbutils.widgets.text("raw_path", "/Volumes/workspace/driveguard/raw")
dbutils.widgets.text("experiment", "/Shared/driveguard")

os.environ["DG_CATALOG"] = dbutils.widgets.get("catalog")
os.environ["DG_SCHEMA"] = dbutils.widgets.get("schema")
os.environ["DG_RAW_PATH"] = dbutils.widgets.get("raw_path")
os.environ["DG_EXPERIMENT"] = dbutils.widgets.get("experiment")
os.environ["DG_TABLE_FORMAT"] = "delta"

from driveguard.config import Settings

settings = Settings.from_env()
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {settings.catalog}.{settings.schema}")

# COMMAND ----------

from driveguard import bronze

n = bronze.run(spark, settings, settings.raw_path)
print({"files_ingested": n})
display(spark.table(settings.table("bronze_smart_daily")).groupBy("month").count().orderBy("month"))
