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
# MAGIC # DriveGuard · 02 · Silver
# MAGIC Clean, type and deduplicate drive-days; drop post-failure ghost rows; build the per-drive lifecycle table.
# MAGIC
# MAGIC Runs the same `driveguard` package as the CLI and the tests; this notebook only wires widgets to settings.

# COMMAND ----------

from driveguard import silver

print(silver.run(spark, settings))
display(
    spark.table(settings.table("silver_drive_lifecycle"))
    .groupBy("model")
    .agg({"serial_number": "count", "failed": "sum"})
    .orderBy("model")
)
