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
dbutils.widgets.text("endpoint_name", "driveguard-failure-30d")

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
# MAGIC # DriveGuard · 07 · Serving (optional)
# MAGIC Create or update a CPU Model Serving endpoint for the champion version and send one request. Free Edition allows a small number of CPU endpoints; scale-to-zero keeps it free when idle.
# MAGIC
# MAGIC Runs the same `driveguard` package as the CLI and the tests; this notebook only wires widgets to settings.

# COMMAND ----------

import json
import time

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
name = settings.registered_model_name()
version = MlflowClient().get_model_version_by_alias(name, settings.champion_alias).version
endpoint = dbutils.widgets.get("endpoint_name")
w = WorkspaceClient()
config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(entity_name=name, entity_version=version, workload_size="Small", scale_to_zero_enabled=True)
    ]
)
existing = {e.name for e in w.serving_endpoints.list()}
if endpoint in existing:
    w.serving_endpoints.update_config(name=endpoint, served_entities=config.served_entities)
else:
    w.serving_endpoints.create(name=endpoint, config=config)
print(f"endpoint {endpoint} -> {name} v{version}")

# COMMAND ----------

# wait until ready, then score one healthy and one flagged drive from today's gold table
for _ in range(60):
    st = w.serving_endpoints.get(endpoint).state
    if st and st.ready and st.ready.value == "READY":
        break
    time.sleep(20)
gold = spark.table(settings.table("gold_drive_day_features"))
latest = gold.agg({"date": "max"}).first()[0]
sample = (
    gold.filter(f"date = '{latest}' AND is_failure_row = 0")
    .orderBy("smart_197_raw", ascending=False)
    .limit(2)
    .toPandas()
)
numeric = [
    c
    for c in sample.columns
    if c
    not in (
        "serial_number",
        "date",
        "day",
        "month",
        "model",
        "label",
        "is_failure_row",
        "censored",
        "days_to_failure",
        "failure",
    )
]
payload = {
    "dataframe_split": {
        "columns": numeric + ["model"],
        "data": sample[numeric + ["model"]]
        .astype(object)
        .where(sample[numeric + ["model"]].notna(), None)
        .values.tolist(),
    }
}
resp = w.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations", body=payload)
print(json.dumps(resp, indent=1))
