"""Spark session and table I/O that work the same on Databricks (Unity Catalog + Delta) and locally (Parquet)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from driveguard.config import Settings

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def get_spark(app_name: str = "driveguard") -> SparkSession:
    from pyspark.sql import SparkSession

    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    builder = SparkSession.builder.appName(app_name)
    if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
        # Spark 3.5 + Arrow on Java 17/21 needs these module opens for pandas UDFs / toPandas
        java_opts = " ".join(
            [
                "-Dio.netty.tryReflectionSetAccessible=true",
                "--add-opens=java.base/java.nio=ALL-UNNAMED",
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
                "--add-opens=java.base/java.lang=ALL-UNNAMED",
                "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
                "--add-opens=java.base/java.util=ALL-UNNAMED",
                "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
                "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
            ]
        )
        # the driver JVM is launched by spark-submit before any builder config is read, so pass the options there
        if "PYSPARK_SUBMIT_ARGS" not in os.environ:
            os.environ["PYSPARK_SUBMIT_ARGS"] = f'--driver-java-options "{java_opts}" pyspark-shell'
        builder = (
            builder.master(os.environ.get("DG_SPARK_MASTER", "local[*]"))
            .config("spark.executor.extraJavaOptions", java_opts)
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.shuffle.partitions", os.environ.get("DG_SHUFFLE_PARTITIONS", "8"))
            .config("spark.driver.memory", os.environ.get("DG_DRIVER_MEMORY", "3g"))
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.ui.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")
        )
    spark = builder.getOrCreate()
    if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
        spark.sparkContext.setLogLevel("WARN")
    return spark


class TableStore:
    """Read/write named tables. Names come from ``Settings.table``: UC names on Databricks, paths locally."""

    def __init__(self, spark: SparkSession, settings: Settings):
        self.spark = spark
        self.s = settings

    def exists(self, name: str) -> bool:
        target = self.s.table(name)
        if self.s.catalog:
            return self.spark.catalog.tableExists(target)
        return os.path.isdir(target) and any(
            not f.startswith("_") and not f.startswith(".") for f in os.listdir(target)
        )

    def read(self, name: str) -> DataFrame:
        target = self.s.table(name)
        if self.s.catalog:
            return self.spark.table(target)
        return self.spark.read.format(self.s.table_format).load(target)

    def write(self, df: DataFrame, name: str, mode: str = "overwrite", partition_by: list[str] | None = None) -> None:
        target = self.s.table(name)
        writer = df.write.format(self.s.table_format).mode(mode)
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        if mode == "overwrite":
            writer = writer.option("overwriteSchema", "true")
        if mode == "append":
            writer = writer.option("mergeSchema", "true")
        if self.s.catalog:
            writer.saveAsTable(target)
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            writer.save(target)

    def ensure_schema(self) -> None:
        if self.s.catalog:
            self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.s.catalog}.{self.s.schema}")
        else:
            os.makedirs(os.path.join(self.s.data_root, "tables"), exist_ok=True)
