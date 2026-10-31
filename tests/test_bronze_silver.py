"""Raw-file quirks: schema drift between files, duplicates, ghost rows, unknown capacity, idempotent ingest."""

from __future__ import annotations

import os

from driveguard import bronze, silver
from driveguard.config import TABLES
from driveguard.spark import TableStore

EARLY = "date,serial_number,model,capacity_bytes,failure,smart_5_normalized,smart_5_raw,smart_9_raw\n"
LATE = "date,serial_number,model,capacity_bytes,failure,smart_5_normalized,smart_5_raw,smart_9_raw,smart_240_raw,datacenter\n"


def _write(path: str, header: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(header + "\n".join(lines) + "\n")


def test_bronze_handles_schema_drift_and_is_idempotent(spark, settings):
    raw = settings.raw_path
    _write(
        f"{raw}/data_Q1_2024/2024-01-01.csv",
        EARLY,
        ["2024-01-01,S1,M,-1,0,100,0,10", "2024-01-01,S2,M,1000,0,100,3,10"],
    )
    _write(
        f"{raw}/data_Q2_2024/2024-04-01.csv",
        LATE,
        ["2024-04-01,S1,M,1000,0,100,0,2170,99,sac2", "2024-04-01,S2,M,1000,1,90,40,2170,99,sac2"],
    )
    assert bronze.run(spark, settings) == 2
    store = TableStore(spark, settings)
    b = store.read(TABLES["bronze"])
    assert b.count() == 4
    assert "smart_240_raw" in b.columns and "datacenter" in b.columns
    early = b.filter("month = '2024-01'").toPandas()
    assert early["smart_240_raw"].isna().all() and early["datacenter"].isna().all()
    assert str(b.schema["date"].dataType) == "DateType()"
    assert b.schema["smart_5_raw"].dataType.simpleString() == "bigint"
    # second run: nothing new
    assert bronze.run(spark, settings) == 0
    assert store.read(TABLES["bronze"]).count() == 4
    # a new file appears later -> only that one lands
    _write(f"{raw}/data_Q2_2024/2024-04-02.csv", LATE, ["2024-04-02,S1,M,1000,0,100,0,2194,99,sac2"])
    assert bronze.run(spark, settings) == 1
    assert store.read(TABLES["bronze"]).count() == 5


def test_silver_rules(spark, settings):
    raw = settings.raw_path
    _write(
        f"{raw}/data_Q1_2024/2024-01-01.csv",
        EARLY,
        [
            "2024-01-01,S1,M,-1,0,100,0,10",
            "2024-01-01,S1,M,-1,0,100,0,10",  # duplicate drive-day
            "2024-01-01,S2,M,1000,0,100,0,10",
        ],
    )
    _write(
        f"{raw}/data_Q1_2024/2024-01-02.csv",
        EARLY,
        [
            "2024-01-02,S1,M,1000,0,100,0,34",
            "2024-01-02,S2,M,1000,1,90,40,34",  # S2 fails
            "2024-01-02,S2,M,1000,0,90,40,34",  # duplicate of the failure day without the flag -> failure row must win
        ],
    )
    _write(
        f"{raw}/data_Q1_2024/2024-01-03.csv",
        EARLY,
        ["2024-01-03,S1,M,1000,0,100,0,58", "2024-01-03,S2,M,1000,0,90,40,58"],
    )  # ghost after failure
    bronze.run(spark, settings)
    out = silver.run(spark, settings)
    store = TableStore(spark, settings)
    sil = store.read(TABLES["silver"]).toPandas().sort_values(["serial_number", "date"])
    assert out["silver_rows"] == 5  # 6 unique drive-days minus the ghost
    s1 = sil[sil.serial_number == "S1"]
    assert (s1.capacity_bytes == 1000).all()  # -1 replaced by the drive's known capacity
    s2 = sil[sil.serial_number == "S2"]
    assert list(s2.failure) == [0, 1] and len(s2) == 2
    lc = store.read(TABLES["lifecycle"]).toPandas().set_index("serial_number")
    assert lc.loc["S2", "failed"] == 1 and str(lc.loc["S2", "failure_date"]) == "2024-01-02"
    assert lc.loc["S1", "failed"] == 0 and lc.loc["S1", "n_days_observed"] == 3
