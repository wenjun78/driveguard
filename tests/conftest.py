from __future__ import annotations

import os

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture(scope="session")
def spark():
    os.environ["DG_SPARK_MASTER"] = "local[2]"
    os.environ["DG_SHUFFLE_PARTITIONS"] = "4"
    from driveguard.spark import get_spark

    s = get_spark("driveguard-tests")
    yield s
    s.stop()


@pytest.fixture
def settings(tmp_path):
    from driveguard.config import Settings

    s = Settings()
    s.data_root = str(tmp_path / "data")
    s.raw_path = str(tmp_path / "raw")
    s.register_model = False
    return s
