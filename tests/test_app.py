"""The console API over the bundled snapshot (no Databricks needed)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def client():
    for k in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_HTTP_PATH"):
        os.environ.pop(k, None)
    os.environ["DG_APP_ROOT"] = ROOT
    os.environ["DG_APP_SNAPSHOT_URL"] = ""  # bundled snapshot only; no network in tests
    from fastapi.testclient import TestClient

    from driveguard.app import server

    return TestClient(server.app)


def test_status_is_snapshot(client):
    j = client.get("/api/status").json()
    assert j["source"] == "snapshot" and j["data"]["primary"] == "snapshot"
    assert client.get("/healthz").json()["ok"]


def test_overview_shape(client):
    d = client.get("/api/overview").json()["data"]
    assert d["drives_scored"] > 0 and d["alerts"] > 0
    assert d["alerts_per_day"] and {"date", "alerts", "hits", "false_alarms", "pending"} <= set(d["alerts_per_day"][0])
    assert d["matured"] and "recall_at_budget" in d["matured"][0]
    assert d["fleet_by_model"]


def test_alerts_and_hindsight(client):
    latest = client.get("/api/alerts").json()["data"]
    assert latest["rows"] and latest["rows"][0]["rank"] == 1
    assert latest["matured"] is False and all(r["outcome"] == "pending" for r in latest["rows"])
    first = latest["dates"][0]
    old = client.get(f"/api/alerts?date={first}").json()["data"]
    assert old["matured"] is True and {r["outcome"] for r in old["rows"]} <= {"failed", "healthy"}
    assert any(r["days_to_failure"] is not None for r in old["rows"] if r["outcome"] == "failed")
    assert client.get("/api/alerts?date=2020-13-01").status_code == 422


def test_drive_history(client):
    rows = client.get("/api/alerts").json()["data"]["rows"]
    d = client.get(f"/api/drive/{rows[0]['serial']}").json()["data"]
    assert d["history"] and {"date", "smart_5_raw", "smart_197_raw", "score"} <= set(d["history"][-1])
    assert client.get("/api/drive/does-not-exist").status_code == 404


def test_monitoring_model_pipeline(client):
    m = client.get("/api/monitoring").json()["data"]
    assert m["runs"] and "psi_by_feature" in m["runs"][-1] and isinstance(m["runs"][-1]["retrain_recommended"], bool)
    md = client.get("/api/model").json()["data"]
    assert md["champion"]["version"] >= 1 and "test" in md["report"] and md["importance"]
    p = client.get("/api/pipeline").json()["data"]
    assert "jobs" in p and "replay" in p


def test_predict_serial_and_manual(client):
    rows = client.get("/api/alerts").json()["data"]["rows"]
    r = client.post("/api/predict", json={"serial": rows[0]["serial"]}).json()["data"]
    assert 0 <= r["p_fail_within_horizon"] <= 1 and r["reason_codes"]
    healthy = client.post(
        "/api/predict", json={"model": "ST12000NM0008", "features": {"smart_9_raw": 20000, "smart_194_raw": 30}}
    ).json()["data"]
    sick = client.post(
        "/api/predict",
        json={
            "model": "ST12000NM0008",
            "features": {
                "smart_9_raw": 20000,
                "smart_194_raw": 33,
                "smart_5_raw": 150,
                "smart_5_delta_30d": 120,
                "smart_5_delta_7d": 40,
                "smart_197_raw": 60,
                "smart_197_delta_7d": 25,
                "smart_197_nzdays_30d": 20,
                "smart_197_nzdays_7d": 7,
                "smart_198_raw": 10,
                "smart_187_raw": 35,
            },
        },
    ).json()["data"]
    assert healthy["p_fail_within_horizon"] < 0.05 < sick["p_fail_within_horizon"]
    assert all("nan" not in c for c in sick["reason_codes"])


def test_admin_requires_token(client):
    assert client.post("/api/admin/run-daily").status_code == 403
    assert client.post("/api/admin/refresh", headers={"X-Admin-Token": "nope"}).status_code == 403
