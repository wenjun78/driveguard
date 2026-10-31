"""Where the console gets its data.

``DatabricksSource`` talks to the workspace the pipeline runs in: Unity Catalog tables through a SQL warehouse,
the registered model through MLflow, job runs through the Jobs API and the champion through Model Serving.
``SnapshotSource`` serves the same shapes from JSON files exported by ``scripts/export_snapshot.py`` - the
console works, with a visible "demo snapshot" badge, before the workspace exists, when the warehouse is asleep,
or when the Free Edition quota for the day is gone.

Both expose the same methods; the API layer never knows which one it is talking to.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from driveguard.config import TABLES, Settings
from driveguard.model import DriveGuardModel

log = logging.getLogger(__name__)

HISTORY_COLUMNS = [
    "smart_5_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw",
    "smart_9_raw",
    "smart_194_raw",
]


def _d(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, (date, datetime)):
        return v.isoformat()[:10]
    return str(v)[:10]


def _list(v) -> list:
    """ARRAY columns arrive as Python lists locally but as numpy arrays from the Databricks SQL connector."""
    if v is None:
        return []
    if hasattr(v, "tolist"):
        return [str(x) for x in v.tolist()]
    return [str(x) for x in v]


def _aliases(mv) -> list[str]:
    """ModelVersion.aliases is a property in some MLflow versions and a method in others; UC search may omit it."""
    v = getattr(mv, "aliases", None)
    if callable(v):
        try:
            v = v()
        except Exception:
            v = None
    try:
        return [str(x) for x in (v or [])]
    except TypeError:
        return []


def _f(v) -> float | None:
    try:
        if v is None or (isinstance(v, float) and v != v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class Outcome:
    """What hindsight says about an alert, once the horizon has elapsed."""

    label: int | None
    days_to_failure: float | None

    @property
    def text(self) -> str:
        if self.label is None:
            return "pending"
        return "failed" if self.label == 1 else "healthy"


# ---------------------------------------------------------------------------------------------------------------
# Snapshot (demo) source
# ---------------------------------------------------------------------------------------------------------------
class SnapshotSource:
    kind = "snapshot"

    def __init__(self, directory: str):
        self.dir = directory
        self._cache: dict[str, Any] = {}
        self.model = (
            DriveGuardModel.load(os.path.join(directory, "model_dir"))
            if os.path.isdir(os.path.join(directory, "model_dir"))
            else None
        )

    def _load(self, name: str):
        if name not in self._cache:
            with open(os.path.join(self.dir, f"{name}.json")) as fh:
                self._cache[name] = json.load(fh)
        return self._cache[name]

    def status(self) -> dict:
        ov = self._load("overview")
        return {
            "source": self.kind,
            "as_of": ov["as_of"],
            "note": ov.get("note", "demo snapshot"),
            "generated_at": ov.get("generated_at"),
        }

    def overview(self) -> dict:
        return self._load("overview")

    def alert_dates(self) -> list[str]:
        return sorted(self._load("alerts").keys())

    def alerts(self, day: str | None) -> dict:
        all_days = self._load("alerts")
        day = day or max(all_days)
        ov = self._load("overview")
        matured = date.fromisoformat(day) + timedelta(days=int(ov.get("horizon_days", 30))) <= date.fromisoformat(
            ov["as_of"]
        )
        return {"date": day, "rows": all_days.get(day, []), "dates": sorted(all_days), "matured": matured}

    def drive(self, serial: str) -> dict | None:
        return self._load("drives").get(serial)

    def monitoring(self) -> dict:
        return self._load("monitoring")

    def model_info(self) -> dict:
        return self._load("model")

    def pipeline(self) -> dict:
        return self._load("pipeline")

    def predict(self, features: pd.DataFrame) -> dict:
        if self.model is None:
            raise RuntimeError("snapshot has no model")
        p = float(self.model.predict_proba(features)[0])
        return {
            "p_fail_within_horizon": p,
            "reason_codes": self.model.reason_codes(features, top=4)[0],
            "served_by": "local copy of the snapshot champion",
        }

    def feature_names(self) -> list[str]:
        return list(self.model.spec.numeric) if self.model else []

    def latest_features(self, serial: str) -> pd.DataFrame | None:
        d = self.drive(serial)
        if not d or not d.get("latest_features"):
            return None
        return pd.DataFrame([d["latest_features"]])


# ---------------------------------------------------------------------------------------------------------------
# Databricks (live) source
# ---------------------------------------------------------------------------------------------------------------
class DatabricksSource:
    kind = "databricks"

    def __init__(
        self,
        settings: Settings,
        host: str,
        token: str,
        http_path: str,
        daily_job_name: str = "driveguard - daily",
        build_job_name: str = "driveguard - build",
        endpoint: str = "driveguard-failure-30d",
    ):
        self.s = settings
        self.host = host.rstrip("/")
        self.token = token
        self.http_path = http_path
        self.daily_job_name = daily_job_name
        self.build_job_name = build_job_name
        self.endpoint = endpoint
        self._model: DriveGuardModel | None = None
        self._model_version: str | None = None
        os.environ.setdefault("DATABRICKS_HOST", self.host)
        os.environ.setdefault("DATABRICKS_TOKEN", self.token)

    # ---- plumbing ----------------------------------------------------------------------------------------------
    def t(self, name: str) -> str:
        return self.s.table(TABLES[name])

    def _connect(self):
        from databricks import sql as dbsql

        host = self.host.replace("https://", "").replace("http://", "")
        return dbsql.connect(server_hostname=host, http_path=self.http_path, access_token=self.token)

    def query(self, sql: str, params: dict | None = None) -> list[dict]:
        """One warehouse connection per source, reused across queries; reconnects once on failure."""
        for attempt in (1, 2):
            try:
                conn = getattr(self, "_conn", None)
                if conn is None:
                    conn = self._conn = self._connect()
                with conn.cursor() as cur:
                    cur.execute(sql, params or {})
                    cols = [c[0] for c in cur.description]
                    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
            except Exception:
                try:
                    if getattr(self, "_conn", None) is not None:
                        self._conn.close()
                except Exception:
                    pass
                self._conn = None
                if attempt == 2:
                    raise
        return []  # unreachable

    def _ws(self):
        from databricks.sdk import WorkspaceClient

        return WorkspaceClient(host=self.host, token=self.token)

    def _mlflow_client(self):
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        return MlflowClient()

    def _champion(self):
        client = self._mlflow_client()
        name = self.s.registered_model_name()
        mv = client.get_model_version_by_alias(name, self.s.champion_alias)
        return client, name, mv

    def _ensure_model(self) -> DriveGuardModel:
        import mlflow

        _, name, mv = self._champion()
        if self._model is None or self._model_version != mv.version:
            local = mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{name}/{mv.version}/artifacts/model_dir", dst_path=tempfile.mkdtemp()
            )
            self._model = DriveGuardModel.load(local)
            self._model_version = mv.version
        return self._model

    def now(self) -> date:
        """The data day that counts as today (replay cursor or latest scored day); cached for two minutes."""
        import time

        from driveguard import replay

        cached = getattr(self, "_now_cache", None)
        if cached and time.time() - cached[0] < 120:
            return cached[1]
        latest = self.query(f"SELECT MAX(date) AS d FROM {self.t('alerts')}")[0]["d"]
        if latest is None:
            latest = self.query(f"SELECT MAX(date) AS d FROM {self.t('gold')}")[0]["d"]
        value = replay.resolve_as_of(None, latest)
        self._now_cache = (time.time(), value)
        return value

    # ---- API shapes --------------------------------------------------------------------------------------------
    def status(self) -> dict:
        return {
            "source": self.kind,
            "as_of": _d(self.now()),
            "workspace": self.host.split("//")[-1],
            "catalog": self.s.catalog,
            "schema": self.s.schema,
            "replay_anchor": os.environ.get("DG_REPLAY_ANCHOR") or None,
        }

    def overview(self) -> dict:
        now = self.now()
        a, g, lc, mon = self.t("alerts"), self.t("gold"), self.t("lifecycle"), self.t("monitor")
        today = self.query(
            f"SELECT COUNT(*) AS drives, SUM(flagged) AS alerts, MAX(alert_budget) AS budget, MAX(horizon_days) AS horizon FROM {a} WHERE date = :d",
            {"d": now},
        )[0]
        per_day = self.query(
            f"""SELECT a.date, COUNT(*) AS drives, SUM(a.flagged) AS alerts,
                       SUM(CASE WHEN a.flagged = 1 AND g.label = 1 THEN 1 ELSE 0 END) AS hits,
                       SUM(CASE WHEN a.flagged = 1 AND g.label = 0 THEN 1 ELSE 0 END) AS false_alarms
                FROM {a} a LEFT JOIN {g} g ON a.serial_number = g.serial_number AND a.date = g.date
                WHERE a.date <= :d GROUP BY a.date ORDER BY a.date DESC LIMIT 60""",
            {"d": now},
        )
        horizon = int(today["horizon"] or self.s.horizon_days)

        def _day(r: dict) -> dict:
            d = r["date"] if isinstance(r["date"], date) else date.fromisoformat(str(r["date"])[:10])
            matured = d + timedelta(days=horizon) <= now
            alerts_n = int(r["alerts"] or 0)
            hits = int(r["hits"] or 0) if matured else 0
            fa = int(r["false_alarms"] or 0) if matured else 0
            return {
                "date": _d(d),
                "drives": int(r["drives"]),
                "alerts": alerts_n,
                "hits": hits,
                "false_alarms": fa,
                "pending": alerts_n - hits - fa,
                "matured": matured,
            }

        by_model = self.query(
            f"""SELECT g.model, COUNT(*) AS drives, SUM(a.flagged) AS alerts
                FROM {a} a JOIN {g} g ON a.serial_number = g.serial_number AND a.date = g.date
                WHERE a.date = :d GROUP BY g.model ORDER BY drives DESC LIMIT 12""",
            {"d": now},
        )
        failures = self.query(
            f"SELECT COUNT(*) AS n FROM {lc} WHERE failure_date > DATE_SUB(:d, 30) AND failure_date <= :d",
            {"d": now},
        )[0]["n"]
        fleet = self.query(f"SELECT COUNT(*) AS drives, SUM(failed) AS failed FROM {lc}")[0]
        matured: list[dict] = []
        try:
            row = self.query(f"SELECT matured_all FROM {mon} ORDER BY monitored_at DESC LIMIT 1")
            if row and row[0]["matured_all"]:
                matured = json.loads(row[0]["matured_all"])
        except Exception as exc:  # column may not exist on an older monitoring table
            log.warning("matured_all unavailable: %s", exc)
        return {
            "as_of": _d(now),
            "drives_scored": int(today["drives"] or 0),
            "alerts": int(today["alerts"] or 0),
            "alert_budget": _f(today["budget"]),
            "horizon_days": int(today["horizon"] or self.s.horizon_days),
            "failures_last_30d": int(failures or 0),
            "fleet_drives_ever": int(fleet["drives"] or 0),
            "fleet_failed_ever": int(fleet["failed"] or 0),
            "alerts_per_day": [_day(r) for r in reversed(per_day)],
            "fleet_by_model": [
                {"model": r["model"], "drives": int(r["drives"]), "alerts": int(r["alerts"] or 0)} for r in by_model
            ],
            "matured": matured,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def alert_dates(self) -> list[str]:
        now = self.now()
        rows = self.query(
            f"SELECT DISTINCT date FROM {self.t('alerts')} WHERE date <= :d ORDER BY date DESC LIMIT 90", {"d": now}
        )
        return sorted(_d(r["date"]) for r in rows)

    def alerts(self, day: str | None) -> dict:
        dates = self.alert_dates()
        day = day or (dates[-1] if dates else None)
        if day is None:
            return {"date": None, "rows": [], "dates": []}
        now = self.now()
        horizon = self.s.horizon_days
        matured = date.fromisoformat(day) + timedelta(days=horizon) <= now
        a, g = self.t("alerts"), self.t("gold")
        rows = self.query(
            f"""SELECT a.rank_in_day, a.serial_number, a.model, a.capacity_tb, a.score, a.reason_codes, g.label, g.days_to_failure
                FROM {a} a LEFT JOIN {g} g ON a.serial_number = g.serial_number AND a.date = g.date
                WHERE a.date = :d AND a.flagged = 1 ORDER BY a.rank_in_day""",
            {"d": date.fromisoformat(day)},
        )
        out = []
        for r in rows:
            label = r["label"] if matured else None
            out.append(
                {
                    "rank": int(r["rank_in_day"]),
                    "serial": r["serial_number"],
                    "model": r["model"],
                    "capacity_tb": _f(r["capacity_tb"]),
                    "score": _f(r["score"]),
                    "reason_codes": _list(r["reason_codes"]),
                    "outcome": Outcome(
                        None if label is None else int(label), _f(r["days_to_failure"]) if matured else None
                    ).text,
                    "days_to_failure": _f(r["days_to_failure"]) if matured and r["label"] == 1 else None,
                }
            )
        return {"date": day, "rows": out, "dates": dates, "matured": bool(matured)}

    def drive(self, serial: str) -> dict | None:
        now = self.now()
        g, lc, a = self.t("gold"), self.t("lifecycle"), self.t("alerts")
        life = self.query(f"SELECT * FROM {lc} WHERE serial_number = :s", {"s": serial})
        if not life:
            return None
        life = life[0]
        cols = ", ".join(f"g.{c}" for c in HISTORY_COLUMNS)
        hist = self.query(
            f"""SELECT g.date, {cols}, a.score, a.flagged, a.rank_in_day
                FROM {g} g LEFT JOIN {a} a ON a.serial_number = g.serial_number AND a.date = g.date
                WHERE g.serial_number = :s AND g.date <= :d ORDER BY g.date DESC LIMIT 120""",
            {"s": serial, "d": now},
        )
        failed_visible = life["failure_date"] is not None and life["failure_date"] <= now
        return {
            "serial": serial,
            "model": life["model"],
            "capacity_tb": _f((life["capacity_bytes"] or 0) / 1e12),
            "first_seen": _d(life["first_seen"]),
            "last_seen": _d(min(life["last_seen"], now)),
            "failed": bool(failed_visible),
            "failure_date": _d(life["failure_date"]) if failed_visible else None,
            "history": [
                {
                    "date": _d(r["date"]),
                    **{c: _f(r[c]) for c in HISTORY_COLUMNS},
                    "score": _f(r["score"]),
                    "flagged": int(r["flagged"] or 0),
                }
                for r in reversed(hist)
            ],
        }

    def drives_bulk(self, serials: list[str], history_days: int = 120) -> dict[str, dict]:
        """Lifecycle + recent history for many drives in a handful of queries (used by the snapshot export)."""
        now = self.now()
        g, lc, a = self.t("gold"), self.t("lifecycle"), self.t("alerts")
        out: dict[str, dict] = {}
        cols = ", ".join(f"g.{c}" for c in HISTORY_COLUMNS)
        for i in range(0, len(serials), 150):
            chunk = [re.sub(r"[^A-Za-z0-9_\-]", "", x) for x in serials[i : i + 150]]
            in_list = ", ".join(f"'{x}'" for x in chunk)
            life = {r["serial_number"]: r for r in self.query(f"SELECT * FROM {lc} WHERE serial_number IN ({in_list})")}
            hist = self.query(
                f"""SELECT * FROM (
                      SELECT g.serial_number, g.date, {cols}, a.score, a.flagged,
                             ROW_NUMBER() OVER (PARTITION BY g.serial_number ORDER BY g.date DESC) AS rn
                      FROM {g} g LEFT JOIN {a} a ON a.serial_number = g.serial_number AND a.date = g.date
                      WHERE g.serial_number IN ({in_list}) AND g.date <= :d)
                    WHERE rn <= {int(history_days)} ORDER BY serial_number, date""",
                {"d": now},
            )
            by_serial: dict[str, list] = {}
            for r in hist:
                by_serial.setdefault(r["serial_number"], []).append(r)
            for sn in chunk:
                lf = life.get(sn)
                if lf is None:
                    continue
                failed_visible = lf["failure_date"] is not None and lf["failure_date"] <= now
                out[sn] = {
                    "serial": sn,
                    "model": lf["model"],
                    "capacity_tb": _f((lf["capacity_bytes"] or 0) / 1e12),
                    "first_seen": _d(lf["first_seen"]),
                    "last_seen": _d(min(lf["last_seen"], now)),
                    "failed": bool(failed_visible),
                    "failure_date": _d(lf["failure_date"]) if failed_visible else None,
                    "history": [
                        {
                            "date": _d(r["date"]),
                            **{c: _f(r[c]) for c in HISTORY_COLUMNS},
                            "score": _f(r["score"]),
                            "flagged": int(r["flagged"] or 0),
                        }
                        for r in by_serial.get(sn, [])
                    ],
                }
        return out

    def latest_features_bulk(self, serials: list[str]) -> dict[str, dict]:
        now = self.now()
        out: dict[str, dict] = {}
        for i in range(0, len(serials), 150):
            chunk = [re.sub(r"[^A-Za-z0-9_\-]", "", x) for x in serials[i : i + 150]]
            in_list = ", ".join(f"'{x}'" for x in chunk)
            rows = self.query(
                f"""SELECT * FROM (SELECT g.*, ROW_NUMBER() OVER (PARTITION BY serial_number ORDER BY date DESC) AS rn
                                   FROM {self.t("gold")} g WHERE serial_number IN ({in_list}) AND date <= :d) WHERE rn = 1""",
                {"d": now},
            )
            for r in rows:
                r.pop("rn", None)
                out[r["serial_number"]] = r
        return out

    def latest_features(self, serial: str) -> pd.DataFrame | None:
        now = self.now()
        rows = self.query(
            f"SELECT * FROM {self.t('gold')} WHERE serial_number = :s AND date <= :d ORDER BY date DESC LIMIT 1",
            {"s": serial, "d": now},
        )
        return pd.DataFrame(rows) if rows else None

    def monitoring(self) -> dict:
        rows = self.query(f"SELECT * FROM {self.t('monitor')} ORDER BY monitored_at DESC LIMIT 60")
        runs = []
        for r in rows:
            runs.append(
                {
                    "monitored_at": r["monitored_at"],
                    "snapshot_date": r["snapshot_date"],
                    "model_uri": r["model_uri"],
                    "drives_scored": int(r["drives_scored"] or 0),
                    "psi_by_feature": json.loads(r["psi_by_feature"] or "{}"),
                    "features_drifted": json.loads(r["features_drifted"] or "[]"),
                    "features_warned": json.loads(r["features_warned"] or "[]"),
                    "score_psi": _f(r["score_psi"]),
                    "matured_snapshots": int(r["matured_snapshots"] or 0),
                    "matured_recent_recall_at_budget": _f(r["matured_recent_recall_at_budget"]),
                    "retrain_recommended": bool(r["retrain_recommended"]),
                    "retrain_reasons": json.loads(r["retrain_reasons"] or "[]"),
                }
            )
        matured = json.loads(rows[0]["matured_all"]) if rows and rows[0].get("matured_all") else []
        return {"runs": list(reversed(runs)), "matured": matured}

    def model_info(self) -> dict:
        import mlflow

        client, name, champ = self._champion()
        run = client.get_run(champ.run_id)
        report: dict = {}
        importance: list = []
        with tempfile.TemporaryDirectory() as tmp:
            try:
                p = mlflow.artifacts.download_artifacts(
                    run_id=champ.run_id, artifact_path="metrics_report.json", dst_path=tmp
                )
                report = json.load(open(p))
                p = mlflow.artifacts.download_artifacts(
                    run_id=champ.run_id, artifact_path="feature_importance.json", dst_path=tmp
                )
                importance = json.load(open(p))
            except Exception as exc:
                log.warning("artifacts unavailable: %s", exc)
        versions = []
        for mv in client.search_model_versions(f"name = '{name}'"):
            try:
                m = client.get_run(mv.run_id).data.metrics
            except Exception:
                m = {}
            versions.append(
                {
                    "version": int(mv.version),
                    "created": datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=timezone.utc).isoformat(),
                    "val_pr_auc": _f(m.get("val_pr_auc")),
                    "test_pr_auc": _f(m.get("test_pr_auc")),
                    "aliases": _aliases(mv) or (["champion"] if str(mv.version) == str(champ.version) else []),
                    "run_id": mv.run_id,
                }
            )
        versions.sort(key=lambda v: -v["version"])
        params = {k.replace("split_", ""): v for k, v in run.data.params.items() if k.startswith("split_")}
        return {
            "name": name,
            "champion": {
                "version": int(champ.version),
                "run_id": champ.run_id,
                "created": datetime.fromtimestamp(champ.creation_timestamp / 1000, tz=timezone.utc).isoformat(),
                "promotion": run.data.tags.get("promotion"),
            },
            "versions": versions,
            "report": report,
            "importance": importance[:25],
            "split": params,
            "horizon_days": int(run.data.params.get("horizon_days", self.s.horizon_days)),
            "alert_budget": _f(run.data.params.get("alert_budget", self.s.alert_budget)),
            "experiment": self.s.experiment,
        }

    def pipeline(self) -> dict:
        w = self._ws()
        jobs_out = []
        all_jobs = [j for j in w.jobs.list() if j.settings and j.settings.name]
        for job_name in (self.daily_job_name, self.build_job_name):
            # bundle targets in development mode prefix names with "[dev user]"; match on the suffix
            for job in [j for j in all_jobs if j.settings.name == job_name or j.settings.name.endswith(job_name)]:
                runs = []
                for r in w.jobs.list_runs(job_id=job.job_id, limit=8):
                    st = r.state
                    runs.append(
                        {
                            "run_id": r.run_id,
                            "state": (st.life_cycle_state.value if st and st.life_cycle_state else None),
                            "result": (st.result_state.value if st and st.result_state else None),
                            "start": datetime.fromtimestamp((r.start_time or 0) / 1000, tz=timezone.utc).isoformat()
                            if r.start_time
                            else None,
                            "duration_s": (r.run_duration or 0) / 1000 if r.run_duration else None,
                            "url": r.run_page_url,
                        }
                    )
                sched = job.settings.schedule if job.settings else None
                jobs_out.append(
                    {
                        "name": job_name,
                        "job_id": job.job_id,
                        "schedule": sched.quartz_cron_expression if sched else None,
                        "paused": (sched.pause_status.value if sched and sched.pause_status else None),
                        "runs": runs,
                    }
                )
        serving = None
        try:
            ep = w.serving_endpoints.get(self.endpoint)
            serving = {
                "name": self.endpoint,
                "state": ep.state.ready.value if ep.state and ep.state.ready else None,
                "config_update": ep.state.config_update.value if ep.state and ep.state.config_update else None,
            }
        except Exception as exc:
            serving = {"name": self.endpoint, "state": None, "error": str(exc)[:120]}
        from driveguard import replay

        return {
            "jobs": jobs_out,
            "serving": serving,
            "replay": {"anchor": os.environ.get("DG_REPLAY_ANCHOR") or None, "cursor": _d(replay.cursor())},
            "workspace": self.host,
        }

    def run_daily_job(self) -> dict:
        w = self._ws()
        for job in w.jobs.list():
            if job.settings and job.settings.name and job.settings.name.endswith(self.daily_job_name):
                run = w.jobs.run_now(job_id=job.job_id)
                return {"job_id": job.job_id, "run_id": run.run_id}
        raise RuntimeError(f"job {self.daily_job_name!r} not found")

    def feature_names(self) -> list[str]:
        return list(self._ensure_model().spec.numeric)

    def predict(self, features: pd.DataFrame) -> dict:
        model = self._ensure_model()
        reasons = model.reason_codes(features, top=4)[0]
        local_p = float(model.predict_proba(features)[0])
        served_by = f"local copy of champion v{self._model_version} (Unity Catalog)"
        p = local_p
        try:
            w = self._ws()
            cols = model.spec.numeric + ["model"]
            payload = {
                "dataframe_split": {
                    "columns": cols,
                    "data": features[cols].astype(object).where(features[cols].notna(), None).values.tolist(),
                }
            }
            resp = w.api_client.do("POST", f"/serving-endpoints/{self.endpoint}/invocations", body=payload)
            preds = resp.get("predictions") if isinstance(resp, dict) else None
            if preds:
                p = float(preds[0] if not isinstance(preds[0], dict) else list(preds[0].values())[0])
                served_by = f"Model Serving endpoint {self.endpoint} (champion v{self._model_version})"
        except Exception as exc:
            log.info("serving endpoint unavailable, using local champion: %s", exc)
        return {"p_fail_within_horizon": p, "reason_codes": reasons, "served_by": served_by, "local_p": local_p}


def healthy_baseline(
    numeric: list[str], power_on_hours: float = 20000.0, temperature: float = 30.0, capacity_tb: float = 12.0
) -> dict[str, float]:
    """A plausible healthy drive: error counters and their deltas at zero, wear/workload counters at ordinary values,
    normalized attributes at 100, age/thermal/capacity as given.

    Used for hand-entered what-if scoring so that unspecified features mean "nothing wrong" rather than "unknown"
    (NaN) or an implausible zero (a drive with zero load cycles or a normalized value of 0 looks odd to the model).
    """
    base: dict[str, float] = {c: 0.0 for c in numeric}
    days = power_on_hours / 24.0
    workload = {  # typical magnitudes for a drive of this age; deltas scale with the window length
        "smart_4_raw": 20.0,
        "smart_12_raw": 20.0,
        "smart_192_raw": 10.0,
        "smart_193_raw": max(100.0, 4.0 * days),
        "smart_240_raw": 0.9 * power_on_hours,
        "smart_241_raw": 2.0e7 * days,
        "smart_242_raw": 3.0e7 * days,
    }
    for c in numeric:
        if c.endswith("_normalized"):
            base[c] = 100.0 - temperature if c.startswith("smart_194_") else 100.0
        elif c.startswith("smart_194_"):
            base[c] = temperature
        elif c == "smart_9_raw":
            base[c] = power_on_hours
        elif c == "smart_9_delta_7d":
            base[c] = 24.0 * 7
        elif c == "smart_9_delta_30d":
            base[c] = 24.0 * 30
        elif c in workload:
            base[c] = workload[c]
        elif c.startswith(("smart_193_delta_", "smart_240_delta_", "smart_241_delta_", "smart_242_delta_")):
            attr = c.split("_delta_")[0] + "_raw"
            window = 7 if c.endswith("_7d") else 30
            base[c] = workload[attr] / max(1.0, days) * window
        elif c == "capacity_tb":
            base[c] = capacity_tb
        elif c == "gap_since_prev_obs":
            base[c] = 1.0
    return base
