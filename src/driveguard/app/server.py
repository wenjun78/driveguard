"""DriveGuard fleet console - public, read-only API + static UI.

Live mode needs four environment variables (a Databricks Free Edition workspace is enough):

    DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
    DATABRICKS_TOKEN=<personal access token, read + jobs.run if the owner trigger is wanted>
    DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/<warehouse id>
    DG_CATALOG=workspace  DG_SCHEMA=driveguard

Without them (or when the workspace is unreachable) the console serves the bundled snapshot and says so.
Every read is cached for ``DG_APP_CACHE_TTL`` seconds (default 10 min) and refreshed in the background, so a
public page never queues requests on the SQL warehouse and a cold warehouse never blocks a visitor.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Annotated, Any

import pandas as pd
from fastapi import Body, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from driveguard import __version__
from driveguard.app.sources import DatabricksSource, SnapshotSource, healthy_baseline
from driveguard.config import Settings

log = logging.getLogger("driveguard.app")


def _find_root() -> str:
    """Repo root holding ``web/`` and ``app_snapshot/``: explicit env, the working directory (Render, Docker), or the
    checkout this file lives in (editable installs)."""
    candidates = [
        os.environ.get("DG_APP_ROOT"),
        os.getcwd(),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "web", "index.html")):
            return c
    return candidates[-1]


ROOT = _find_root()
WEB_DIR = os.environ.get("DG_APP_WEB_DIR", os.path.join(ROOT, "web"))
SNAPSHOT_DIR = os.environ.get("DG_APP_SNAPSHOT_DIR", os.path.join(ROOT, "app_snapshot"))
TTL = int(os.environ.get("DG_APP_CACHE_TTL", "600"))
# The daily workflow force-pushes a fresh snapshot to the `snapshot` branch (single commit, so the repo never grows);
# the console pulls it at start-up and every few hours. Empty string disables; the bundled app_snapshot/ is the floor.
SNAPSHOT_URL = os.environ.get(
    "DG_APP_SNAPSHOT_URL", "https://raw.githubusercontent.com/wenjun78/driveguard/snapshot/app_snapshot"
)
SNAPSHOT_FILES = [
    "meta.json",
    "overview.json",
    "alerts.json",
    "drives.json",
    "monitoring.json",
    "model.json",
    "pipeline.json",
    "model_dir/booster.ubj",
    "model_dir/feature_spec.json",
]


def fetch_remote_snapshot(url: str, dest: str, timeout: int = 30) -> bool:
    """Download the published snapshot into ``dest``; True on success (all files), False otherwise."""
    import shutil
    import tempfile
    import urllib.request

    url = url.rstrip("/")
    tmp = tempfile.mkdtemp(prefix="dg-snapshot-")
    try:
        for name in SNAPSHOT_FILES:
            target = os.path.join(tmp, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with urllib.request.urlopen(f"{url}/{name}", timeout=timeout) as r, open(target, "wb") as fh:
                shutil.copyfileobj(r, fh)
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(tmp, dest)
        return True
    except Exception as exc:
        log.info("remote snapshot unavailable (%s): %s", url, str(exc)[:120])
        shutil.rmtree(tmp, ignore_errors=True)
        return False


class Cache:
    """TTL cache with stale-while-revalidate: a stale value is served immediately and refreshed in a thread."""

    def __init__(self, ttl: int):
        self.ttl = ttl
        self._v: dict[str, tuple[float, Any]] = {}
        self._inflight: set[str] = set()
        self._lock = threading.Lock()

    def get(self, key: str, fn: Callable[[], Any]) -> Any:
        now = time.time()
        with self._lock:
            hit = self._v.get(key)
        if hit and now - hit[0] < self.ttl:
            return hit[1]
        if hit:  # stale: refresh in the background, serve what we have
            self._refresh_async(key, fn)
            return hit[1]
        value = fn()
        with self._lock:
            self._v[key] = (time.time(), value)
        return value

    def _refresh_async(self, key: str, fn: Callable[[], Any]) -> None:
        with self._lock:
            if key in self._inflight:
                return
            self._inflight.add(key)

        def run():
            try:
                value = fn()
                with self._lock:
                    self._v[key] = (time.time(), value)
            except Exception as exc:
                log.warning("background refresh of %s failed: %s", key, exc)
            finally:
                with self._lock:
                    self._inflight.discard(key)

        threading.Thread(target=run, daemon=True).start()

    def age(self, key: str) -> float | None:
        hit = self._v.get(key)
        return None if not hit else time.time() - hit[0]

    def clear(self) -> None:
        with self._lock:
            self._v.clear()


def _snapshot_source() -> SnapshotSource | None:
    """Prefer the published snapshot (real, recent); fall back to the one bundled in the repo."""
    if SNAPSHOT_URL:
        remote_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "driveguard_snapshot")
        if fetch_remote_snapshot(SNAPSHOT_URL, remote_dir):
            log.info("using published snapshot from %s", SNAPSHOT_URL)
            return SnapshotSource(remote_dir)
    return SnapshotSource(SNAPSHOT_DIR) if os.path.isdir(SNAPSHOT_DIR) else None


def build_sources() -> tuple[Any, Any | None]:
    """(primary, fallback). Primary is Databricks when configured, else the snapshot."""
    snapshot = _snapshot_source()
    host, token, http_path = (
        os.environ.get("DATABRICKS_HOST"),
        os.environ.get("DATABRICKS_TOKEN"),
        os.environ.get("DATABRICKS_HTTP_PATH"),
    )
    if host and token and http_path:
        settings = Settings.from_env()
        if not settings.catalog:
            settings.catalog = os.environ.get("DG_CATALOG", "workspace")
        live = DatabricksSource(
            settings, host, token, http_path, endpoint=os.environ.get("DG_SERVING_ENDPOINT", "driveguard-failure-30d")
        )
        return live, snapshot
    if snapshot is None:
        raise RuntimeError("neither Databricks credentials nor a snapshot directory are available")
    return snapshot, None


app = FastAPI(title="DriveGuard console", version=__version__, docs_url="/api/docs", redoc_url=None)
cache = Cache(TTL)
PRIMARY, FALLBACK = build_sources()
_state: dict[str, Any] = {"live_ok": PRIMARY.kind == "databricks", "last_error": None, "degraded_since": None}


def _call(name: str, fn_name: str, *args, key: str | None = None) -> tuple[Any, str]:
    """Run ``fn_name`` on the primary source through the cache; fall back to the snapshot on failure."""
    key = key or f"{fn_name}:{args}"

    def primary():
        try:
            out = getattr(PRIMARY, fn_name)(*args)
            _state.update(live_ok=PRIMARY.kind == "databricks", last_error=None, degraded_since=None)
            return (out, PRIMARY.kind)
        except Exception as exc:
            if FALLBACK is None:
                raise
            log.warning("%s failed on %s, using snapshot: %s", name, PRIMARY.kind, exc)
            _state.update(
                live_ok=False,
                last_error=f"{type(exc).__name__}: {str(exc)[:160]}",
                degraded_since=_state["degraded_since"] or time.time(),
            )
            return (getattr(FALLBACK, fn_name)(*args), FALLBACK.kind)

    return cache.get(key, primary)


def _envelope(data: Any, source: str, key: str) -> JSONResponse:
    return JSONResponse(
        {"data": data, "source": source, "cache_age_s": round(cache.age(key) or 0), "live_ok": _state["live_ok"]}
    )


# ---- API ------------------------------------------------------------------------------------------------------
@app.get("/api/status")
def status():
    data, source = _call("status", "status", key="status")
    return _envelope(
        {
            **data,
            "primary": PRIMARY.kind,
            "fallback": FALLBACK.kind if FALLBACK else None,
            "last_error": _state["last_error"],
            "cache_ttl_s": TTL,
            "version": __version__,
        },
        source,
        "status",
    )


@app.get("/api/overview")
def overview():
    data, source = _call("overview", "overview", key="overview")
    return _envelope(data, source, "overview")


@app.get("/api/alerts")
def alerts(date: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
    if date:
        try:
            dt.date.fromisoformat(date)
        except ValueError as exc:
            raise HTTPException(422, f"invalid date {date!r}") from exc
    key = f"alerts:{date}"
    data, source = _call("alerts", "alerts", date, key=key)
    return _envelope(data, source, key)


@app.get("/api/drive/{serial}")
def drive(serial: str):
    serial = serial.strip()[:64]
    key = f"drive:{serial}"
    data, source = _call("drive", "drive", serial, key=key)
    if data is None:
        raise HTTPException(404, f"drive {serial!r} not found")
    return _envelope(data, source, key)


@app.get("/api/monitoring")
def monitoring():
    data, source = _call("monitoring", "monitoring", key="monitoring")
    return _envelope(data, source, "monitoring")


@app.get("/api/model")
def model():
    data, source = _call("model", "model_info", key="model")
    return _envelope(data, source, "model")


@app.get("/api/pipeline")
def pipeline():
    data, source = _call("pipeline", "pipeline", key="pipeline")
    return _envelope(data, source, "pipeline")


@app.post("/api/predict")
def predict(body: Annotated[dict, Body()]):
    """Score one drive: ``{"serial": "..."}`` uses its latest features; ``{"features": {...}, "model": "..."}`` scores
    hand-entered SMART values (anything omitted is treated as a healthy zero)."""
    serial = (body.get("serial") or "").strip()
    src = PRIMARY
    try:
        if serial:
            feats = PRIMARY.latest_features(serial)
            if feats is None and FALLBACK is not None:
                feats, src = FALLBACK.latest_features(serial), FALLBACK
            if feats is None:
                raise HTTPException(404, f"no features for drive {serial!r}")
        else:
            given = {k: float(v) for k, v in dict(body.get("features") or {}).items() if v not in (None, "")}
            names = src.feature_names()
            base = healthy_baseline(
                names,
                power_on_hours=given.get("smart_9_raw", 20000.0),
                temperature=given.get("smart_194_raw", 30.0),
                capacity_tb=given.get("capacity_tb", 12.0),
            )
            base.update({k: v for k, v in given.items() if k in base})
            base["model"] = body.get("model") or "UNKNOWN"
            feats = pd.DataFrame([base])
        try:
            out = src.predict(feats)
            source = src.kind
        except Exception as exc:
            if FALLBACK is None or src is FALLBACK:
                raise
            log.warning("predict failed on %s, using snapshot model: %s", src.kind, exc)
            out, source = FALLBACK.predict(feats), FALLBACK.kind
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"prediction unavailable: {type(exc).__name__}: {str(exc)[:160]}") from exc
    return JSONResponse(
        {
            "data": {
                **out,
                "serial": serial or None,
                "inputs": None
                if serial
                else {k: v for k, v in feats.iloc[0].items() if k == "model" or (isinstance(v, float) and v != 0.0)},
            },
            "source": source,
            "live_ok": _state["live_ok"],
        }
    )


@app.post("/api/admin/run-daily")
def run_daily(x_admin_token: str | None = Header(default=None)):
    expected = os.environ.get("DG_APP_ADMIN_TOKEN")
    if not expected or x_admin_token != expected:
        raise HTTPException(403, "admin token required")
    if PRIMARY.kind != "databricks":
        raise HTTPException(409, "no Databricks workspace configured")
    try:
        out = PRIMARY.run_daily_job()
    except Exception as exc:
        raise HTTPException(502, f"could not start job: {str(exc)[:160]}") from exc
    cache.clear()
    return JSONResponse({"data": out})


@app.post("/api/admin/refresh")
def refresh(x_admin_token: str | None = Header(default=None)):
    expected = os.environ.get("DG_APP_ADMIN_TOKEN")
    if not expected or x_admin_token != expected:
        raise HTTPException(403, "admin token required")
    cache.clear()
    return JSONResponse({"data": {"cleared": True}})


@app.get("/healthz")
def healthz():
    return {"ok": True, "primary": PRIMARY.kind, "live_ok": _state["live_ok"]}


# ---- UI -------------------------------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def _warm():
    for name in ("status", "overview", "alerts", "monitoring", "model", "pipeline"):
        try:
            if name == "alerts":
                _call("alerts", "alerts", None, key="alerts:None")
            elif name == "model":
                _call("model", "model_info", key="model")
            else:
                _call(name, name, key=name)
        except Exception as exc:
            log.warning("warm-up %s failed: %s", name, exc)


def _refresh_snapshot_loop():
    """Every 6 h, pick up the latest published snapshot without a redeploy."""
    global FALLBACK, PRIMARY
    while True:
        time.sleep(6 * 3600)
        try:
            fresh = _snapshot_source()
            if fresh is None:
                continue
            if PRIMARY.kind == "databricks":
                FALLBACK = fresh
            else:
                PRIMARY = fresh
            cache.clear()
        except Exception as exc:
            log.warning("snapshot refresh failed: %s", exc)


@app.on_event("startup")
def _startup():
    threading.Thread(target=_warm, daemon=True).start()
    if SNAPSHOT_URL:
        threading.Thread(target=_refresh_snapshot_loop, daemon=True).start()
