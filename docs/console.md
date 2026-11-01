# The fleet console (live app)

A public, read-only web console over the pipeline's own outputs: today's ranked alerts with reason codes and -
once each snapshot's 30 days have elapsed - what actually happened; matured recall / precision / lead time over
time; PSI drift and the retraining decisions; the registered model's versions, champion and evaluation; job runs
and the serving endpoint; and a what-if scorer that calls the champion.

It is a FastAPI service (`driveguard.app.server`) with a static single-page UI (`web/`), no build step. It has two
data sources with one interface:

| source | when | what it talks to |
|---|---|---|
| **Databricks (live)** | `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_HTTP_PATH` set | Unity Catalog tables through the SQL warehouse (`databricks-sql-connector`), the registered model through MLflow (`databricks-uc`), job runs through the Jobs API, the champion through Model Serving (falls back to a local copy of the same version for reason codes and when the endpoint is asleep) |
| **Snapshot** | credentials absent, or the workspace fails | JSON exported by `scripts/export_snapshot.py` with the champion's model files - the same shapes, produced by the same query code. The daily workflow exports it from the live workspace and force-pushes it to the single-commit `snapshot` branch; the console fetches that at start-up and every 6 h (`DG_APP_SNAPSHOT_URL`), falling back to the `app_snapshot/` bundled in the repo. (`app/local_source.py` produces the same files from a local run through DuckDB.) |

Every response carries `source` and the UI shows a status pill (live / demo snapshot) and a banner when it is
degraded, so a visitor is never shown snapshot numbers as if they were live.

## Running it

```bash
pip install -e ".[app,local]"
uvicorn driveguard.app.server:app --reload          # demo mode on http://localhost:8000

DATABRICKS_HOST=https://dbc-xxxx.cloud.databricks.com \
DATABRICKS_TOKEN=dapi... \
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/abcdef \
DG_CATALOG=workspace DG_SCHEMA=driveguard \
uvicorn driveguard.app.server:app                     # live
```

`render.yaml` deploys it on Render's free tier (`driveguard-kf09.onrender.com`); fill the three Databricks values in
the dashboard to go live, leave them blank for demo mode. A `Dockerfile` is included for anywhere else.

## Why it does not hammer the warehouse

Every read is cached for `DG_APP_CACHE_TTL` seconds (default 10 min) with stale-while-revalidate: a stale value is
served immediately and refreshed in a background thread. The six main views are warmed at start-up. On Free
Edition this keeps the console within the fair-use quota and makes a cold SQL warehouse (10-20 s to wake)
invisible to visitors. When the workspace is unreachable or over quota, the console degrades to the snapshot and
says so; `live_ok` flips back automatically on the next successful refresh.

The only writes are owner-gated: `POST /api/admin/run-daily` (triggers the daily Databricks job) and
`POST /api/admin/refresh` (clears the cache), both requiring `X-Admin-Token: $DG_APP_ADMIN_TOKEN`.

## Replay: why the fleet moves

Backblaze publishes history. To run the system as if live, `DG_REPLAY_ANCHOR="2025-01-01@2026-11-01"` (same
value as the Databricks job parameter) pins a data day to a calendar day; the daily job scores the next data day
each real day and the monitor treats that cursor as *now* - so matured outcomes catch up day by day, drift
accumulates as the fleet ages, and the retrain gate can fire for real. Nothing after the cursor is scored or
shown. Without an anchor, "today" is simply the latest day in the gold table.

## API

| route | returns |
|---|---|
| `GET /api/status` | source, as-of day, workspace / catalog, cache TTL, last error |
| `GET /api/overview` | KPIs, alerts per day split into hits / false alarms / pending, fleet by model, matured metrics |
| `GET /api/alerts?date=` | ranked list for a snapshot with reason codes and hindsight outcome |
| `GET /api/drive/{serial}` | lifecycle and 120-day history of key counters and daily scores |
| `GET /api/monitoring` | monitoring runs (PSI per feature, drifted / warned, retrain decisions) and matured series |
| `GET /api/model` | champion, versions, evaluation report vs operator rule, feature importance, split |
| `GET /api/pipeline` | job runs, serving endpoint state, replay cursor |
| `POST /api/predict` | `{"serial": ...}` or `{"model": ..., "features": {...}}` → probability + reason codes |

Interactive docs at `/api/docs`.
