# Running on Databricks (Free Edition included)

Everything below works on Databricks Free Edition: serverless compute only, CPU-only serving, a cap on concurrent job
tasks, one SQL warehouse. The bundle is written to those limits (a linear chain of tasks, one small CPU endpoint,
scale-to-zero). Nothing in the project needs a classic cluster or a GPU.

## 0. One-time setup

```bash
pip install databricks-cli build          # Databricks CLI >= 0.218 (bundles)
databricks configure                      # host + personal access token of your workspace
```

Create the raw-file Volume once (SQL editor or a notebook):

```sql
CREATE SCHEMA IF NOT EXISTS workspace.driveguard;
CREATE VOLUME IF NOT EXISTS workspace.driveguard.raw;
```

## 1. Get data into the Volume

Free Edition restricts outbound internet from compute, so download on your laptop and upload the subset:

```bash
# ~10 % of drives (whole histories, by serial hash), three quarters, SMART columns the pipeline uses only
python scripts/download_backblaze.py --quarters 2024Q3 2024Q4 2025Q1 --serial-frac 0.10 --out data/raw

# upload (parquet, a few hundred MB for this slice)
databricks fs cp -r data/raw dbfs:/Volumes/workspace/driveguard/raw
```

Start smaller than you think: one quarter at 5 % is enough to see every stage run end to end; widen afterwards. The
protocol needs more than `test_days + val_days + 2*gap + horizon` = 240 days of history with the defaults, so use at
least three quarters, or lower `DG_TEST_DAYS` / `DG_VAL_DAYS` for a first pass.

## 2. Deploy and run

```bash
databricks bundle validate
databricks bundle deploy -t dev
databricks bundle run -t dev driveguard_build      # bronze -> silver -> gold -> train -> 45-day scoring replay -> monitor
```

The build job ends with a registered model `workspace.driveguard.driveguard_failure_30d@champion`, the `gold_risk_alerts`
table with ranked alerts and reason codes for 45 snapshot days, and a first row in `ops_monitoring`.

Then unpause `driveguard - daily` (Workflows UI or set `pause_status: UNPAUSED` in `resources/jobs.yml`). Each run:
ingest new files → features → score today → monitor → **condition task** → retrain only when
`retrain_recommended == true`.

## 3. What lands where

| object | name | made by |
|---|---|---|
| raw files | `/Volumes/workspace/driveguard/raw/**` | you |
| bronze | `workspace.driveguard.bronze_smart_daily` (Delta, partitioned by month) | `01_bronze` |
| ingest log | `workspace.driveguard.ops_ingest_log` | `01_bronze` |
| silver | `workspace.driveguard.silver_smart_daily`, `silver_drive_lifecycle` | `02_silver` |
| gold | `workspace.driveguard.gold_drive_day_features` | `03_gold` |
| experiment | `/Shared/driveguard` | `04_train` |
| model | `workspace.driveguard.driveguard_failure_30d` (+ alias `champion`) | `04_train` |
| alerts | `workspace.driveguard.gold_risk_alerts` (Delta, `replaceWhere` per snapshot date) | `05_score` |
| monitoring | `workspace.driveguard.ops_monitoring` | `06_monitor` |
| endpoint | `driveguard-failure-30d` (optional) | `07_serving` |

## 4. Serving (optional)

`notebooks/07_serving.py` creates a Small CPU endpoint with scale-to-zero for the champion version and sends one
request. The model is logged as an MLflow pyfunc with its own code path, so the endpoint needs no extra wheel. Free
Edition limits the number of active endpoints and offers no GPU - neither matters here.

## 5. CI/CD

`.github/workflows/ci.yml` lints, runs the unit + end-to-end tests on local Spark, builds the wheel and validates the
bundle YAML on every push. If the repository has `DATABRICKS_HOST` (variable) and `DATABRICKS_TOKEN` (secret) set, a
second job deploys the bundle to the `dev` target on `main`; without them the deploy job is skipped and CI stays green
for anyone who clones the repo.

## 6. Knobs

All settings are environment variables read by `driveguard.config.Settings` (`DG_*`); notebooks expose the common ones
as widgets and the bundle passes them as job parameters:

| variable | default | meaning |
|---|---|---|
| `DG_HORIZON_DAYS` | 30 | label horizon |
| `DG_ALERT_BUDGET` | 0.005 | share of the fleet flagged per day |
| `DG_TEST_DAYS` / `DG_VAL_DAYS` / `DG_GAP_DAYS` | 90 / 60 / 30 | evaluation protocol |
| `DG_GROUP_HOLDOUT_FRAC` | 0 | unseen-drive-model holdout share |
| `DG_NEGATIVE_SAMPLE_RATIO` | 20 | negatives per positive in the training set (weights restore the prior) |
| `DG_PSI_ALERT` / `DG_PSI_FEATURES_FOR_RETRAIN` / `DG_MATURED_RECALL_FLOOR` | 0.2 / 3 / 0.5 | retraining trigger |

## 7. Scaling up

The code is the same at 5 % and 100 % of the fleet; what changes is where it runs. On a full quarter (~27 M rows)
bronze/silver/gold are plain Spark jobs; training still happens on the driver because the negatives-sampled training
set is small (all positives + 20× negatives), and scoring is distributed with `mapInPandas`. Beyond Free Edition,
point `databricks.yml` at a workspace with classic compute and nothing else needs to change.
