# DriveGuard — fact sheet

One page of verifiable facts about the project, kept current with the deployed system (last updated 1 Nov 2026).

## What it is
Predictive hard-drive failure alerts on real Backblaze SMART telemetry, built as an end-to-end ML system on
Databricks: PySpark medallion tables, a right-censored 30-day label, time-ordered evaluation with gaps, an XGBoost
ranker measured at a fixed alert budget against the operator rule, MLflow + Unity Catalog lifecycle with a champion
gate, daily alerts with reason codes, drift and matured-label monitoring, condition-triggered retraining, Asset
Bundle jobs — operated from GitHub Actions, with a public console that reads the workspace live.

## Links
| | |
|---|---|
| Source | https://github.com/wenjun78/driveguard |
| Live console | https://driveguard-kf09.onrender.com |
| Case study | https://wenjun78.github.io/portfolio/driveguard.html |
| Data | Backblaze Drive Stats — https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data (free with attribution) |
| Platform | Databricks Free Edition (serverless compute, Unity Catalog, MLflow, Model Serving, Asset Bundles) |

## Data loaded
- Quarters 2024Q3–2025Q1 (1 Jul 2024 – 31 Mar 2025); 10 % of drives selected by serial hash so every kept drive has its whole history.
- 33,144 drives, 366 failures, ≈31k drives scored per day; fleet is 20+ drive models (Toshiba MG08/MG07, WDC WUH72, Seagate ST16000/ST12000 …).
- Newer quarters are added by the `fetch-quarters` workflow as Backblaze publishes them (Q2 2026 was published Sep 2026).

## Problem definition
- **Label**: drive fails within (t, t+30]. A row is 0 only when the drive is observed surviving the window; drives that leave the fleet or hit the end of the data earlier are censored (NULL) and excluded from training and metrics. The failure-day row is excluded as the outcome, not a decision point.
- **Features**: 7-day and 30-day range-frame windows over calendar days (deltas, non-zero-day counts, extremes) on SMART 5/187/188/197/198, plus power-on hours, temperature, power cycles, load cycles, CRC errors, workload counters, drive model and capacity. No feature grows with calendar time.
- **Split**: time-ordered with 30-day unlabeled gaps. Current champion: train ≤ 2 Sep 2024 · validation 3 Oct – 1 Dec 2024 · test 1 Jan – 31 Mar 2025 (90 days, 120 failing drives, 1.85 M labeled drive-days).
- **Budget**: the top 0.5 % of the fleet per day is flagged (≈155 drives).
- **Bar**: Backblaze's five-counter operator rule (any of SMART 5/187/188/197/198 non-zero) through the identical budgeted evaluation.

## Results — champion `workspace.driveguard.driveguard_failure_30d` v4
| metric at the 0.5 % budget | validation | test | operator rule (test) |
|---|---|---|---|
| PR-AUC | 0.145 | 0.184 | 0.017 |
| ROC-AUC | 0.839 | 0.924 | 0.863 |
| Drive recall (flagged ≥ once in its last 30 d) | 57.0 % | **70.8 %** | 5.0 % |
| Recall (drive-days) | 39.1 % | 45.3 % | 1.5 % |
| Precision | 8.6 % | 13.5 % | 0.5 % |
| Median lead time of first alert | 23 d | **18 d** | 8 d |
| Caught with ≥ 7 d lead | 84 % | 79 % | 67 % |
| False alerts / 1,000 healthy drive-days | 4.59 | 4.47 | 5.03 |
| Failing drives · labeled drive-days | 100 · 1.77 M | 120 · 1.85 M | 120 · 1.85 M |

- Headline: of the 120 drives that failed in the test period, 71 % were on the daily list within their last 30 days, with a median 18 days' warning; the list is 13.5 % right per drive-day against a 0.15 % base rate.
- The operator rule is a flag, not a ranking: at the same budget it surfaces the drives with the largest counter values (mostly old survivors) and catches 5 % of failures. Its ROC-AUC (0.86) and PR-AUC (0.017) disagree for exactly that reason.
- Top features by gain: `smart_197_nzdays_7d` (days with pending sectors in the last week), `any_key_smart_nonzero`, `smart_197_delta_30d`, `smart_187_delta_30d`, `smart_187_delta_7d`, `smart_5_delta_30d`. Persistence beats magnitude.
- Live matured recall per replayed day runs 0.3–0.6 (median ≈ 0.5); the retraining floor is 0.3.
- Regenerate: `python scripts/report.py` against the workspace, or read `app_snapshot/model.json`.

## How it runs
- **Databricks**: jobs `driveguard - build` (one-off) and `driveguard - daily` (bronze → silver → gold → score → monitor → condition task → retrain), serverless environment v3, Delta tables in `workspace.driveguard`, experiment `/Shared/driveguard`, registered model with `@champion` alias, optional CPU serving endpoint.
- **GitHub Actions**: `ci` (lint, 28 tests on local Spark, wheel, bundle deploy) · `databricks-bootstrap` (download → Volume → deploy → build → snapshot) · `daily` (10:40 MYT: run the job, publish a fresh snapshot to the `snapshot` branch) · `fetch-quarters` (Mondays: new Backblaze quarters into the Volume).
- **Console**: FastAPI + static UI on Render; reads the SQL warehouse, MLflow/UC, Jobs API and Model Serving; 10-minute cache with stale-while-revalidate; falls back to the latest published snapshot and says so.
- **Replay clock**: `DG_REPLAY_ANCHOR=2025-01-01@2026-11-01`. "Today" started at 1 Jan 2025 on 1 Nov 2026 and advances one data-day per real day — walking the champion's held-out test period. Bronze lands only files up to the cursor; training cuts at cursor − 30 d; scoring and monitoring treat the cursor as now.

## Serverless lessons learned (things the first runs broke on)
- `input_file_name()` is rejected on Unity Catalog compute → `_metadata.file_path`.
- No `SparkContext` on serverless → no `broadcast`; ship the model inside the `mapInPandas` closure.
- The `spark` global is injected lazily → create `DatabricksSession` explicitly in the first cell.
- Pin `numpy<3`, `pandas<3`: letting pip upgrade numpy breaks the preinstalled pyarrow (`numpy.core.multiarray failed to import`).
- `toPandas()` attaches query-plan metrics to `DataFrame.attrs`, which `to_parquet` cannot JSON-serialize → strip them.
- The SQL connector returns ARRAY columns as numpy arrays; UC `search_model_versions` may not populate `aliases`.
- `secrets.*` is not allowed in a job-level `if:` in GitHub Actions — the whole workflow file is refused.

## Limitations
- Training is driver-side on a weighted negatives sample (all positives + 20× negatives); scoring is distributed. Right for a tabular model at fleet scale; not distributed training.
- Free Edition quotas mean "subset first" (10 % of drives). The code is identical at 100 %; the compute is not.
- Precision at a 0.5 % budget is 13.5 %: most flagged drives do not fail within 30 days. That is the trade an operator makes for 71 % drive recall with 18 days' notice; the budget is a knob.
- The synthetic fixture (`scripts/make_fixture.py`) is for tests and dry runs only; nothing computed from it is reported as a result.
