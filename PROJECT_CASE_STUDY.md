# DriveGuard — Engineering Case Study

> **Project type:** Portfolio implementation / end-to-end ML engineering on real hardware telemetry (Backblaze Drive Stats), built and run on Databricks; the daily replay is driven by a GitHub Actions workflow with a public run history  
> **Core stack:** Python, PySpark, Delta Lake, Unity Catalog, MLflow, XGBoost, Databricks Asset Bundles (serverless jobs), FastAPI, GitHub Actions  
> **Data and platform options:** Databricks Free Edition (serverless, CPU model serving) or any workspace; local mode with Parquet, local Spark and a sqlite MLflow store  
> **Repository:** https://github.com/wenjun78/driveguard  
> **Public console URL supplied by the project owner (live workspace reads or labelled snapshot fallback):** https://driveguard-kf09.onrender.com/  
> **Related documentation:** [README](README.md) · [Portfolio page](https://wenjun78.github.io/portfolio/driveguard.html) · [Evaluation protocol](docs/evaluation.md) · [Architecture](docs/architecture.md) · [Data notes](docs/data.md) · [Databricks guide](docs/databricks.md) · [Console](docs/console.md)

## The data

[Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data) (free to use with attribution): one row per drive per day for every hard drive in Backblaze's data centres since 2013 — serial number, model, capacity, whether the drive failed that day, and its SMART attributes (raw and normalized) — published as daily CSVs bundled by quarter. The fleet is in the hundreds of thousands of drives; a quarter is tens of millions of rows and several gigabytes of CSV.

This project loads a **10 % subset selected by serial-number hash** so that every kept drive carries its whole history (lifecycles, labels and censoring stay intact; a random sample of rows would destroy the windows). Quarters 2024Q3–2025Q1 are loaded (1 July 2024 – 31 March 2025): <!--dg:data.drives-->33,144<!--/dg--> drives, <!--dg:data.failures-->366<!--/dg--> recorded failures, about <!--dg:data.scored_per_day-->30,589<!--/dg--> drives observed on any given day, more than twenty drive models. `scripts/download_backblaze.py` performs the subsetting; `scripts/databricks_bootstrap.py` lands the Parquet files in a Unity Catalog Volume; the `fetch-quarters` workflow adds newer quarters as Backblaze publishes them.

## 1. Executive summary

DriveGuard answers an operational question — **which drives in a fleet should an operator inspect or replace this week, before they fail?** — as a complete ML system rather than a model in a notebook. Raw daily files land in Delta tables (bronze), are cleaned into one row per drive-day with a lifecycle table (silver), and become time-safe window features with a 30-day failure label that respects right-censoring (gold). An XGBoost ranker is trained on a time-ordered split with horizon-long gaps, evaluated at a fixed alert budget against the rule operators already use, logged to MLflow with its feature specification and a drift reference, registered in Unity Catalog, and promoted through a champion/challenger gate. A daily job scores the fleet, writes a ranked alert list with TreeSHAP reason codes, measures drift and real matured outcomes, and triggers retraining only when the monitor says so. A public console reads all of this from the workspace live.

Its distinguishing design choices are the **right-censored label**, the **label-disjoint time split**, the **budgeted operational metrics** (recall, precision, lead time and false-alert rate at 0.5 % of the fleet per day, not just PR-AUC), and the **replay clock** that lets a historical dataset be operated as if live, one data-day per real day, with every stage blind to days after the cursor.

On the real data, of the <!--dg:test.failing_drives-->120<!--/dg--> drives that failed in the <!--dg:split.test_days-->90<!--/dg-->-day held-out test period, **<!--dg:test.drive_recall0-->71<!--/dg--> % were on the daily list at least once in their last 30 days, with a median <!--dg:test.lead_median-->18<!--/dg--> days' warning**, at a budget of 0.5 % of the fleet; the five-counter operator rule catches <!--dg:rule.drive_recall0-->5<!--/dg--> % at the same budget.

**Evidence boundary:** This is a working system on a public dataset, run on Databricks Free Edition, not a deployment inside a storage operator. The metrics above are computed by the project's own evaluation code on a 10 % subset over three quarters; they have not been independently reproduced by a third party, and no operator has acted on its alerts. A live URL, green CI and a registered model do not establish production availability, cost, or a measured maintenance saving.

## 2. Background and business problem

Storage operators replace drives on a mixture of vendor SMART thresholds, rules of thumb and failure. Backblaze itself publishes the five SMART attributes it watches (5 reallocated sectors, 187 reported uncorrectable errors, 188 command timeouts, 197 pending sectors, 198 offline uncorrectable): any non-zero value is a reason to look. That rule has two problems at fleet scale:

| Situation | What the rule does | What an operator needs |
|---|---|---|
| Tens of thousands of drives, hundreds with some non-zero counter | Flags all of them equally | A ranked list sized to the people and spares available this week |
| A drive whose pending-sector count has been climbing for ten days | Flagged the same as one with a single old reallocation | Trajectory, not a snapshot |
| A drive that was decommissioned last week without failing | Looks like a survivor in any naive label | Treated as unknown, not healthy |
| A model evaluated on random held-out rows | Looks far better than it will be next month | Evaluation forward in time, on days and drives it never saw |
| A model that was good in March | Silently degrades as the fleet's models and ages change | Drift measurement and real outcome tracking once labels mature |

The project turns these into engineering requirements and tests. The business framing is predictive maintenance for storage; the ML-engineering framing is a rare-event, time-series-of-tabular problem with delayed labels, censoring and population shift — the shape of most industrial failure-prediction work.

## 3. Objectives, boundaries, and success criteria

**Primary objective:** a credible, inspectable, end-to-end ML system on Databricks whose evaluation matches the decision it supports, operated unattended from GitHub.

**Functional objectives**

1. Land Backblaze daily files idempotently, tolerating a decade of schema drift.
2. Build time-safe features and a censoring-aware 30-day label.
3. Train and evaluate on a time-ordered, label-disjoint split; compare against the operator rule at the same budget.
4. Register the model in Unity Catalog with a champion gate; package feature spec and drift reference with the weights.
5. Score the fleet daily, rank at a budget, attach reason codes, make re-runs idempotent.
6. Monitor input and score drift, back-fill real outcomes as labels mature, and gate retraining on that evidence.
7. Expose everything through a public console that reads the workspace live and degrades honestly.
8. Run the whole lifecycle from the repository: CI, one-click bootstrap, scheduled daily operation, automatic data top-up.

**Boundaries**

- Databricks Free Edition: serverless compute only, CPU-only serving, five concurrent job tasks, one SQL warehouse. The bundle respects these; the code does not depend on them.
- A 10 % subset first. The code is identical at 100 % of the fleet; the compute is not.
- No real operator in the loop. "Impact" is a design objective, not a measured outcome (section 10).

**Engineering acceptance criteria represented in code**

- `tests/test_labels_and_features.py`: label semantics, censoring, failure-row exclusion, calendar-day windows, and a test that deletes all future rows and asserts features are unchanged.
- `tests/test_splits_metrics.py`: split boundaries and gaps, unseen-model holdout routing, budgeted metrics, PSI.
- `tests/test_bronze_silver.py`: schema drift across files, duplicates, ghost rows, unknown capacities, idempotent ingest, and files held back after the replay cursor.
- `tests/test_pipeline_e2e.py`: the whole pipeline on a synthetic fleet, including backfill scoring and monitoring.
- `tests/test_app.py`: the console API over the real snapshot, including hindsight outcomes and what-if scoring.
- `tests/test_lifecycle_gates.py`: promotion held inside the incumbent's held-out period and resumed after it; the relative retraining floor; and that every marked figure in this document and the README equals the committed snapshot (CI fails if a champion changed and the docs did not).

## 4. Architecture and major components

```
Backblaze daily files (UC Volume)
   │  bronze.py   typed, schema-drift-safe, ingest log, replay hold-back
   ▼
bronze_smart_daily ──► silver.py ──► silver_smart_daily + silver_drive_lifecycle
                                        │  gold.py   range-frame windows · 30-day label · censoring
                                        ▼
                                 gold_drive_day_features
                   ┌────────────────────┼────────────────────────┐
                   ▼                    ▼                        ▼
             train.py             score.py                  monitor.py
   time split w/ gaps · XGBoost   mapInPandas · rank/day      PSI · matured outcomes
   metrics at budget vs rule      reason codes · replaceWhere  retrain_recommended
   MLflow pyfunc · UC @champion   gold_risk_alerts             ops_monitoring
                   │                    ▲                        │
                   └──── champion ──────┘        condition task ─┘──► retrain
```

| Component | Responsibility | Key decisions |
|---|---|---|
| `bronze.py` | Discover files, group by header, union by name, type, append; `ops_ingest_log` for idempotency | `_metadata.file_path` (UC-safe); files dated after the replay cursor are held back |
| `silver.py` | One row per drive-day, failure terminal, ghost rows dropped, capacities filled, lifecycle table | Failure wins duplicates; rows after a failure day are removed |
| `gold.py` | Window features over calendar days; label with censoring | `RANGE BETWEEN 30 PRECEDING AND CURRENT ROW` on the day number; no calendar-monotone features |
| `splits.py` | Train / val / test with horizon-long gaps; optional unseen-model holdout | Label-disjoint, not merely feature-disjoint |
| `model.py` | `FeatureSpec` + booster; MLflow pyfunc wrapper; reason codes | TreeSHAP via `pred_contribs`; code path packaged with the model |
| `train.py` | Negatives sampling with weights; early stopping; evaluation; registration; promotion gate | Reference sample for drift stored inside the model; `@champion` only once the incumbent's held-out period has elapsed *and* val PR-AUC ≥ incumbent's |
| `scoring.py` / `score.py` | Distributed scoring; per-day rank at budget; operator baseline; alert table | Model in the closure (no SparkContext on serverless); Delta `replaceWhere` per snapshot date |
| `monitor.py` | PSI per feature and on scores; matured precision/recall/lead time; retrain decision | Decision exposed as a job task value for the Workflows condition task |
| `replay.py` | The replay clock | `data_date@real_date` anchor; cursor clamps to the data |
| `app/` | Console: FastAPI, Databricks live source, snapshot source, DuckDB twin for exports | Same query code produces the live view and the snapshot |
| `resources/jobs.yml`, `databricks.yml` | Asset Bundle: build job and daily job on serverless | `live` target without dev-mode name prefixes so the console can find jobs by name |
| `.github/workflows/` | `ci`, `databricks-bootstrap`, `daily`, `fetch-quarters` | GitHub is the control plane; Databricks is the system of record |

## 5. End-to-end implementation walkthrough

1. **Bootstrap (one click).** The `databricks-bootstrap` workflow downloads the requested quarters on a GitHub runner, keeps whole histories for 10 % of serials, uploads Parquet to `/Volumes/workspace/driveguard/raw`, deploys the bundle, runs the build job, exports the console snapshot from the workspace through the SQL warehouse, publishes it to the single-commit `snapshot` branch and commits the first real snapshot to `main`.
2. **Bronze.** Files are grouped by header signature (attributes were added over the years; 2023+ files carry datacenter/pod columns), read per group, unioned by name, typed, and appended partitioned by month. Each landed file is logged; re-runs skip it. With a replay anchor, files dated after the cursor stay in the Volume.
3. **Silver.** Deduplicate on (serial, date) with the failure row winning; drop rows after a drive's failure day; replace `capacity_bytes = -1` with the drive's known capacity; derive first/last observation and failure date per drive.
4. **Gold.** For each drive-day: current raw values of the key and context attributes, change over 7 and 30 calendar days, days-with-non-zero counts, temperature mean/max, observation gaps, model and capacity. Label 1 if the recorded failure falls in `(t, t+30]`; 0 only if the drive is observed at or after `t+30` without failing; otherwise NULL (censored). The failure-day row is flagged and excluded.
5. **Train.** Split by day with 30-day gaps; keep all positives and a weighted 20× sample of negatives; XGBoost with early stopping on validation PR-AUC; evaluate validation, test and (optionally) unseen-model test with the budgeted metrics; evaluate the operator rule on the same test days; log everything to MLflow with the feature spec, an unbiased drift reference sample and a code path; register in Unity Catalog; set `@champion` if validation PR-AUC is at least the incumbent's (read from the incumbent's own run).
6. **Score.** Resolve `models:/…@champion`, score the snapshot day with `mapInPandas`, rank per day, flag the top 0.5 %, compute top-3 reason codes for the flagged rows, write `gold_risk_alerts` with `replaceWhere` on the date. A 45-day backfill gives monitoring history on day one.
7. **Monitor.** PSI of key features and of the score distribution against the reference inside the champion; real precision/recall/lead time for every snapshot whose 30 days have elapsed; `retrain_recommended` when at least three key features exceed PSI 0.2 or matured recall at budget over the last 7 snapshots falls below half the champion's own held-out recall (its `test_recall_at_budget`, read from its run); written to `ops_monitoring` and set as a task value.
8. **Daily job.** bronze → silver → gold → score → monitor → condition task → retrain, driven by the `daily` workflow at 02:40 UTC, which then refreshes the published snapshot. On the replay clock the job lands one more historical day each run.
9. **Console.** Overview (alerts per day split into failed / stayed healthy / pending), ranked alerts with reason codes and hindsight, drive history, monitoring, model lifecycle, pipeline runs and serving state, what-if scoring. Live reads are cached ten minutes with stale-while-revalidate; a failed read falls back to the published snapshot and the view says so.

## 6. Technical concepts explained in this implementation

- **Right-censoring.** A drive that leaves the fleet before its 30-day window closes has an unknown outcome, not a healthy one. Roughly a quarter of rows in a one-year window are censored. Labelling them 0 teaches the model that disappearing looks like health.
- **Label-disjoint splits.** A row's label depends on outcomes up to 30 days later. Without a gap between train and validation, training labels encode validation-period failures. Gaps equal to the horizon make the periods disjoint in outcomes, not just in features.
- **Calendar-day range frames.** Drives have gaps. "The last 30 rows" is not "the last 30 days"; the window is a `RANGE` frame over the integer day number.
- **Budgeted metrics.** The deployed artefact is a ranked list with a budget. Recall and precision at the budget, per-drive recall (flagged at least once before failure), lead time of the first alert and false alerts per 1,000 healthy drive-days describe what an operator experiences; PR-AUC is reported alongside.
- **Weighted negatives sampling.** All positives plus 20 negatives per positive, with sample weights that restore the fleet's prior, keep training on the driver at any fleet size while leaving probabilities interpretable (Brier score and score-distribution PSI remain meaningful).
- **Champion/challenger gate.** Two conditions. A challenger is *held* (registered, no alias) while the replay is still inside the incumbent's held-out test period: a model trained at cursor − 30 d is fitted on data overlapping that period and validated on a different window, so its metrics are not comparable, and promoting it would make every published result describe a model that is no longer scoring. Once that period has elapsed, promotion needs val PR-AUC ≥ the incumbent's, read from the incumbent's own run. Scoring always resolves the alias.
- **Relative retraining floor.** The matured-recall floor is half the champion's own held-out recall at budget rather than a fixed number, so it marks a collapse relative to the model that is scoring instead of a threshold tuned once by hand; drift (PSI ≥ 0.2 on ≥ 3 key features) triggers on its own.
- **Reason codes from the booster.** XGBoost's `pred_contribs` is TreeSHAP; an alert says which features pushed it up without a second model.
- **Replay clock.** `DG_REPLAY_ANCHOR=2025-01-01@2026-11-01` pins a data day to a calendar day. Bronze lands only files up to the cursor, training cuts at cursor − 30 d, scoring and monitoring treat the cursor as now. The replay is a forward simulation, not a replay of answers the model has seen.

## 7. Design decisions, alternatives, and trade-offs

| Decision | Alternative considered | Why this one | Cost |
|---|---|---|---|
| Censored labels (NULL) | "No failure in 30 d ⇒ 0" | Decommissioned drives and the last month of data are unknowns | Fewer labeled rows; the last horizon of every period is unusable for metrics |
| 30-day gaps between periods | Contiguous periods | Training labels must not depend on evaluation-period outcomes | 60 days of history spent on gaps |
| XGBoost on a weighted sample | Distributed GBT on all rows | Positives are rare; all of them plus 20× negatives is small and fits the driver; distributed training buys nothing here | Not "distributed training" on the CV |
| Scoring with `mapInPandas` | Collect to the driver | Every drive must be scored daily; this is the part that must scale | Model serialized into the closure per task |
| Operator rule as the bar | Random or majority baseline | A baseline the audience already uses is the honest comparison | Its ROC-AUC (<!--dg:rule.roc_auc2-->0.86<!--/dg-->) flatters it; PR-AUC (<!--dg:rule.pr_auc-->0.017<!--/dg-->) and recall at budget (<!--dg:rule.recall-->1.5<!--/dg--> %) show it is a flag, not a ranker |
| Drift reference stored inside the model | A separate reference table | The reference must travel with the champion it describes | Model artefact ~1 MB larger |
| Retrain gated on evidence; promotion held during the incumbent's held-out period | Calendar retraining; promote whatever beats the incumbent's number | Retrain when PSI or matured recall say so; a challenger trained mid-replay is not comparable to the incumbent, so it waits | Needs matured labels, so the first 30 days of a deployment have no outcome signal |
| Replay clock | Static tables + a static demo | The system is operated, not demonstrated; the fleet moves | Backblaze data is historical; the "live" rhythm is simulated and labelled as such |
| Console outside Databricks (Render) | Databricks App | Free Edition apps stop after 24 h and need a workspace login; a recruiter cannot open them | A second place to deploy; mitigated by the Render blueprint and the snapshot fallback |
| Snapshot on a force-pushed branch | Daily commits to `main` | A 5 MB snapshot committed daily would grow the repository by ~150 MB a month | One more branch; the console fetches it at start-up and every 6 h |
| GitHub as control plane | Databricks schedules only | Everything visible and reproducible from the repository; secrets in one place | GitHub runners do the download/upload; a token expiry stops the schedule |

## 8. Failure handling and operational considerations

- **Idempotency.** Bronze logs every landed file; the alerts table is overwritten per snapshot date with `replaceWhere`; monitoring appends one row per run; re-running any day is safe.
- **Degradation.** The console tries the workspace, falls back per endpoint to the published snapshot and shows a per-view source tag; the status endpoint lists the error per endpoint; the warm-up and background refresh keep a cold warehouse invisible to visitors.
- **Quotas.** All reads are cached; one warehouse connection is reused; the exporter uses bulk queries (a few dozen statements rather than thousands).
- **Serverless-specific failures encountered and fixed during bring-up:** `input_file_name()` rejected on Unity Catalog compute; no `SparkContext` for `broadcast`; the `spark` global injected lazily; pip upgrading numpy to 2.x and breaking the preinstalled pyarrow; `toPandas()` attaching query-plan metrics to `DataFrame.attrs`; the SQL connector returning ARRAY columns as numpy arrays; `secrets.*` not permitted in a job-level `if:` in GitHub Actions. Each is a one-line fix once seen and a day lost if not; the fixes are in [`docs/databricks.md`](docs/databricks.md) and the notebooks' first cells.
- **Credentials.** A personal access token in a GitHub secret and a Render environment variable; the workspace's maximum token lifetime is raised so the schedule does not stop every two weeks; rotation is manual.
- **What is not handled.** Backfilling features when a late file arrives for a past day (bronze appends; silver/gold recompute fully each run, which covers it at this scale but would not at 100 % of the fleet); multi-workspace promotion; per-user access control on the console (it is read-only and public by design).

## 9. Testing, evaluation, and evidence

### Evaluation design

- Time-ordered split on the real subset: train ≤ <!--dg:split.train_end-->2 Sep 2024<!--/dg--> · validation <!--dg:split.val_start-->3 Oct 2024<!--/dg--> – <!--dg:split.val_end-->1 Dec 2024<!--/dg--> · test <!--dg:split.test_start-->1 Jan 2025<!--/dg--> – <!--dg:split.test_end-->31 Mar 2025<!--/dg--> (<!--dg:split.test_days-->90<!--/dg--> days), with <!--dg:split.gap-->30<!--/dg-->-day gaps. The last 30 days of each period are censored by construction.
- Metrics at a 0.5 % daily budget (≈<!--dg:test.alerts_per_day-->156<!--/dg--> drives) for the model and for the operator rule on the same test days; PR-AUC, ROC-AUC and Brier alongside.
- Live monitoring back-fills real outcomes for every scored snapshot whose horizon has elapsed; the console shows them per day.
- 34 test functions are defined and the latest CI completed successfully; tests run on local Spark in CI (label semantics, censoring, time safety, schema drift, replay hold-back, splits, budgeted metrics, PSI, end-to-end pipeline, console API, promotion hold and retraining floor, docs–champion consistency).

### Results — champion `workspace.driveguard.driveguard_failure_30d` v<!--dg:champion.version-->4<!--/dg-->

| metric at the 0.5 % budget | validation | test | operator rule (test) |
|---|---|---|---|
| PR-AUC | <!--dg:val.pr_auc-->0.145<!--/dg--> | <!--dg:test.pr_auc-->0.184<!--/dg--> | <!--dg:rule.pr_auc-->0.017<!--/dg--> |
| ROC-AUC | <!--dg:val.roc_auc-->0.839<!--/dg--> | <!--dg:test.roc_auc-->0.924<!--/dg--> | <!--dg:rule.roc_auc-->0.863<!--/dg--> |
| Drive recall (flagged ≥ once in its last 30 d) | <!--dg:val.drive_recall-->57.0<!--/dg--> % | **<!--dg:test.drive_recall-->70.8<!--/dg--> %** | <!--dg:rule.drive_recall-->5.0<!--/dg--> % |
| Recall (drive-days) | <!--dg:val.recall-->39.1<!--/dg--> % | <!--dg:test.recall-->45.3<!--/dg--> % | <!--dg:rule.recall-->1.5<!--/dg--> % |
| Precision | <!--dg:val.precision-->8.6<!--/dg--> % | <!--dg:test.precision-->13.5<!--/dg--> % | <!--dg:rule.precision-->0.5<!--/dg--> % |
| Median lead time of first alert | <!--dg:val.lead_median-->23<!--/dg--> d | **<!--dg:test.lead_median-->18<!--/dg--> d** | <!--dg:rule.lead_median-->8<!--/dg--> d |
| Caught with ≥ 7 d lead | <!--dg:val.caught_7d-->84<!--/dg--> % | <!--dg:test.caught_7d-->79<!--/dg--> % | <!--dg:rule.caught_7d-->67<!--/dg--> % |
| False alerts / 1,000 healthy drive-days | <!--dg:val.false_alerts-->4.59<!--/dg--> | <!--dg:test.false_alerts-->4.47<!--/dg--> | <!--dg:rule.false_alerts-->5.03<!--/dg--> |
| Failing drives · labeled drive-days | <!--dg:val.failing_drives-->100<!--/dg--> · <!--dg:val.rows_labeled_m-->1.77<!--/dg--> M | <!--dg:test.failing_drives-->120<!--/dg--> · <!--dg:test.rows_labeled_m-->1.85<!--/dg--> M | <!--dg:rule.failing_drives-->120<!--/dg--> · <!--dg:rule.rows_labeled_m-->1.85<!--/dg--> M |

Top features by gain: <!--dg:importance.top6-->`smart_197_nzdays_7d`, `any_key_smart_nonzero`, `smart_197_delta_30d`, `smart_187_delta_30d`, `smart_187_delta_7d`, `smart_5_delta_30d`<!--/dg--> — persistence and change of the pending/uncorrectable counters, not their raw magnitude. Live matured recall on the replayed snapshots in the bundled console snapshot (as of <!--dg:data.as_of-->1 Jan 2025<!--/dg-->) runs <!--dg:live.matured_range-->0.3–0.4 per day (median ≈ 0.3)<!--/dg-->; the retraining floor is half the champion's held-out recall (<!--dg:live.floor-->23<!--/dg--> %).

The figures in this section, the README and the executive summary are rewritten from the console snapshot by `scripts/refresh_docs.py` whenever the champion version changes (the `daily` workflow commits the result), so they always describe the model that is scoring the fleet.

### What can and cannot be claimed

| Claim | Evidence status |
|---|---|
| A PySpark medallion pipeline, MLflow/UC lifecycle, scoring and monitoring have run on Databricks serverless | Directly supported: job runs in the workspace, tables in `workspace.driveguard`, registered model v1–v<!--dg:champion.version-->4<!--/dg--> |
| The label respects right-censoring and features are time-safe | Directly supported by code and tests |
| <!--dg:test.drive_recall0-->71<!--/dg--> % drive recall at <!--dg:test.lead_median-->18<!--/dg--> days' median lead on a <!--dg:split.test_days-->90<!--/dg-->-day held-out test | **Computed by the project's evaluation code** on the 10 % subset; recorded in the champion's MLflow run and `app_snapshot/model.json`; not independently reproduced |
| The model beats the operator rule | Supported at the 0.5 % budget on the same test days; the rule's ROC-AUC is comparable, its ranking is not |
| The replay advances one data-day per calendar day | Driven by the `daily` workflow since 1 Nov 2026. How many days have been replayed is not a claim in this document: the evidence is the workflow's [run history](https://github.com/wenjun78/driveguard/actions/workflows/daily.yml) and the job runs and replay cursor in the console's Pipeline view. The job's native Databricks schedule is paused on purpose — GitHub is the scheduler |
| Results hold at 100 % of the fleet or on newer quarters | **Not established** — the code is unchanged, the numbers are not re-run |
| An operator saved money or avoided data loss | **Not measured**; no operator is in the loop |
| The serving endpoint meets a latency or availability target | **Not measured** |

**How to reproduce local evidence**

```bash
pip install -e ".[local,dev,app]"
make fixture && make demo          # synthetic fleet end to end (proves the pipeline, not the model)
pytest -q                          # 34 tests
python scripts/download_backblaze.py --quarters 2024Q3 2024Q4 2025Q1 --serial-frac 0.10 --out data/raw
make demo && python scripts/report.py   # real numbers, locally
```

For the Databricks path, the README's "Run it from GitHub" section lists the two secrets and one click.

## 10. Impact — demonstrated versus intended

**Demonstrated technical capability**

- Real telemetry for 33k drives lands, is cleaned, featurised, labelled with censoring, and scored daily in a Databricks workspace from a one-click workflow.
- A model evaluated forward in time, at the budget an operator would use, against the rule an operator already has — with the comparison logged in MLflow rather than asserted.
- Real outcomes flow back into monitoring as labels mature; retraining is gated on that evidence.
- A public console reads the workspace live and degrades honestly.

**Intended business impact, not a measured outcome**

- Earlier replacement of drives that are about to fail, with fewer rebuilds and less exposure to multi-drive loss.
- Maintenance effort spent on a ranked, budget-sized list instead of every drive with a non-zero counter.
- Visibility of when the model stops working, instead of discovering it from failures.

No adoption, cost, downtime or data-loss figure exists for this project; these are hypotheses a storage operator would validate in a pilot, not outcomes to present as achieved.

## 11. Challenges, limitations, and next experiments

| Priority | Next work | Why it matters / evidence needed |
|---|---|---|
| P0 | Load 2024Q1–Q2 and newer quarters, reset, rebuild with the replay cursor from day one | A full year of history before the cursor; tables physically day-by-day; results re-run on more data |
| P0 | Enable the unseen-model holdout (`DG_GROUP_HOLDOUT_FRAC`) in the registered run | New drive models enter the fleet every quarter; transfer to unseen populations is the deployment-relevant question |
| P1 | Calibrate the budget: report recall/precision/lead at 0.25 %, 0.5 %, 1 % | The budget is the operator's knob; show the curve, not one point |
| P1 | Time-to-event framing (survival model or multi-horizon labels) | Lead time is a first-class objective; a single 30-day horizon flattens it |
| P1 | Incremental silver/gold (Delta MERGE by partition) | Full recompute is fine at 10 %; not at 100 % |
| P2 | Per-model calibration and model-specific SMART scales | Seagate/HGST raw scales differ; the tree copes, calibration may not |
| P2 | Cost and latency measurements for the daily job and the serving endpoint | Operational readiness beyond "it runs" |
| P2 | A Databricks App variant of the console for in-workspace users | The public console is for outsiders; operators live inside the platform |

Known limitations: precision at the 0.5 % budget is <!--dg:test.precision-->13.5<!--/dg--> % — most flagged drives do not fail within 30 days, which is the trade for <!--dg:test.drive_recall0-->71<!--/dg--> % drive recall with <!--dg:test.lead_median-->18<!--/dg--> days' notice; the subset is 10 % of the fleet over three quarters; the replay's rhythm is simulated; training is driver-side.

## 12. Interview preparation: answer the system, not just the model

These are suggested explanations consistent with the checked-in implementation; adapt them to the measured environment rather than reciting them as production claims.

**Q: What problem does DriveGuard solve?**  
A: Turning daily SMART telemetry into a ranked, budget-sized replacement list, with the evaluation done the way the list is used — forward in time, at the budget, against the rule operators already have.

**Q: Why is the label the hard part?**  
A: A drive that leaves the fleet before its 30-day window closes has an unknown outcome. Labelling it healthy teaches the model that disappearing is good. About a quarter of rows in a year are censored; they are scored in production and excluded from metrics.

**Q: Why gaps between train, validation and test?**  
A: Labels look 30 days ahead. A training row on the last training day carries a label that reads into validation. Gaps equal to the horizon make the periods label-disjoint, not just feature-disjoint.

**Q: Why PR-AUC and budgeted metrics rather than accuracy or ROC-AUC?**  
A: The base rate is <!--dg:test.base_rate-->0.15<!--/dg--> %. ROC-AUC is <!--dg:rule.roc_auc2-->0.86<!--/dg--> for the operator rule and <!--dg:test.roc_auc2-->0.92<!--/dg--> for the model; at a 0.5 % budget the rule catches <!--dg:rule.recall-->1.5<!--/dg--> % of positive drive-days and the model <!--dg:test.recall0-->45<!--/dg--> %. The budgeted numbers are the ones an operator feels.

**Q: Why XGBoost on a sample instead of distributed training?**  
A: Positives are rare. All of them plus twenty negatives each is small, fits the driver, and sample weights restore the prior so probabilities stay interpretable. Scoring — which touches every drive — is the part that is distributed.

**Q: How do you know when to retrain?**  
A: PSI of key features and of the score distribution against the reference stored with the champion, plus real recall at the budget for every snapshot whose 30 days have elapsed, against a floor set at half the champion's own held-out recall. A condition task in the daily job reads the flag. Promotion is a separate gate: held while the replay is inside the incumbent's held-out period, and then only for a challenger that beats the incumbent's validation PR-AUC — so the published numbers always belong to the model scoring the fleet.

**Q: What does "live" mean here, honestly?**  
A: Backblaze publishes history. The daily job runs on a replay clock — one historical day per real day — and every stage is blind to days after the cursor, including bronze and training. It is a forward simulation of operating the system, labelled as such everywhere it appears.

**Q: What broke on Databricks serverless?**  
A: `input_file_name()`, `sparkContext.broadcast`, the lazily injected `spark` global, pip upgrading numpy under the preinstalled pyarrow, and pandas `attrs` carrying plan metrics into Parquet. Each cost a run; each is one line once understood.

**Q: What was the impact?**  
A: A working, inspectable system with evaluation an operator could act on. The maintenance saving is the design objective; no operator has used it, so no saving is claimed.

**Q: What would you do before a real rollout?**  
A: Rebuild on the full fleet and more quarters with the unseen-model holdout on; show the budget curve; move to incremental feature computation; measure cost and latency; put the console inside the platform as a Databricks App for the people who would act on it.

---

### Source-of-truth guidance

The implementation, the MLflow runs and the tables in the workspace take precedence over prose. For a claim about the live deployment, inspect the workspace's job runs and the console's `/api/status`. This document is a candid engineering explanation, not proof that every optional path is active or that unmeasured operational benefits occurred.

## 13. Render and Databricks configuration

**Databricks.** Free Edition workspace; bundle target `live`; serverless environment version 3 with the project wheel; jobs `driveguard - build` and `driveguard - daily`; catalog `workspace`, schema `driveguard`, Volume `raw`; experiment `/Shared/driveguard`; registered model `workspace.driveguard.driveguard_failure_30d` with alias `champion`; optional CPU serving endpoint `driveguard-failure-30d` with scale-to-zero. Job parameters: `catalog`, `schema`, `raw_path`, `experiment`, `replay_anchor`.

**GitHub.** Secret `DATABRICKS_TOKEN`; variables `DATABRICKS_HOST`, `DG_REPLAY_ANCHOR` (and optionally `DATABRICKS_HTTP_PATH`, otherwise detected). Workflows: `ci` (push), `databricks-bootstrap` (manual; inputs `quarters`, `serial_frac`, `urls`, `skip_download`, `reset_tables`), `daily` (02:40 UTC), `fetch-quarters` (Mondays 03:15 UTC). The daily snapshot is force-pushed to the `snapshot` branch; when the exported champion version differs from the one committed in `app_snapshot/model.json`, the workflow also runs `scripts/refresh_docs.py` and commits the new snapshot plus the rewritten README and case-study figures to `main`. The portfolio page reads the same `model.json` from the repository at load time, so its figures follow too. There is no manual step between a new champion and every published number.

**Render.** `render.yaml` blueprint, Python runtime, free plan, Singapore region, `uvicorn driveguard.app.server:app`, health check `/healthz`. Environment: `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_HTTP_PATH`, `DG_CATALOG`, `DG_SCHEMA`, `DG_SERVING_ENDPOINT`, `DG_REPLAY_ANCHOR`, `DG_APP_CACHE_TTL`, generated `DG_APP_ADMIN_TOKEN` for the owner-only `POST /api/admin/run-daily` and `/api/admin/refresh`. Exact live values can differ from the checked-in blueprint because Render environment values are edited independently; `/api/status` reports which source is live and a length/prefix hint for the configured token.

## 14. Interface for external users

The console is a single static page over the API (no build step). DM Sans for text, Space Grotesk for headings, IBM Plex Mono for identifiers and figures; the portfolio's dark-green and lime palette; Chart.js vendored. Six views: Overview (KPIs, alerts per day split into failed / stayed healthy / pending, matured recall and precision, fleet by model); Alerts (the ranked list for any snapshot with reason codes and a hindsight column that becomes "failed in N d" / "healthy" / "left fleet" once the horizon has elapsed, with a jump to the latest snapshot that has outcomes; a drive panel with 120 days of key counters and daily scores, flagged days marked); Monitoring (matured series, runs table with PSI, drifted/warned features and the retrain decision, PSI by feature); Model (champion, evaluation table against the operator rule, feature importance, registered versions); Pipeline (stage strip, replay clock, serving state, job runs with links into the workspace); What-if (score a serial or hand-entered SMART values from a realistic healthy baseline, with reason codes and which route served the prediction). Every view carries its own "live" or "snapshot" tag; the header pill names the workspace when live and the snapshot date otherwise. Verified at 1360 px and the API documented at `/api/docs`. Code: [`web/`](web/), [`src/driveguard/app/`](src/driveguard/app/).
