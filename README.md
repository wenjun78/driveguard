# DriveGuard — predictive storage failure intelligence

[![ci](https://github.com/wenjun78/driveguard/actions/workflows/ci.yml/badge.svg)](https://github.com/wenjun78/driveguard/actions/workflows/ci.yml)

End-to-end ML engineering on real hardware telemetry: **which drives in a fleet should an operator inspect or replace
this week, before they fail?** Built on [Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)
(daily SMART snapshots of ~300k production drives), as a Databricks lakehouse pipeline — PySpark medallion tables,
MLflow + Unity Catalog model lifecycle, budgeted batch alerting with reason codes, drift and matured-label monitoring,
condition-triggered retraining, Asset Bundle jobs, CI. Runs on Databricks Free Edition; runs identically on a laptop.

```mermaid
flowchart LR
    A[Backblaze daily SMART files<br/>UC Volume] --> B[bronze<br/>typed · schema-drift safe · idempotent]
    B --> C[silver<br/>1 row / drive-day · lifecycle]
    C --> D[gold<br/>time-safe window features<br/>30-day label with censoring]
    D --> E[train<br/>time split with gaps · XGBoost<br/>vs operator rule · MLflow]
    E --> F[(UC model<br/>@champion)]
    D --> G[score<br/>ranked alerts at budget<br/>+ reason codes]
    F --> G --> H[monitor<br/>PSI · matured recall]
    H --> I{retrain?} -- yes --> E
    F --> S[Model Serving<br/>CPU · scale-to-zero]
```

## What makes it an ML *system*, not a notebook

| concern | what DriveGuard does |
|---|---|
| **Label** | `fails within (t, t+30]`. Drives that leave the fleet or hit the end of the data before `t+30` are **censored** (NULL), not labelled healthy. The failure-day row is excluded as the outcome, not a decision point. |
| **Time safety** | Every feature is a function of the drive's past only; windows are range frames over *calendar days*, not row counts. A test deletes the future and asserts features are unchanged. No calendar-monotone features. |
| **Evaluation** | Strictly time-ordered train / val / test with **horizon-long unlabeled gaps**, so training labels never depend on validation-period outcomes. Optional held-out drive *models* (`test_unseen`). |
| **Metrics that match the decision** | Per day the top 0.5 % of the fleet is flagged. Reported: recall / precision **at that budget**, per-drive recall, **lead time** of the first alert, false alerts per 1,000 healthy drive-days — alongside PR-AUC. |
| **The bar** | The operator's rule (Backblaze's five SMART counters, non-zero ⇒ alert) goes through the identical budgeted evaluation. The model must beat it on the same test days. |
| **Lifecycle** | MLflow pyfunc with the feature spec and a drift reference sample inside the artefact; Unity Catalog registration; `@champion` alias assigned only if validation PR-AUC ≥ the current champion's, read from its own run. |
| **Operations** | `gold_risk_alerts`: score, per-day rank, flag, top-3 TreeSHAP reason codes; `replaceWhere` makes re-runs idempotent. `ops_monitoring`: PSI on key features and on scores, **real** precision/recall back-filled once each snapshot's 30 days have elapsed, and a `retrain_recommended` task value that gates the retrain task in the daily job. |
| **Platform** | Delta tables in UC, serverless jobs from a Databricks Asset Bundle, optional CPU serving endpoint, GitHub Actions running lint + unit + end-to-end tests on local Spark and building the wheel. |

Details: [`docs/evaluation.md`](docs/evaluation.md) · [`docs/architecture.md`](docs/architecture.md) ·
[`docs/data.md`](docs/data.md) · [`docs/databricks.md`](docs/databricks.md)

## Quickstart

**Laptop, synthetic fleet (3 minutes, no download):**

```bash
pip install -e ".[local,dev]"
make fixture            # 3,000 synthetic drives × 400 days in the Backblaze file layout -> data/raw
make demo               # bronze -> silver -> gold -> train -> 45-day scoring replay -> monitor
python scripts/report.py
pytest -q               # 17 tests incl. an end-to-end run
```

**Laptop or Databricks, real data:**

```bash
# whole drive histories for ~10 % of the fleet, three quarters, the SMART columns the pipeline uses
python scripts/download_backblaze.py --quarters 2024Q3 2024Q4 2025Q1 --serial-frac 0.10 --out data/raw
make demo                                                   # locally, or:
databricks fs cp -r data/raw dbfs:/Volumes/workspace/driveguard/raw
databricks bundle deploy -t dev && databricks bundle run -t dev driveguard_build
```

Free Edition notes, table names, knobs and the daily job are in [`docs/databricks.md`](docs/databricks.md).

## Results

### Real data

> Pending — the Backblaze subset has to be downloaded on a machine with internet access and run (`make demo` or the
> `driveguard_build` job). `python scripts/report.py` prints this table from the latest MLflow run; paste it here
> with the subset used (quarters, serial fraction) and the number of drives / failures in the test period.

### Synthetic dry run — proves the pipeline, **not** the model

3,000 synthetic drives × 400 days (1.03 M drive-days, 127 failures, exaggerated 4 % AFR), 2 CPU cores, local Spark:
bronze 31 s · silver 21 s · gold 35 s · train 38 s · 45-day scoring replay 28 s · monitor 14 s.

| metric | Validation | Test | Operator rule (test) |
|---|---|---|---|
| PR-AUC | 0.548 | 0.548 | 0.513 |
| Recall @ budget (drive-days) | 57.7% | 57.5% | 52.1% |
| Precision @ budget | 36.0% | 49.1% | 45.6% |
| Drive recall @ budget | 75.0% | 73.3% | 73.3% |
| Median lead time (days) | 15 | 22 | 18 |
| Caught with ≥ 7 d lead | 81% | 91% | 86% |
| False alerts / 1,000 healthy drive-days | 3.42 | 2.96 | 3.09 |
| Failing drives in period | 28 | 30 | 30 |

The fixture's degradation *is* the five counters the operator rule watches, so a small margin over the rule is the
expected outcome there; the point of the table is that every stage runs, the metrics are the operational ones, and
the comparison against the rule is wired in. Real-data numbers replace this section.

## Repository

```
src/driveguard/        the system: config, spark/table store, bronze, silver, gold, splits, model, train, scoring, score, monitor, metrics, cli
notebooks/             Databricks notebooks 01-07 — thin wrappers that pass widgets to the package
resources/jobs.yml     Asset Bundle jobs: one-off build; daily ingest -> score -> monitor -> condition -> retrain
databricks.yml         bundle, variables, wheel artifact, targets
scripts/               download_backblaze.py (subset by serial hash) · make_fixture.py (synthetic fleet) · report.py
tests/                 label semantics, censoring, time safety, schema drift, splits, budgeted metrics, PSI, end-to-end
docs/                  evaluation protocol, architecture, data notes, Databricks guide
.github/workflows/     lint, tests on local Spark, wheel build, optional bundle deploy
```

## Limitations, stated plainly

* Training happens on the driver on a negatives-sampled set (all positives + 20× negatives, weighted). That is the right
  trade at fleet scale for a tabular model, but it is not distributed training.
* Free Edition quotas mean "subset first". The pipeline is the same at 100 % of the fleet; the compute is not.
* The synthetic fixture is for tests and dry runs. Nothing computed from it is a result.
* Serving is a CPU endpoint for the champion; the operational surface is the batch alert table.

## Attribution and licence

Data: Backblaze Drive Stats, used with attribution per Backblaze's terms. Code: MIT © 2026 wjun78.
