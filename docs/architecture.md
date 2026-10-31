# Architecture

```mermaid
flowchart LR
    subgraph Raw["Unity Catalog Volume"]
        A[Backblaze daily CSV / Parquet subset<br/>quarterly drops]
    end
    subgraph Lakehouse["Delta tables (medallion)"]
        B[bronze_smart_daily<br/>typed, every column, schema-drift safe<br/>ops_ingest_log = idempotency]
        C[silver_smart_daily<br/>1 row per drive-day, failure terminal<br/>silver_drive_lifecycle]
        D[gold_drive_day_features<br/>rolling SMART features · 30-day label · censoring]
    end
    subgraph ML["MLflow + Unity Catalog"]
        E[train<br/>time split with gaps · XGBoost<br/>operator baseline · metrics at budget]
        F[(driveguard_failure_30d<br/>@champion)]
    end
    subgraph Ops["Daily job"]
        G[score<br/>mapInPandas · rank per day<br/>gold_risk_alerts + reason codes]
        H[monitor<br/>PSI · matured recall · retrain flag<br/>ops_monitoring]
        I{retrain_recommended?}
    end
    S[Model Serving endpoint<br/>CPU · scale-to-zero]
    A --> B --> C --> D --> E --> F
    D --> G --> H --> I
    I -- true --> E
    F --> G
    F --> S
```

## One package, three entry points

`src/driveguard/` is the whole system. The CLI (`driveguard bronze|silver|gold|train|score|backfill|monitor`), the
Databricks notebooks and the tests all call the same functions. Settings come from `DG_*` environment variables, so
a wheel built once runs on a laptop with Parquet and local MLflow, in CI with local Spark, and in a Databricks job with
Delta, Unity Catalog and the workspace tracking server - nothing is rewritten between the three.

| module | responsibility |
|---|---|
| `config.py` | settings, table names, the SMART attribute lists |
| `spark.py` | session (local or Databricks) and `TableStore`: UC tables on Databricks, Parquet paths locally |
| `bronze.py` | file discovery, header grouping, typed union, ingest log |
| `silver.py` | cleaning rules and the lifecycle table |
| `gold.py` | time-safe window features; label with censoring |
| `splits.py` | time-ordered split with gaps; unseen-model holdout |
| `model.py` | `FeatureSpec` + booster, MLflow pyfunc wrapper, reason codes (TreeSHAP contributions) |
| `train.py` | negatives sampling with weights, XGBoost early stopping, evaluation, MLflow logging, registration, promotion gate |
| `scoring.py` | distributed scoring (`mapInPandas`, broadcast model), per-day ranking at budget, operator rule baseline |
| `score.py` | daily / backfill scoring, alert table with `replaceWhere` idempotency |
| `monitor.py` | PSI drift, matured-label performance, retraining decision |
| `metrics.py` | budgeted alert metrics, PSI |
| `replay.py` | the replay clock (`DG_REPLAY_ANCHOR`): which data day is "today" |
| `app/` | the console: `server.py` (FastAPI, cache, fallback), `sources.py` (Databricks live + snapshot), `local_source.py` (DuckDB twin used to export the snapshot) |

## Design decisions worth defending in a review

* **Range-frame windows over calendar days.** Row-count windows silently change meaning when a drive has gaps.
* **Censored labels, not zero labels.** Decommissioned drives and the last month of data are unknowns, not healthy.
* **Gaps equal to the horizon between periods.** Label-disjoint splits, not merely feature-disjoint.
* **No calendar-monotone features.** "Days observed" would be out of distribution for every test row.
* **Negatives sampled with weights.** Training stays small enough for the driver at any fleet size; probabilities keep
  the fleet's prior so the Brier score and the score-PSI monitor mean something.
* **Budgeted metrics first.** PR-AUC is logged, but recall / precision / lead time *at the alert budget* are what an
  operator can act on, and the operator's own rule is evaluated the same way as the bar to clear.
* **The reference sample travels with the model.** Drift is measured against the fleet the champion was trained on,
  not against a rebalanced training set and not against whatever happened to be in a table last week.
* **Promotion is gated on a metric read from the champion's own run.** A worse retrain never becomes champion.
* **`replaceWhere` on the alerts table.** Re-running a day is idempotent; the monitor's matured-label join is never
  double counted.
* **Reason codes from the booster itself.** `pred_contribs` is TreeSHAP; an alert says *why* without a second model.
