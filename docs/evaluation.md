# Evaluation protocol

The question the model answers is operational: **given everything observed about a drive up to today, should it be
on the short list a storage administrator inspects or replaces this week?** The protocol is built so that the answer
is measured the way it would be used - forward in time, on drives and periods the model never saw, at a fixed alert
budget - and so that nothing from the future leaks into the past.

## 1. Label: fails within the next 30 days

For a drive *d* on day *t*:

| situation | label | used for |
|---|---|---|
| recorded failure day in `(t, t+30]` | **1** | train / eval |
| observed on or after `t+30` without failing in the window (or failed later) | **0** | train / eval |
| left the fleet, or the data ends, before `t+30` with no failure | **NULL** (censored) | scored in production, excluded from metrics |
| the failure-day row itself | NULL, `is_failure_row=1` | excluded: it is the outcome, not a decision point |

The censoring rule matters more than it looks. The naive alternative - "no failure within 30 days ⇒ 0" - labels every
drive that was decommissioned or that sits in the last month of the data as healthy, which is exactly where a model
learns that "disappearing" looks like health. Roughly a quarter of the rows in a one-year window are censored.

A handful of serials in the raw files report rows *after* their failure day. Silver drops those, so a failure is always
the drive's final observation and `days_to_failure` is well defined.

## 2. Features: a function of the past only

Every feature on day *t* is computed from rows with `day <= t` of the same drive. Window features use **range frames
over calendar days** (`RANGE BETWEEN 30 PRECEDING AND CURRENT ROW` on the integer day), not row counts - drives have
gaps, and the two differ. The suite has a test that recomputes features after deleting all future rows and asserts
they are identical.

Deliberately absent: "days since first observed". It grows with calendar time, so under a time-ordered split every
test row carries values the model never saw in training. Power-on hours (SMART 9) is the age signal instead; it is a
property of the drive, not of the dataset window.

## 3. Split: time-ordered with horizon-long gaps

```
|------------- train -------------|  gap  |---- val ----|  gap  |-------- test --------|
                                 T1-30   T1          T2-30   T2                      end
```

The gaps are the subtle part. A training row on day `T1` carries a label that depends on outcomes up to `T1+30` -
inside the validation period. Without a gap the model is fitted on the very failures it is then evaluated on. With
`gap = horizon` the periods are *label*-disjoint, not just feature-disjoint. The last 30 days of the test period are
censored by construction, so `test_days` is set well above the horizon.

Optional **unseen-population test**: `DG_GROUP_HOLDOUT_FRAC` removes a hashed share of drive *models* from training and
validation and reports their test rows separately as `test_unseen` - does the model transfer to a drive population it
has never seen? New models enter real fleets every quarter, so this is the deployment-relevant question.

## 4. Metrics at the alert budget

The deployed artefact is a ranked list with a budget, not a probability threshold. Per day the top
`ceil(0.5 % × drives scored)` are flagged (`DG_ALERT_BUDGET`). Reported on every evaluation split, for the model
and for the operator baseline:

| metric | meaning |
|---|---|
| `pr_auc`, `roc_auc`, `brier` | threshold-free ranking quality and calibration |
| `recall_at_budget` | of all drive-days that precede a failure within 30 days, the share that were flagged |
| `precision_at_budget` | of all flagged drive-days, the share that really preceded a failure |
| `drive_recall_at_budget` | of all drives that failed, the share flagged **at least once** inside their 30-day window |
| `lead_time_median_days`, `share_caught_with_7d_lead` | how early the first alert came for the drives that were caught |
| `false_alerts_per_1000_drive_days` | noise generated on healthy drives |
| `alerts_per_day` | the budget in absolute terms |

Positives are weighed per drive as well as per row because a drive that is flagged on 25 of its last 30 days and one
flagged only on the last day both count as "caught" to an operator - but only the first one was actionable, which is
why lead time is reported next to recall.

## 5. The bar: the operator rule

Backblaze's own guidance to operators is to watch five SMART counters (5, 187, 188, 197, 198). `rule_baseline_score`
ranks drives by "any of the five is non-zero, then by their sum" and is put through the identical budgeted
evaluation. The model has to beat that rule on the same test days to be worth deploying; the comparison is logged to
MLflow as `baseline_test_*` next to `test_*`.

## 6. Promotion: champion / challenger

Every training run registers a version in Unity Catalog. It receives the `champion` alias only if its validation
PR-AUC is at least the current champion's (the metric is read back from the champion's own run, not from a cached
number). Scoring and monitoring always resolve `models:/<name>@champion`, so a worse retrain can never silently take
over production.

## 7. Monitoring and the retraining trigger

Each monitoring run:

1. computes PSI between the training-period reference sample stored *with the model* and today's fleet for the key
   SMART features, plus PSI of the score distribution;
2. back-fills real precision / recall / lead time for every past snapshot whose 30-day horizon has now elapsed
   (`matured_*`);
3. writes a row to `ops_monitoring` and sets the job task value `retrain_recommended` when either at least three key
   features exceed PSI 0.2 or matured recall at budget averaged over the last snapshots falls below 0.3 (real-fleet single-day values run 0.3–0.6).

The Databricks daily job has a condition task on that value, so retraining happens when the evidence says so, not on
a calendar.

## Replay and retraining

On the replay clock (`DG_REPLAY_ANCHOR`) "now" is a cursor inside the historical data. **Bronze lands only files dated
up to the cursor** - later quarters can sit in the Volume and each daily run ingests exactly one more day, like a
live feed - so silver, gold, scoring, monitoring and any retrain can only ever see the past. (If a cursor is set on
tables that already contain later days, a retrain additionally cuts at `now − horizon`; the clean way is to reset
the tables and rebuild, which the bootstrap workflow's `reset_tables` input does.) The replay is a genuine forward
simulation, not a replay of answers the model has seen.

## What the synthetic dry run can and cannot show

`scripts/make_fixture.py` produces a fleet with the same file layout, lifecycle mechanics and quirks as the real
data, with an exaggerated failure rate so a small fleet yields enough positives to exercise every code path. It is
used by the tests and for a laptop dry run. Numbers from it are **not** results - they show the pipeline works, not
that the model does. Real numbers come from the Backblaze subset (`docs/data.md`).
