# Data: Backblaze Drive Stats

Backblaze publishes a daily snapshot of every hard drive in its data centres - serial number, model, capacity,
whether it failed that day, and the drive's SMART attributes (raw and normalized) - as one CSV per day, bundled into
quarterly zips, since 2013. The fleet is in the hundreds of thousands of drives; a quarter is tens of millions of
rows. The data is free to use with attribution to Backblaze.

Catalogue and licence note: <https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data>

## What the pipeline uses

| column | role |
|---|---|
| `date`, `serial_number`, `model`, `capacity_bytes`, `failure` | identity, lifecycle, label source |
| `smart_5_raw`, `smart_187_raw`, `smart_188_raw`, `smart_197_raw`, `smart_198_raw` | the five counters Backblaze itself watches: reallocated sectors, reported uncorrectable errors, command timeouts, pending sectors, offline uncorrectable |
| `smart_9_raw` (power-on hours), `smart_194_raw` (temperature), `smart_12_raw` (power cycles), `smart_192/193_raw`, `smart_199_raw` (CRC errors), `smart_1/3/4/7/10/196_raw`, `smart_240/241/242_raw` (head flying hours, LBAs written/read) | age, thermal, workload and wear context |
| `smart_{5,187,197,198,194}_normalized` | vendor-normalized versions of the key counters |

Everything else in the files is landed in bronze untouched but not used downstream.

## Quirks the pipeline handles explicitly

* **Schema drift** - attributes were added over the years and 2023+ files carry `datacenter`, `cluster_id`, `vault_id`,
  `pod_id`, `pod_slot_num`, `is_legacy_format`. Bronze groups files by header and unions by name.
* **Duplicate drive-days** and rows for the same serial on the failure day with and without the flag. Silver keeps one
  row per drive-day, failure winning.
* **`capacity_bytes = -1`** when unknown. Silver fills with the drive's known capacity.
* **Rows after the failure day** for a few serials. Silver drops them so a failure is terminal.
* **Drives that leave without failing** (decommissioned, migrated) and the end of the data. Gold marks their last 30
  days of rows as censored rather than labelling them healthy (`docs/evaluation.md`).
* **Vendor-specific raw scales** (Seagate's SMART 1/7 raw values are huge, HGST's are 0). Trees cope; the model
  column and the normalized counters carry the context.
* **Missing attributes per model** - not every drive reports every attribute. XGBoost handles NaN natively; nothing is
  imputed.

## Subsetting

`scripts/download_backblaze.py` keeps whole drive histories by hashing the serial number (`crc32(serial) % 10000 <
frac × 10000`), so a 10 % subset is a 10 % fleet with intact lifecycles, labels and censoring - not a 10 % random sample
of rows, which would destroy the windows. The same hash selects the same drives in every quarter.

## The synthetic fixture

`scripts/make_fixture.py` writes a fleet with the same layout and quirks - degradation in the key counters before
most (not all) failures, drives joining and leaving, model mix, schema drift, duplicates, -1 capacities, a ghost row -
with an exaggerated failure rate so a small fleet has enough positives. It exists for tests and dry runs only.
