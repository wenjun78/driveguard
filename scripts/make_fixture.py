#!/usr/bin/env python3
"""Generate a SYNTHETIC fleet in the exact file layout Backblaze publishes (daily CSVs in quarterly folders).

Purpose: unit tests, CI and a laptop dry run of the whole pipeline *without* the real multi-GB download.
It is not a substitute for the real data and no number computed from it is reported as a result anywhere.

What it imitates deliberately: drives joining and leaving the fleet (right-censoring), a failure flagged on the
drive's last day, degradation in the five key SMART counters in the weeks before most - not all - failures, model
mix with different base rates, schema drift between early and late files, duplicate rows, -1 capacities and a
post-failure ghost row.
"""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

MODELS = [
    # name, capacity bytes, relative hazard, mean temperature
    ("ST12000NM0008", 12_000_138_625_024, 1.0, 29),
    ("HGST HUH721212ALE604", 12_000_138_625_024, 0.5, 27),
    ("TOSHIBA MG08ACA16TA", 16_000_900_661_248, 0.8, 31),
    ("WDC WUH721816ALE6L4", 16_000_900_661_248, 0.6, 30),
]
EARLY_ONLY_MISSING = ["smart_240_raw", "smart_241_raw", "smart_242_raw"]
LATE_EXTRA = ["datacenter", "cluster_id", "vault_id", "pod_id", "pod_slot_num", "is_legacy_format"]


def quarter_dir(d: date) -> str:
    return f"data_Q{(d.month - 1) // 3 + 1}_{d.year}"


def generate(
    out: str, n_drives: int, n_days: int, start: date, annual_failure_rate: float, seed: int, drift_day: int
) -> dict:
    rng = np.random.default_rng(seed)
    model_idx = rng.choice(len(MODELS), size=n_drives, p=[0.4, 0.25, 0.2, 0.15])
    hazard = np.array([MODELS[i][2] for i in model_idx])
    daily_p = annual_failure_rate / 365.0 * hazard / hazard.mean()

    # fleet dynamics
    install = np.where(rng.random(n_drives) < 0.8, 0, rng.integers(1, max(2, n_days - 60), n_drives))
    fail_day = np.full(n_drives, 10**9)
    for i in range(n_drives):
        # geometric time to failure from install
        t = rng.geometric(daily_p[i])
        if install[i] + t < n_days:
            fail_day[i] = install[i] + t
    removed = np.where(rng.random(n_drives) < 0.08, rng.integers(30, n_days, n_drives), 10**9)
    removed = np.where(removed < fail_day, removed, 10**9)
    last_day = np.minimum(fail_day, removed)
    # degradation: ~75 % of failures show SMART growth starting 5-60 days before; the rest fail "silently"
    shows_signal = rng.random(n_drives) < 0.75
    lead = rng.integers(5, 61, n_drives)
    degrade_start = np.where(shows_signal, fail_day - lead, 10**9)

    # static per-drive characteristics
    poh0 = rng.integers(500, 40000, n_drives)  # power-on hours at start
    temp_mean = np.array([MODELS[i][3] for i in model_idx]) + rng.normal(0, 1.5, n_drives)
    s1_scale = np.where(np.isin(model_idx, [0]), 1e8, 0)  # Seagate-style large raw read-error counters
    healthy_realloc = np.where(rng.random(n_drives) < 0.04, rng.integers(1, 40, n_drives), 0)

    # cumulative counters
    s5 = healthy_realloc.astype(float).copy()
    s187 = np.zeros(n_drives)
    s188 = np.zeros(n_drives)
    s197 = np.zeros(n_drives)
    s198 = np.zeros(n_drives)
    s12 = rng.integers(5, 60, n_drives).astype(float)
    s193 = rng.integers(100, 5000, n_drives).astype(float)
    s199 = np.where(rng.random(n_drives) < 0.03, rng.integers(1, 10, n_drives), 0).astype(float)
    s241 = rng.integers(10**9, 5 * 10**10, n_drives).astype(float)
    s242 = rng.integers(10**9, 8 * 10**10, n_drives).astype(float)

    serials = np.array([f"ZCH{seed % 10}{i:07d}" for i in range(n_drives)])
    stats = {"drives": n_drives, "days": n_days, "failures": int((fail_day < 10**9).sum()), "files": 0, "rows": 0}
    ghost_written = False
    for d in range(n_days):
        day = start + timedelta(days=d)
        alive = (install <= d) & (d <= last_day)
        idx = np.where(alive)[0]
        if len(idx) == 0:
            continue
        # healthy noise
        s12[idx] += rng.random(len(idx)) < 0.01
        s193[idx] += rng.poisson(3, len(idx))
        s199[idx] += rng.random(len(idx)) < 0.0005
        s241[idx] += rng.integers(10**6, 5 * 10**7, len(idx))
        s242[idx] += rng.integers(10**6, 8 * 10**7, len(idx))
        # transient pending sectors on healthy drives that clear
        blip = idx[rng.random(len(idx)) < 0.002]
        s197[blip] += rng.integers(1, 9, len(blip))
        clear = idx[(s197[idx] > 0) & (degrade_start[idx] > d) & (rng.random(len(idx)) < 0.3)]
        s197[clear] = np.maximum(0, s197[clear] - rng.integers(1, 9, len(clear)))
        # degradation
        deg = idx[degrade_start[idx] <= d]
        if len(deg):
            progress = (d - degrade_start[deg]) / np.maximum(1, lead[deg])
            s197[deg] += rng.poisson(2 + 20 * progress)
            conv = rng.poisson(1 + 10 * progress)
            s5[deg] += conv
            s197[deg] = np.maximum(0, s197[deg] - conv * 0.5)
            s198[deg] += rng.poisson(0.3 + 4 * progress)
            s187[deg] += rng.poisson(0.2 + 6 * progress)
            s188[deg] += (rng.random(len(deg)) < 0.1 * progress) * rng.integers(1, 5, len(deg))

        temp = np.round(temp_mean[idx] + rng.normal(0, 1.2, len(idx)) + 2 * (degrade_start[idx] <= d)).astype(int)
        poh = poh0[idx] + 24 * (d - install[idx])
        rows = pd.DataFrame(
            {
                "date": day.isoformat(),
                "serial_number": serials[idx],
                "model": [MODELS[i][0] for i in model_idx[idx]],
                "capacity_bytes": [MODELS[i][1] for i in model_idx[idx]],
                "failure": (fail_day[idx] == d).astype(int),
                "smart_1_normalized": np.where(s1_scale[idx] > 0, rng.integers(70, 90, len(idx)), 100),
                "smart_1_raw": (rng.random(len(idx)) * s1_scale[idx]).astype(np.int64),
                "smart_3_normalized": 95,
                "smart_3_raw": 0,
                "smart_4_normalized": 100,
                "smart_4_raw": s12[idx].astype(np.int64),
                "smart_5_normalized": np.clip(100 - (s5[idx] // 10), 1, 100).astype(int),
                "smart_5_raw": s5[idx].astype(np.int64),
                "smart_7_normalized": np.where(s1_scale[idx] > 0, rng.integers(80, 95, len(idx)), 100),
                "smart_7_raw": (rng.random(len(idx)) * s1_scale[idx] / 100).astype(np.int64),
                "smart_9_normalized": np.clip(100 - poh // 8760, 1, 100),
                "smart_9_raw": poh.astype(np.int64),
                "smart_10_normalized": 100,
                "smart_10_raw": 0,
                "smart_12_normalized": 100,
                "smart_12_raw": s12[idx].astype(np.int64),
                "smart_187_normalized": np.clip(100 - s187[idx], 1, 100).astype(int),
                "smart_187_raw": s187[idx].astype(np.int64),
                "smart_188_normalized": 100,
                "smart_188_raw": s188[idx].astype(np.int64),
                "smart_192_normalized": 100,
                "smart_192_raw": (s12[idx] // 2).astype(np.int64),
                "smart_193_normalized": np.clip(100 - s193[idx] // 10000, 1, 100).astype(int),
                "smart_193_raw": s193[idx].astype(np.int64),
                "smart_194_normalized": np.clip(100 - temp, 1, 100),
                "smart_194_raw": temp,
                "smart_196_normalized": 100,
                "smart_196_raw": (s5[idx] // 2).astype(np.int64),
                "smart_197_normalized": np.clip(100 - s197[idx] // 8, 1, 100).astype(int),
                "smart_197_raw": s197[idx].astype(np.int64),
                "smart_198_normalized": np.clip(100 - s198[idx] // 8, 1, 100).astype(int),
                "smart_198_raw": s198[idx].astype(np.int64),
                "smart_199_normalized": 200,
                "smart_199_raw": s199[idx].astype(np.int64),
                "smart_240_raw": (poh * 0.9).astype(np.int64),
                "smart_241_raw": s241[idx].astype(np.int64),
                "smart_242_raw": s242[idx].astype(np.int64),
            }
        )
        if d < drift_day:
            rows = rows.drop(columns=EARLY_ONLY_MISSING)
        else:
            rows["datacenter"] = "sac2"
            rows["cluster_id"] = 10
            rows["vault_id"] = 1200 + (idx % 40)
            rows["pod_id"] = idx % 20
            rows["pod_slot_num"] = idx % 60
            rows["is_legacy_format"] = 0
        # quirks
        if d % 50 == 7 and len(rows) > 2:
            rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)  # duplicate drive-day
            rows.loc[1, "capacity_bytes"] = -1
        if not ghost_written and d > 1:
            failed_yesterday = np.where(fail_day == d - 1)[0]
            if len(failed_yesterday):
                g = failed_yesterday[0]
                ghost = rows.iloc[[0]].copy()
                ghost["serial_number"] = serials[g]
                ghost["model"] = MODELS[model_idx[g]][0]
                ghost["failure"] = 0
                rows = pd.concat([rows, ghost], ignore_index=True)
                ghost_written = True
        folder = os.path.join(out, quarter_dir(day))
        os.makedirs(folder, exist_ok=True)
        rows.to_csv(os.path.join(folder, f"{day.isoformat()}.csv"), index=False)
        stats["files"] += 1
        stats["rows"] += len(rows)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--drives", type=int, default=3000)
    p.add_argument("--days", type=int, default=400)
    p.add_argument("--start", default="2024-01-01")
    p.add_argument(
        "--annual-failure-rate",
        type=float,
        default=0.04,
        help="exaggerated vs the real ~1.5-2 % so a small fleet yields enough failures",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--drift-day",
        type=int,
        default=90,
        help="files before this day lack some columns; files after add datacenter columns",
    )
    a = p.parse_args()
    stats = generate(a.out, a.drives, a.days, date.fromisoformat(a.start), a.annual_failure_rate, a.seed, a.drift_day)
    print(stats)


if __name__ == "__main__":
    main()
