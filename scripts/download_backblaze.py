#!/usr/bin/env python3
"""Download Backblaze Drive Stats quarters and cut a deterministic subset for the pipeline.

Why a subset: one quarter is ~300k drives x ~90 days x 100+ columns - several GB of CSV. Free Edition compute and a
laptop both prefer to start smaller. The subset keeps *whole drive histories* (hash-sampled by serial number), never
random rows, so lifecycles, labels and censoring stay intact; it also keeps only the SMART columns the pipeline uses.
The same hash picks the same drives in every quarter, so adding quarters later extends the same fleet.

Usage (laptop, then upload ``data/raw`` to a Unity Catalog Volume):

    python scripts/download_backblaze.py --quarters 2024Q3 2024Q4 2025Q1 --serial-frac 0.10 --out data/raw
    python scripts/download_backblaze.py --zip ~/Downloads/data_Q1_2025.zip --serial-frac 0.10 --out data/raw

The quarterly zip names changed over the years; ``--url`` overrides the guess. Browse the catalogue at
https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data (data is free to use with attribution).
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import zipfile
import zlib

import pandas as pd

BASE = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/"
KEEP_SMART = [5, 187, 188, 197, 198, 1, 3, 4, 7, 9, 10, 12, 192, 193, 194, 196, 199, 240, 241, 242]
KEEP_COLUMNS = (
    ["date", "serial_number", "model", "capacity_bytes", "failure"]
    + [f"smart_{n}_raw" for n in KEEP_SMART]
    + [f"smart_{n}_normalized" for n in (5, 187, 197, 198, 194)]
)


def candidate_urls(quarter: str) -> list[str]:
    m = re.fullmatch(r"(\d{4})Q([1-4])", quarter.upper())
    if not m:
        raise SystemExit(f"quarter must look like 2024Q3, got {quarter}")
    y, q = m.group(1), m.group(2)
    return [f"{BASE}data_Q{q}_{y}.zip", f"{BASE}data_{y}_Q{q}.zip"]


def keep_serial(serial: str, frac: float) -> bool:
    return (zlib.crc32(serial.encode()) % 10_000) < int(frac * 10_000)


def download(url: str, dest: str) -> str:
    import requests

    if os.path.exists(dest):
        print(f"exists, skipping download: {dest}")
        return dest
    with requests.get(url, stream=True, timeout=60) as r:
        if r.status_code != 200:
            raise FileNotFoundError(f"{url} -> HTTP {r.status_code}")
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest + ".part", "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{dest}: {done / total:6.1%}", end="", file=sys.stderr)
        print(file=sys.stderr)
    os.replace(dest + ".part", dest)
    return dest


def extract_subset(zip_path: str, out: str, serial_frac: float, start: str | None, end: str | None) -> dict:
    stats = {"files": 0, "rows_in": 0, "rows_out": 0}
    with zipfile.ZipFile(zip_path) as z:
        members = sorted(n for n in z.namelist() if n.lower().endswith(".csv") and "__MACOSX" not in n)
        for name in members:
            day = os.path.splitext(os.path.basename(name))[0]
            if (start and day < start) or (end and day > end):
                continue
            with z.open(name) as fh:
                header = pd.read_csv(io.BytesIO(fh.read(1 << 16)), nrows=0).columns.tolist()
            usecols = [c for c in KEEP_COLUMNS if c in header]
            with z.open(name) as fh:
                df = pd.read_csv(fh, usecols=usecols, dtype={"serial_number": str, "model": str}, low_memory=False)
            stats["rows_in"] += len(df)
            if serial_frac < 1.0:
                df = df[df["serial_number"].map(lambda s: keep_serial(str(s), serial_frac))]
            for c in usecols:
                if c.startswith("smart_") or c in ("capacity_bytes", "failure"):
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            folder = os.path.join(out, os.path.splitext(os.path.basename(zip_path))[0])
            os.makedirs(folder, exist_ok=True)
            df.to_parquet(os.path.join(folder, f"{day}.parquet"), index=False)
            stats["files"] += 1
            stats["rows_out"] += len(df)
            print(f"{day}: {len(df):>8,} of {stats['rows_in']:,} rows kept so far", file=sys.stderr)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quarters", nargs="*", default=[], help="e.g. 2024Q3 2024Q4")
    p.add_argument("--url", nargs="*", default=[], help="explicit zip URLs (override the name guess)")
    p.add_argument("--zip", nargs="*", default=[], help="already-downloaded zip files")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--cache", default="data/zips", help="where downloaded zips are kept")
    p.add_argument(
        "--serial-frac", type=float, default=0.10, help="share of drives to keep (by serial hash); 1.0 = all"
    )
    p.add_argument("--start", default=None, help="first day to keep, YYYY-MM-DD")
    p.add_argument("--end", default=None, help="last day to keep, YYYY-MM-DD")
    a = p.parse_args()
    os.makedirs(a.cache, exist_ok=True)

    zips = list(a.zip)
    for url in a.url:
        zips.append(download(url, os.path.join(a.cache, os.path.basename(url))))
    for q in a.quarters:
        last_err = None
        for url in candidate_urls(q):
            try:
                zips.append(download(url, os.path.join(a.cache, os.path.basename(url))))
                break
            except FileNotFoundError as exc:
                last_err = exc
        else:
            raise SystemExit(
                f"could not find a zip for {q}: {last_err}. Pass --url with the exact link from the Backblaze page."
            )
    if not zips:
        raise SystemExit("nothing to do: give --quarters, --url or --zip")
    for zp in zips:
        print(f"== {zp}", file=sys.stderr)
        print(extract_subset(zp, a.out, a.serial_frac, a.start, a.end))


if __name__ == "__main__":
    main()
