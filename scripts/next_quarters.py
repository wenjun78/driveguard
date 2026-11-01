#!/usr/bin/env python3
"""Print the Backblaze quarters that come after the latest one already in the Volume, up to the last complete quarter.

    python scripts/next_quarters.py            -> e.g. "2025Q2 2025Q3 2025Q4 2026Q1 2026Q2"

Used by the fetch-quarters workflow; quarters Backblaze has not published yet are skipped by the downloader.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import date

from databricks.sdk import WorkspaceClient


def loaded_quarters(w: WorkspaceClient, root: str) -> list[tuple[int, int]]:
    out = []
    try:
        for e in w.files.list_directory_contents(root):
            m = re.search(r"data_Q([1-4])_(\d{4})|data_(\d{4})_Q([1-4])", e.path)
            if m:
                q, y = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
                out.append((int(y), int(q)))
    except Exception as exc:
        print(f"could not list {root}: {exc}", file=sys.stderr)
    return sorted(set(out))


def main() -> None:
    catalog, schema = os.environ.get("DG_CATALOG", "workspace"), os.environ.get("DG_SCHEMA", "driveguard")
    root = f"/Volumes/{catalog}/{schema}/raw"
    have = loaded_quarters(WorkspaceClient(), root)
    if not have:
        print("", end="")
        return
    y, q = have[-1]
    today = date.today()
    last_complete = (today.year, (today.month - 1) // 3) if today.month > 3 else (today.year - 1, 4)
    wanted = []
    while True:
        q += 1
        if q == 5:
            q, y = 1, y + 1
        if (y, q) > last_complete:
            break
        wanted.append(f"{y}Q{q}")
    print(" ".join(wanted))


if __name__ == "__main__":
    main()
