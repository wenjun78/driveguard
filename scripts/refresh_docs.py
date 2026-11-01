#!/usr/bin/env python
"""Keep the numbers in the written docs equal to the numbers of the model that is scoring the fleet.

Every figure in README.md and PROJECT_CASE_STUDY.md that comes from the champion's evaluation is wrapped in an
invisible marker pair, e.g. ``<!--dg:test.drive_recall0-->71<!--/dg-->``. This script recomputes each value from
the console snapshot (``app_snapshot/model.json`` and friends — the same export the console serves) and rewrites
the text between the markers. The ``daily`` workflow runs it whenever the champion version changes and commits
the result, so the prose can never describe a model that is no longer the champion.

    python scripts/refresh_docs.py            # rewrite in place
    python scripts/refresh_docs.py --check    # exit 1 if anything would change (CI)
    python scripts/refresh_docs.py --print    # show every key and its current value
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "app_snapshot"
FILES = [ROOT / "README.md", ROOT / "PROJECT_CASE_STUDY.md"]
MARK = re.compile(r"<!--dg:([A-Za-z0-9_.]+)-->(.*?)<!--/dg-->", re.S)
EPOCH = date(1970, 1, 1)


def _day(n: str | int) -> date:
    return EPOCH + timedelta(days=int(n))


def _dmy(d: date) -> str:
    return f"{d.day} {d.strftime('%b %Y')}"


def _pct(x: float | None, nd: int) -> str:
    return "n/a" if x is None else f"{100 * x:.{nd}f}"


def _num(x: float | None, nd: int) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _int(x: float | None) -> str:
    return "n/a" if x is None else f"{math.floor(x + 0.5):,}"  # half-up, same as the portfolio's JS


def split_values(prefix: str, m: dict) -> dict[str, str]:
    g = m.get
    rows = g("rows_labeled")
    pos = g("positives")
    return {
        f"{prefix}.pr_auc": _num(g("pr_auc"), 3),
        f"{prefix}.roc_auc": _num(g("roc_auc"), 3),
        f"{prefix}.roc_auc2": _num(g("roc_auc"), 2),
        f"{prefix}.brier": _num(g("brier"), 4),
        f"{prefix}.drive_recall": _pct(g("drive_recall_at_budget"), 1),
        f"{prefix}.drive_recall0": _pct(g("drive_recall_at_budget"), 0),
        f"{prefix}.recall": _pct(g("recall_at_budget"), 1),
        f"{prefix}.recall0": _pct(g("recall_at_budget"), 0),
        f"{prefix}.precision": _pct(g("precision_at_budget"), 1),
        f"{prefix}.lead_median": _int(g("lead_time_median_days")),
        f"{prefix}.caught_7d": _pct(g("share_caught_with_7d_lead"), 0),
        f"{prefix}.false_alerts": _num(g("false_alerts_per_1000_drive_days"), 2),
        f"{prefix}.failing_drives": _int(g("failing_drives")),
        f"{prefix}.rows_labeled": _int(rows),
        f"{prefix}.rows_labeled_m": "n/a" if rows is None else f"{rows / 1e6:.2f}",
        f"{prefix}.positives": _int(pos),
        f"{prefix}.base_rate": "n/a" if not rows or pos is None else f"{100 * pos / rows:.2f}",
        f"{prefix}.alerts_per_day": _int(g("alerts_per_day")),
    }


def values(snapshot: Path = SNAPSHOT) -> dict[str, str]:
    model = json.loads((snapshot / "model.json").read_text())
    report = model.get("report", {})
    out: dict[str, str] = {"champion.version": str(model.get("champion", {}).get("version", "?"))}
    out.update(split_values("val", report.get("val", {})))
    out.update(split_values("test", report.get("test", {})))
    out.update(split_values("rule", report.get("baseline_test", {})))

    sp = model.get("split", {})
    if sp:
        test_start, end = _day(sp["test_start"]), _day(sp["end_day"])
        out.update(
            {
                "split.train_end": _dmy(_day(sp["train_end"])),
                "split.val_start": _dmy(_day(sp["val_start"])),
                "split.val_end": _dmy(_day(sp["val_end"])),
                "split.test_start": _dmy(test_start),
                "split.test_end": _dmy(end),
                "split.test_days": str((end - test_start).days + 1),
                "split.gap": str(sp.get("gap", "")),
            }
        )
    top = [f for f, _ in model.get("importance", [])[:6]]
    out["importance.top6"] = ", ".join(f"`{f}`" for f in top)

    test_recall = report.get("test", {}).get("recall_at_budget")
    out["live.floor"] = _pct(0.5 * test_recall, 0) if test_recall else "n/a"

    overview = snapshot / "overview.json"
    if overview.exists():
        o = json.loads(overview.read_text())
        out["data.drives"] = _int(o.get("fleet_drives_ever"))
        out["data.failures"] = _int(o.get("fleet_failed_ever"))
        out["data.scored_per_day"] = _int(o.get("drives_scored"))
        if o.get("as_of"):
            out["data.as_of"] = _dmy(date.fromisoformat(o["as_of"]))

    monitoring = snapshot / "monitoring.json"
    if monitoring.exists():
        mat = [r.get("recall_at_budget") for r in json.loads(monitoring.read_text()).get("matured", [])]
        mat = [float(x) for x in mat if x is not None]
        if mat:
            out["live.matured_range"] = f"{min(mat):.1f}–{max(mat):.1f} per day (median ≈ {statistics.median(mat):.1f})"
    return out


def render(text: str, vals: dict[str, str]) -> str:
    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in vals:
            raise KeyError(f"unknown doc marker dg:{key}")
        return f"<!--dg:{key}-->{vals[key]}<!--/dg-->"

    return MARK.sub(sub, text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if any file would change")
    ap.add_argument("--print", action="store_true", help="print all values")
    ap.add_argument("--snapshot", default=str(SNAPSHOT))
    ap.add_argument("files", nargs="*", default=[str(f) for f in FILES])
    a = ap.parse_args()

    vals = values(Path(a.snapshot))
    if a.print:
        for k, v in sorted(vals.items()):
            print(f"{k:28s} {v}")
        return 0
    changed = []
    for f in map(Path, a.files):
        before = f.read_text()
        after = render(before, vals)
        if after != before:
            changed.append(f)
            if not a.check:
                f.write_text(after)
    if changed:
        print(("would update: " if a.check else "updated: ") + ", ".join(p.name for p in changed))
        return 1 if a.check else 0
    print("docs already match the champion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
