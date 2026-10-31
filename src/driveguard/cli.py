"""Command line entry point. Each stage is one sub-command, so a Databricks job task, a Makefile target and a
developer at a shell all call exactly the same code."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from driveguard.config import Settings


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, default=str))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser(prog="driveguard", description="Predictive storage failure intelligence pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bronze", help="land raw Backblaze files")
    b.add_argument("--raw-path", default=None)
    sub.add_parser("silver", help="clean and deduplicate; build drive lifecycle")
    sub.add_parser("gold", help="features and 30-day labels")
    sub.add_parser("train", help="train, evaluate, log to MLflow, promote")
    sc = sub.add_parser("score", help="batch score one snapshot day")
    sc.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: latest day in gold)")
    sc.add_argument("--model-uri", default=None)
    bf = sub.add_parser("backfill", help="replay daily scoring over the last N snapshot days")
    bf.add_argument("--days", type=int, default=45)
    bf.add_argument("--model-uri", default=None)
    mo = sub.add_parser("monitor", help="drift, matured performance, retrain decision")
    mo.add_argument("--as-of", default=None)
    mo.add_argument("--model-uri", default=None)
    ra = sub.add_parser("run-all", help="bronze -> silver -> gold -> train -> score -> monitor")
    ra.add_argument("--raw-path", default=None)
    args = p.parse_args(argv)

    s = Settings.from_env()
    from driveguard.spark import get_spark

    spark = get_spark()
    t0 = time.time()
    if args.cmd == "bronze":
        from driveguard import bronze

        _print({"files_ingested": bronze.run(spark, s, args.raw_path)})
    elif args.cmd == "silver":
        from driveguard import silver

        _print(silver.run(spark, s))
    elif args.cmd == "gold":
        from driveguard import gold

        _print(gold.run(spark, s))
    elif args.cmd == "train":
        from driveguard import train

        _print(train.run(spark, s))
    elif args.cmd == "score":
        from driveguard import score

        _print(score.run(spark, s, args.as_of, args.model_uri))
    elif args.cmd == "backfill":
        from driveguard import score

        _print(score.backfill(spark, s, args.days, args.model_uri))
    elif args.cmd == "monitor":
        from driveguard import monitor

        _print(monitor.run(spark, s, args.as_of, args.model_uri))
    elif args.cmd == "run-all":
        from driveguard import bronze, gold, monitor, score, silver, train

        out = {"bronze_files": bronze.run(spark, s, args.raw_path)}
        out["silver"] = silver.run(spark, s)
        out["gold"] = gold.run(spark, s)
        out["train"] = train.run(spark, s)
        out["backfill"] = score.backfill(spark, s, s.horizon_days + 15)
        out["monitor"] = monitor.run(spark, s)
        _print(out)
    logging.getLogger("driveguard").info("%s finished in %.1fs", args.cmd, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
