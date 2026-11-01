#!/usr/bin/env python3
"""Prepare a Databricks workspace for DriveGuard and upload the raw subset - using only the SDK (no SQL warehouse
needed for the setup itself).

    python scripts/databricks_bootstrap.py info                       # workspace, warehouses (prints the HTTP path)
    python scripts/databricks_bootstrap.py ensure                     # catalog schema + raw Volume exist
    python scripts/databricks_bootstrap.py upload --src data/raw      # copy new files into the Volume (skips existing)
    python scripts/databricks_bootstrap.py reset                      # drop the tables; next build re-ingests from the Volume

Auth: DATABRICKS_HOST + DATABRICKS_TOKEN (or a CLI profile). Catalog/schema from DG_CATALOG / DG_SCHEMA.
"""

from __future__ import annotations

import argparse
import os
import sys

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType


def client() -> WorkspaceClient:
    return WorkspaceClient()


def info(w: WorkspaceClient, args) -> None:
    me = w.current_user.me()
    print(f"workspace: {w.config.host}\nuser: {me.user_name}")
    http_path = None
    for wh in w.warehouses.list():
        p = wh.odbc_params.path if wh.odbc_params else None
        state = wh.state.value if wh.state else "?"
        print(f"warehouse: {wh.name} ({wh.id}) state={state} http_path={p}")
        http_path = http_path or p
    if http_path:
        print(f"DATABRICKS_HTTP_PATH={http_path}")
        if out := os.environ.get("GITHUB_OUTPUT"):
            with open(out, "a") as fh:
                fh.write(f"http_path={http_path}\n")
    else:
        print("no SQL warehouse found - create one (SQL Warehouses) for the console", file=sys.stderr)


def ensure(w: WorkspaceClient, args) -> None:
    catalog, schema, volume = args.catalog, args.schema, args.volume
    try:
        w.schemas.get(f"{catalog}.{schema}")
        print(f"schema {catalog}.{schema} exists")
    except Exception:
        w.schemas.create(name=schema, catalog_name=catalog)
        print(f"created schema {catalog}.{schema}")
    try:
        w.volumes.read(f"{catalog}.{schema}.{volume}")
        print(f"volume {catalog}.{schema}.{volume} exists")
    except Exception:
        w.volumes.create(catalog_name=catalog, schema_name=schema, name=volume, volume_type=VolumeType.MANAGED)
        print(f"created volume {catalog}.{schema}.{volume}")
    print(f"raw_path=/Volumes/{catalog}/{schema}/{volume}")


def _existing(w: WorkspaceClient, directory: str) -> set[str]:
    found: set[str] = set()

    def walk(d: str) -> None:
        try:
            entries = list(w.files.list_directory_contents(d))
        except Exception:
            return
        for e in entries:
            if e.is_directory:
                walk(e.path)
            else:
                found.add(e.path)

    walk(directory)
    return found


def upload(w: WorkspaceClient, args) -> None:
    root = f"/Volumes/{args.catalog}/{args.schema}/{args.volume}"
    have = _existing(w, root)
    n = skipped = 0
    for dirpath, _, files in os.walk(args.src):
        for f in sorted(files):
            if not f.endswith((".parquet", ".csv")):
                continue
            local = os.path.join(dirpath, f)
            rel = os.path.relpath(local, args.src).replace(os.sep, "/")
            remote = f"{root}/{rel}"
            if remote in have:
                skipped += 1
                continue
            parent = remote.rsplit("/", 1)[0]
            try:
                w.files.create_directory(parent)
            except Exception:
                pass
            with open(local, "rb") as fh:
                w.files.upload(remote, fh, overwrite=True)
            n += 1
            print(f"uploaded {rel}", file=sys.stderr)
    print(f"uploaded {n} files, skipped {skipped} already present, raw_path={root}")


def reset(w: WorkspaceClient, args) -> None:
    """Drop every DriveGuard table so the next build re-ingests from the Volume (e.g. to start a day-by-day replay)."""
    from driveguard.config import TABLES

    for name in TABLES.values():
        full = f"{args.catalog}.{args.schema}.{name}"
        try:
            w.tables.delete(full)
            print(f"dropped {full}")
        except Exception as exc:
            print(f"skip {full}: {str(exc)[:80]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["info", "ensure", "upload", "reset"])
    p.add_argument("--catalog", default=os.environ.get("DG_CATALOG", "workspace"))
    p.add_argument("--schema", default=os.environ.get("DG_SCHEMA", "driveguard"))
    p.add_argument("--volume", default="raw")
    p.add_argument("--src", default="data/raw")
    a = p.parse_args()
    w = client()
    {"info": info, "ensure": ensure, "upload": upload, "reset": reset}[a.cmd](w, a)


if __name__ == "__main__":
    main()
