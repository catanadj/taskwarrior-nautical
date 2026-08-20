#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only operational health check for the Nautical lifecycle outbox."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path


OUTBOX_NAME = ".nautical_lifecycle_outbox.db"


def _safe_stat(path: Path) -> tuple[int, float]:
    try:
        if not path.exists():
            return 0, 0.0
        stat = path.stat()
        return int(stat.st_size), float(stat.st_mtime)
    except OSError:
        return -1, 0.0


def _sum_sizes(*values: int) -> int:
    return -1 if any(value < 0 for value in values) else sum(values)


def _outbox_rows(path: Path, now: float) -> dict[str, int]:
    empty = {
        "schema_version": 0,
        "rows": 0,
        "ready": 0,
        "retry": 0,
        "claimed": 0,
        "expired_claims": 0,
        "manual_review": 0,
        "quarantined": 0,
        "acknowledged": 0,
    }
    try:
        if not path.exists():
            return empty
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.5) as conn:
            schema_version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
            states = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    "SELECT processing_state, COUNT(*) FROM lifecycle_outbox GROUP BY processing_state"
                )
            }
            expired_claims = int(
                conn.execute(
                    "SELECT COUNT(*) FROM lifecycle_outbox "
                    "WHERE processing_state='claimed' AND lease_expires_at <= ?",
                    (now,),
                ).fetchone()[0]
                or 0
            )
    except (OSError, sqlite3.Error):
        return {key: -1 for key in empty}
    return {
        "schema_version": schema_version,
        "rows": sum(states.values()),
        "ready": states.get("ready", 0),
        "retry": states.get("retry", 0),
        "claimed": states.get("claimed", 0),
        "expired_claims": expired_claims,
        "manual_review": states.get("manual_review", 0),
        "quarantined": states.get("quarantined", 0),
        "acknowledged": states.get("acknowledged", 0),
    }


def _add_check(checks: list[dict[str, object]], name: str, value: int, warn: int, crit: int) -> None:
    if value < 0:
        checks.append({"name": name, "value": value, "status": "warn", "message": "unreadable"})
        return
    status = "crit" if value >= crit else "warn" if value >= warn else "ok"
    checks.append({"name": name, "value": value, "status": status, "warn": warn, "crit": crit})


def main() -> int:
    parser = argparse.ArgumentParser(description="Nautical lifecycle outbox health check")
    parser.add_argument("--taskdata", default=os.environ.get("TASKDATA", "~/.task"), help="Taskwarrior data dir")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--outbox-warn-bytes", type=int, default=262144)
    parser.add_argument("--outbox-crit-bytes", type=int, default=524288)
    parser.add_argument("--outbox-warn-rows", type=int, default=100)
    parser.add_argument("--outbox-crit-rows", type=int, default=1000)
    parser.add_argument("--claimed-warn-rows", type=int, default=1)
    parser.add_argument("--claimed-crit-rows", type=int, default=100)
    parser.add_argument("--quarantine-warn-rows", type=int, default=1)
    parser.add_argument("--quarantine-crit-rows", type=int, default=10)
    args = parser.parse_args()

    taskdata = Path(args.taskdata).expanduser().resolve()
    outbox = taskdata / ".nautical-state" / OUTBOX_NAME
    wal = outbox.with_name(f"{OUTBOX_NAME}-wal")
    shm = outbox.with_name(f"{OUTBOX_NAME}-shm")
    main_bytes, modified_at = _safe_stat(outbox)
    wal_bytes, _ = _safe_stat(wal)
    shm_bytes, _ = _safe_stat(shm)
    outbox_bytes = _sum_sizes(main_bytes, wal_bytes, shm_bytes)
    now = time.time()
    rows = _outbox_rows(outbox, now)
    age_s = max(0, int(now - modified_at)) if rows["rows"] > 0 and modified_at > 0 else 0

    checks: list[dict[str, object]] = []
    _add_check(checks, "outbox_bytes", outbox_bytes, args.outbox_warn_bytes, args.outbox_crit_bytes)
    _add_check(checks, "outbox_rows", rows["rows"], args.outbox_warn_rows, args.outbox_crit_rows)
    _add_check(checks, "expired_claims", rows["expired_claims"], args.claimed_warn_rows, args.claimed_crit_rows)
    _add_check(checks, "quarantined_rows", rows["quarantined"], args.quarantine_warn_rows, args.quarantine_crit_rows)
    status = "crit" if any(check["status"] == "crit" for check in checks) else "warn" if any(
        check["status"] == "warn" for check in checks
    ) else "ok"
    payload = {
        "status": status,
        "taskdata": str(taskdata),
        "outbox": {
            "path": str(outbox),
            "bytes": outbox_bytes,
            "main_bytes": main_bytes,
            "wal_bytes": wal_bytes,
            "shm_bytes": shm_bytes,
            "age_s": age_s,
            **rows,
        },
        "checks": checks,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        print(f"status={status} taskdata={taskdata}")
        for key, value in payload["outbox"].items():
            print(f"{key}={value}")
        print("checks:")
        for check in checks:
            print(f"  - {check['name']}: {check['status']} (value={check['value']})")
    return 2 if status == "crit" else 1 if status == "warn" else 0


if __name__ == "__main__":
    raise SystemExit(main())
