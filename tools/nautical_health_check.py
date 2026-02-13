#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operational health check for Nautical queue/dead-letter state."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _safe_read_lines(path: Path) -> int:
    try:
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return -1


def _safe_stat(path: Path) -> tuple[int, float]:
    try:
        if not path.exists():
            return 0, 0.0
        st = path.stat()
        return int(st.st_size), float(st.st_mtime)
    except Exception:
        return -1, 0.0


def _safe_lock_fail_count(path: Path) -> int:
    try:
        if not path.exists():
            return 0
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw or "{}")
        return int(obj.get("count") or 0)
    except Exception:
        return -1


def _safe_last_ts(path: Path) -> str:
    try:
        if not path.exists():
            return ""
        last = ""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    last = ln
        if not last:
            return ""
        obj = json.loads(last)
        ts = obj.get("ts")
        return str(ts) if ts else ""
    except Exception:
        return ""


def _add_check(checks: list[dict], name: str, value: int | float, warn: int | float, crit: int | float) -> None:
    if value < 0:
        checks.append(
            {
                "name": name,
                "value": value,
                "status": "warn",
                "message": "unreadable",
            }
        )
        return
    if value >= crit:
        status = "crit"
    elif value >= warn:
        status = "warn"
    else:
        status = "ok"
    checks.append({"name": name, "value": value, "status": status, "warn": warn, "crit": crit})


def main() -> int:
    ap = argparse.ArgumentParser(description="Nautical queue/dead-letter health check")
    ap.add_argument("--taskdata", default=os.environ.get("TASKDATA", "~/.task"), help="Taskwarrior data dir")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    ap.add_argument("--queue-warn-bytes", type=int, default=262144)
    ap.add_argument("--queue-crit-bytes", type=int, default=524288)
    ap.add_argument("--processing-warn-lines", type=int, default=1)
    ap.add_argument("--processing-crit-lines", type=int, default=100)
    ap.add_argument("--dead-letter-warn-lines", type=int, default=1)
    ap.add_argument("--dead-letter-crit-lines", type=int, default=10)
    ap.add_argument("--lock-fail-warn-count", type=int, default=1)
    ap.add_argument("--lock-fail-crit-count", type=int, default=10)
    ap.add_argument("--queue-stale-warn-seconds", type=int, default=300)
    ap.add_argument("--queue-stale-crit-seconds", type=int, default=1800)
    args = ap.parse_args()

    td = Path(args.taskdata).expanduser().resolve()
    queue = td / ".nautical_spawn_queue.jsonl"
    processing = td / ".nautical_spawn_queue.processing.jsonl"
    dead = td / ".nautical_dead_letter.jsonl"
    bad = td / ".nautical_spawn_queue.bad.jsonl"
    lock_fail_count = td / ".nautical_spawn_queue.lock_failed.count"
    diag_log = td / ".nautical_diag.jsonl"

    queue_bytes, queue_mtime = _safe_stat(queue)
    processing_bytes, _ = _safe_stat(processing)
    dead_bytes, _ = _safe_stat(dead)
    bad_bytes, _ = _safe_stat(bad)
    queue_lines = _safe_read_lines(queue)
    processing_lines = _safe_read_lines(processing)
    dead_lines = _safe_read_lines(dead)
    bad_lines = _safe_read_lines(bad)
    lock_fail = _safe_lock_fail_count(lock_fail_count)

    now = time.time()
    queue_age_s = 0
    if queue_bytes > 0 and queue_mtime > 0:
        queue_age_s = max(0, int(now - queue_mtime))

    checks: list[dict] = []
    _add_check(checks, "queue_bytes", queue_bytes, args.queue_warn_bytes, args.queue_crit_bytes)
    _add_check(checks, "processing_lines", processing_lines, args.processing_warn_lines, args.processing_crit_lines)
    _add_check(checks, "dead_letter_lines", dead_lines, args.dead_letter_warn_lines, args.dead_letter_crit_lines)
    _add_check(checks, "lock_fail_count", lock_fail, args.lock_fail_warn_count, args.lock_fail_crit_count)
    _add_check(checks, "queue_age_s", queue_age_s, args.queue_stale_warn_seconds, args.queue_stale_crit_seconds)

    status = "ok"
    for chk in checks:
        st = chk.get("status")
        if st == "crit":
            status = "crit"
            break
        if st == "warn" and status == "ok":
            status = "warn"

    payload = {
        "status": status,
        "taskdata": str(td),
        "metrics": {
            "queue_bytes": queue_bytes,
            "queue_lines": queue_lines,
            "queue_age_s": queue_age_s,
            "processing_bytes": processing_bytes,
            "processing_lines": processing_lines,
            "dead_letter_bytes": dead_bytes,
            "dead_letter_lines": dead_lines,
            "bad_queue_bytes": bad_bytes,
            "bad_queue_lines": bad_lines,
            "lock_fail_count": lock_fail,
            "last_dead_letter_ts": _safe_last_ts(dead),
            "diag_log_enabled_hint": "set NAUTICAL_DIAG_LOG=1 to persist hook diagnostics",
            "diag_log_path": str(diag_log),
        },
        "checks": checks,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        print(f"status={payload['status']} taskdata={payload['taskdata']}")
        for k, v in payload["metrics"].items():
            print(f"{k}={v}")
        print("checks:")
        for chk in checks:
            name = chk.get("name")
            st = chk.get("status")
            value = chk.get("value")
            print(f"  - {name}: {st} (value={value})")

    if status == "crit":
        return 2
    if status == "warn":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
