#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detailed read-only Nautical queue/state inspector."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path


def _prefer_path(primary: Path, legacy: Path) -> Path:
    try:
        if primary.exists():
            return primary
    except Exception:
        pass
    try:
        if legacy.exists():
            return legacy
    except Exception:
        pass
    return primary


def _safe_stat(path: Path) -> dict[str, object]:
    try:
        exists = path.exists()
    except Exception:
        return {"path": str(path), "exists": False, "bytes": -1, "mtime": 0.0, "error": "unreadable"}
    if not exists:
        return {"path": str(path), "exists": False, "bytes": 0, "mtime": 0.0}
    try:
        st = path.stat()
        return {"path": str(path), "exists": True, "bytes": int(st.st_size), "mtime": float(st.st_mtime)}
    except Exception:
        return {"path": str(path), "exists": True, "bytes": -1, "mtime": 0.0, "error": "unreadable"}


def _safe_line_count(path: Path) -> int:
    try:
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return -1


def _safe_json_count(path: Path, field: str) -> int:
    try:
        if not path.exists():
            return 0
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
        return int(data.get(field) or 0)
    except Exception:
        return -1


def _safe_glob_count(path: Path, pattern: str) -> int:
    try:
        if not path.exists():
            return 0
        return sum(1 for _ in path.glob(pattern))
    except Exception:
        return -1


def _safe_sqlite_summary(path: Path, stale_after: float, limit: int) -> tuple[dict[str, object], list[str]]:
    summary: dict[str, object] = {
        "exists": False,
        "queued": 0,
        "processing": 0,
        "stale_processing": 0,
        "max_attempts": 0,
        "oldest_claimed_age_s": 0,
        "sample": [],
    }
    issues: list[str] = []
    try:
        if not path.exists():
            return summary, issues
    except Exception:
        summary["error"] = "unreadable"
        issues.append("queue db unreadable")
        return summary, issues

    summary["exists"] = True
    conn = None
    now = time.time()
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.5)
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN state='queued' THEN 1 ELSE 0 END), 0) AS queued_rows,
                COALESCE(SUM(CASE WHEN state='processing' THEN 1 ELSE 0 END), 0) AS processing_rows,
                COALESCE(SUM(CASE WHEN state='processing' AND claimed_at IS NOT NULL AND claimed_at < ? THEN 1 ELSE 0 END), 0) AS stale_processing_rows,
                COALESCE(MAX(attempts), 0) AS max_attempts,
                MIN(CASE WHEN state='processing' AND claimed_at IS NOT NULL THEN claimed_at END) AS min_claimed_at
            FROM queue_entries
            """,
            (now - stale_after,),
        ).fetchone()
        if row:
            queued = int(row[0] or 0)
            processing = int(row[1] or 0)
            stale_processing = int(row[2] or 0)
            max_attempts = int(row[3] or 0)
            min_claimed_at = float(row[4] or 0.0)
            summary.update(
                {
                    "queued": queued,
                    "processing": processing,
                    "stale_processing": stale_processing,
                    "max_attempts": max_attempts,
                    "oldest_claimed_age_s": max(0, int(now - min_claimed_at)) if min_claimed_at > 0 else 0,
                }
            )
            if queued > 0:
                issues.append(f"{queued} queued sqlite entries")
            if processing > 0:
                issues.append(f"{processing} processing sqlite entries")
            if stale_processing > 0:
                issues.append(f"{stale_processing} stale sqlite processing entries")

        sample = []
        for r in conn.execute(
            """
            SELECT id, spawn_intent_id, state, attempts, claimed_at
            FROM queue_entries
            ORDER BY attempts DESC, updated_at ASC, id ASC
            LIMIT ?
            """,
            (max(0, int(limit)),),
        ):
            claimed_at = float(r[4] or 0.0)
            sample.append(
                {
                    "id": int(r[0]),
                    "spawn_intent_id": str(r[1] or ""),
                    "state": str(r[2] or ""),
                    "attempts": int(r[3] or 0),
                    "claimed_age_s": max(0, int(now - claimed_at)) if claimed_at > 0 else 0,
                }
            )
        summary["sample"] = sample
        return summary, issues
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return summary, issues
        summary["error"] = str(e)
        issues.append(f"queue db error: {e}")
        return summary, issues
    except Exception as e:
        summary["error"] = str(e)
        issues.append(f"queue db error: {e}")
        return summary, issues
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def _status_payload(taskdata: Path, *, stale_after: float, limit: int) -> dict[str, object]:
    state_dir = taskdata / ".nautical-state"
    lock_dir = taskdata / ".nautical-locks"

    queue_jsonl = _prefer_path(state_dir / ".nautical_spawn_queue.jsonl", taskdata / ".nautical_spawn_queue.jsonl")
    processing_jsonl = _prefer_path(
        state_dir / ".nautical_spawn_queue.processing.jsonl",
        taskdata / ".nautical_spawn_queue.processing.jsonl",
    )
    queue_db = _prefer_path(state_dir / ".nautical_queue.db", taskdata / ".nautical_queue.db")
    queue_db_wal = state_dir / ".nautical_queue.db-wal" if queue_db.parent == state_dir else taskdata / ".nautical_queue.db-wal"
    queue_db_shm = state_dir / ".nautical_queue.db-shm" if queue_db.parent == state_dir else taskdata / ".nautical_queue.db-shm"
    dead_letter = _prefer_path(state_dir / ".nautical_dead_letter.jsonl", taskdata / ".nautical_dead_letter.jsonl")
    quarantine = _prefer_path(state_dir / ".nautical_spawn_queue.bad.jsonl", taskdata / ".nautical_spawn_queue.bad.jsonl")
    intent_log = _prefer_path(state_dir / ".nautical_spawn_intents.jsonl", taskdata / ".nautical_spawn_intents.jsonl")
    queue_lock_fail_count = _prefer_path(
        lock_dir / ".nautical_spawn_queue.lock_failed.count",
        taskdata / ".nautical_spawn_queue.lock_failed.count",
    )
    queue_lock_fail_marker = _prefer_path(
        lock_dir / ".nautical_spawn_queue.lock_failed",
        taskdata / ".nautical_spawn_queue.lock_failed",
    )

    files = {
        "queue_jsonl": {**_safe_stat(queue_jsonl), "lines": _safe_line_count(queue_jsonl)},
        "processing_jsonl": {**_safe_stat(processing_jsonl), "lines": _safe_line_count(processing_jsonl)},
        "queue_db": {
            **_safe_stat(queue_db),
            "wal_bytes": int(_safe_stat(queue_db_wal)["bytes"]),
            "shm_bytes": int(_safe_stat(queue_db_shm)["bytes"]),
        },
        "dead_letter": {**_safe_stat(dead_letter), "lines": _safe_line_count(dead_letter)},
        "quarantine": {**_safe_stat(quarantine), "lines": _safe_line_count(quarantine)},
        "intent_log": {**_safe_stat(intent_log), "lines": _safe_line_count(intent_log)},
    }

    queue, issues = _safe_sqlite_summary(queue_db, stale_after, limit)
    locks = {
        "queue_lock_failure_count": _safe_json_count(queue_lock_fail_count, "count"),
        "queue_lock_failure_marker": bool(_safe_stat(queue_lock_fail_marker).get("exists")),
        "all_lock_files": _safe_glob_count(lock_dir, "*.lock"),
        "parent_nextlink_lock_files": _safe_glob_count(lock_dir, ".nautical_parent_nextlink.*.lock"),
    }

    if int(files["dead_letter"]["lines"]) > 0:
        issues.append(f"{files['dead_letter']['lines']} dead-letter lines")
    if int(files["quarantine"]["lines"]) > 0:
        issues.append(f"{files['quarantine']['lines']} quarantine lines")
    if int(locks["queue_lock_failure_count"]) > 0:
        issues.append(f"{locks['queue_lock_failure_count']} recorded queue lock failures")
    if locks["queue_lock_failure_marker"]:
        issues.append("queue lock failure marker present")

    status = "warn" if issues else "ok"
    return {
        "status": status,
        "taskdata": str(taskdata),
        "paths": {
            "state_dir": str(state_dir),
            "lock_dir": str(lock_dir),
        },
        "queue": queue,
        "files": files,
        "locks": locks,
        "issues": issues,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Detailed Nautical queue/state inspector")
    ap.add_argument("--taskdata", default=os.environ.get("TASKDATA", "~/.task"), help="Taskwarrior data dir")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    ap.add_argument("--limit", type=int, default=5, help="number of sample sqlite rows to include")
    ap.add_argument(
        "--stale-after-seconds",
        type=float,
        default=float(os.environ.get("NAUTICAL_QUEUE_PROCESSING_STALE_AFTER") or 300.0),
        help="processing claim age considered stale",
    )
    args = ap.parse_args()

    td = Path(args.taskdata).expanduser().resolve()
    payload = _status_payload(td, stale_after=max(0.0, float(args.stale_after_seconds)), limit=max(0, int(args.limit)))

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        print(f"status={payload['status']} taskdata={payload['taskdata']}")
        queue = payload["queue"]
        print(
            "queue:"
            f" queued={queue.get('queued', 0)}"
            f" processing={queue.get('processing', 0)}"
            f" stale_processing={queue.get('stale_processing', 0)}"
            f" max_attempts={queue.get('max_attempts', 0)}"
        )
        files = payload["files"]
        print(
            "files:"
            f" dead_letter_lines={files['dead_letter'].get('lines', 0)}"
            f" quarantine_lines={files['quarantine'].get('lines', 0)}"
            f" queue_db_bytes={files['queue_db'].get('bytes', 0)}"
        )
        locks = payload["locks"]
        print(
            "locks:"
            f" queue_lock_failure_count={locks.get('queue_lock_failure_count', 0)}"
            f" parent_nextlink_lock_files={locks.get('parent_nextlink_lock_files', 0)}"
        )
        issues = payload.get("issues") or []
        if issues:
            print("issues:")
            for issue in issues:
                print(f"  - {issue}")
        sample = queue.get("sample") or []
        if sample:
            print("sample:")
            for row in sample:
                print(
                    "  - "
                    f"id={row.get('id')} sid={row.get('spawn_intent_id') or '-'} "
                    f"state={row.get('state')} attempts={row.get('attempts')} "
                    f"claimed_age_s={row.get('claimed_age_s')}"
                )

    return 1 if payload["status"] == "warn" else 0


if __name__ == "__main__":
    raise SystemExit(main())
