#!/usr/bin/env python3
"""Read-only lifecycle outbox inspector used by Nautical operators and doctor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nautical_core.lifecycle_outbox import OUTBOX_SCHEMA_VERSION, lifecycle_outbox_path  # noqa: E402


_JSON_SCHEMA = "nautical.lifecycle_outbox_status"
_JSON_SCHEMA_VERSION = 1


def _safe_stat(path: Path) -> dict[str, object]:
    try:
        if not path.exists():
            return {"path": str(path), "exists": False, "bytes": 0, "mtime": 0.0}
        stat = path.stat()
        return {"path": str(path), "exists": True, "bytes": int(stat.st_size), "mtime": float(stat.st_mtime)}
    except OSError as exc:
        return {"path": str(path), "exists": False, "bytes": -1, "mtime": 0.0, "error": str(exc)}


def _outbox_summary(path: Path, *, stale_after: float, limit: int) -> tuple[dict[str, Any], list[str]]:
    summary: dict[str, Any] = {
        "exists": False,
        "schema": {"status": "absent", "version": 0, "expected_version": OUTBOX_SCHEMA_VERSION},
        "integrity": "not_checked",
        "states": {},
        "stale_claims": 0,
        "max_attempts": 0,
        "sample": [],
    }
    issues: list[str] = []
    if not path.exists():
        return summary, issues
    summary["exists"] = True
    conn: sqlite3.Connection | None = None
    now = time.time()
    try:
        conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=0.5)
        integrity_row = conn.execute("PRAGMA quick_check").fetchone()
        integrity = str(integrity_row[0] if integrity_row else "unknown")
        summary["integrity"] = integrity
        if integrity.lower() != "ok":
            issues.append(f"lifecycle outbox integrity check failed: {integrity}")
        version_row = conn.execute("PRAGMA user_version").fetchone()
        version = int(version_row[0] if version_row else 0)
        schema = summary["schema"]
        schema["version"] = version
        if version != OUTBOX_SCHEMA_VERSION:
            schema["status"] = "error"
            issues.append(
                f"lifecycle outbox schema v{version} is incompatible with v{OUTBOX_SCHEMA_VERSION}"
            )
            return summary, issues
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(lifecycle_outbox)")}
        required = {"intent_id", "processing_state", "lifecycle_stage", "attempts", "lease_expires_at", "failure_json"}
        missing = sorted(required - columns)
        if missing:
            schema["status"] = "error"
            issues.append(f"lifecycle outbox schema missing {', '.join(missing)}")
            return summary, issues
        schema["status"] = "ok"
        states = {
            str(row[0]): int(row[1])
            for row in conn.execute("SELECT processing_state, COUNT(*) FROM lifecycle_outbox GROUP BY processing_state")
        }
        summary["states"] = states
        stale = int(
            conn.execute(
                "SELECT COUNT(*) FROM lifecycle_outbox WHERE processing_state='claimed' AND lease_expires_at <= ?",
                (now - max(0.0, stale_after),),
            ).fetchone()[0]
            or 0
        )
        summary["stale_claims"] = stale
        summary["max_attempts"] = int(
            conn.execute("SELECT COALESCE(MAX(attempts), 0) FROM lifecycle_outbox").fetchone()[0] or 0
        )
        if stale:
            issues.append(f"{stale} stale lifecycle outbox claim{'s' if stale != 1 else ''}")
        for state in ("retry", "manual_review", "quarantined"):
            count = int(states.get(state, 0))
            if count:
                issues.append(f"{count} lifecycle intent{'s' if count != 1 else ''} in {state}")
        for row in conn.execute(
            "SELECT intent_id, processing_state, lifecycle_stage, attempts, lease_expires_at, failure_json "
            "FROM lifecycle_outbox ORDER BY updated_at ASC, intent_id ASC LIMIT ?",
            (max(0, int(limit)),),
        ):
            item: dict[str, Any] = {
                "intent_id": str(row[0]),
                "state": str(row[1]),
                "stage": str(row[2]),
                "attempts": int(row[3] or 0),
                "lease_age_s": max(0, int(now - float(row[4] or 0))) if row[4] else 0,
            }
            if row[5]:
                try:
                    failure = json.loads(str(row[5]))
                    if isinstance(failure, dict):
                        item["reason"] = str(failure.get("message") or "")
                        item["failure_code"] = str(failure.get("code") or "")
                except (TypeError, ValueError, json.JSONDecodeError):
                    item["reason"] = "invalid failure evidence"
            summary["sample"].append(item)
        return summary, issues
    except (OSError, sqlite3.Error) as exc:
        summary["error"] = str(exc)
        summary["schema"]["status"] = "error"
        issues.append(f"lifecycle outbox error: {exc}")
        return summary, issues
    finally:
        if conn is not None:
            conn.close()


def _status_payload(taskdata: Path, *, stale_after: float, limit: int) -> dict[str, Any]:
    taskdata = Path(taskdata).expanduser().resolve()
    outbox_path = lifecycle_outbox_path(taskdata)
    outbox, issues = _outbox_summary(outbox_path, stale_after=stale_after, limit=limit)
    status = "error" if outbox["schema"].get("status") == "error" or outbox["integrity"] not in {"ok", "not_checked"} else ("warn" if issues else "ok")
    return {
        "schema": _JSON_SCHEMA,
        "schema_version": _JSON_SCHEMA_VERSION,
        "status": status,
        "taskdata": str(taskdata),
        "paths": {"state_dir": str(outbox_path.parent), "outbox_db": str(outbox_path)},
        "outbox": outbox,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Nautical lifecycle outbox inspector")
    parser.add_argument("--taskdata", default=os.environ.get("TASKDATA", "~/.task"))
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--limit", type=int, default=5, help="number of sample intents")
    parser.add_argument("--stale-after-seconds", type=float, default=300.0)
    args = parser.parse_args()
    payload = _status_payload(
        Path(args.taskdata),
        stale_after=max(0.0, float(args.stale_after_seconds)),
        limit=max(0, int(args.limit)),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        outbox = payload["outbox"]
        print(f"status={payload['status']} taskdata={payload['taskdata']}")
        print(
            "outbox:"
            f" states={outbox.get('states', {})}"
            f" stale_claims={outbox.get('stale_claims', 0)}"
            f" max_attempts={outbox.get('max_attempts', 0)}"
        )
        schema = outbox["schema"]
        print(
            "schema:"
            f" status={schema.get('status')} version={schema.get('version')}"
            f" expected={schema.get('expected_version')} integrity={outbox.get('integrity')}"
        )
        for issue in payload["issues"]:
            print(f"issue: {issue}")
        for record in outbox.get("sample", []):
            print(
                "intent:"
                f" id={record.get('intent_id')} state={record.get('state')}"
                f" stage={record.get('stage')} attempts={record.get('attempts')}"
                + (f" reason={record['reason']}" if record.get("reason") else "")
            )
    return 2 if payload["status"] == "error" else 1 if payload["status"] == "warn" else 0


if __name__ == "__main__":
    raise SystemExit(main())
