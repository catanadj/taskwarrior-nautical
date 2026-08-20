#!/usr/bin/env python3
"""Read-only lifecycle outbox inspector used by Nautical operators and doctor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nautical_core.lifecycle_outbox import (  # noqa: E402
    OUTBOX_SCHEMA_VERSION,
    OUTBOX_ACK_RETENTION_SECONDS,
    LifecycleOutboxRepository,
    lifecycle_outbox_path,
)


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
        "retention": {
            "retention_seconds": OUTBOX_ACK_RETENTION_SECONDS,
            "acknowledged": 0,
            "eligible": 0,
            "oldest_age_s": 0,
        },
        "sample": [],
    }
    issues: list[str] = []
    if not path.exists():
        return summary, issues
    summary["exists"] = True
    # ``path`` is the derived .../.nautical-state/database path; the
    # repository accepts the Taskdata root and owns that derivation.
    repository = LifecycleOutboxRepository(path.parent.parent)
    result, data = repository.status(limit=limit, stale_after=stale_after)
    summary["integrity"] = str(data.get("integrity") or "not_checked")
    summary["states"] = dict(data.get("states") or {})
    summary["stale_claims"] = int(data.get("stale_claims") or 0)
    summary["max_attempts"] = int(data.get("max_attempts") or 0)
    summary["retention"] = dict(data.get("retention") or summary["retention"])
    version = int(data.get("schema_version") or 0)
    schema = summary["schema"]
    schema["version"] = version
    if result.ok and version == OUTBOX_SCHEMA_VERSION:
        schema["status"] = "ok"
    else:
        schema["status"] = "error"
        reason = result.reason or f"lifecycle outbox schema v{version} is incompatible"
        summary["error"] = reason
        issues.append(f"lifecycle outbox error: {reason}")
        return summary, issues
    if summary["integrity"].lower() != "ok":
        issues.append(f"lifecycle outbox integrity check failed: {summary['integrity']}")
    stale = summary["stale_claims"]
    if stale:
        issues.append(f"{stale} stale lifecycle outbox claim{'s' if stale != 1 else ''}")
    states = summary["states"]
    for state in ("retry", "manual_review", "quarantined"):
        count = int(states.get(state, 0))
        if count:
            issues.append(f"{count} lifecycle intent{'s' if count != 1 else ''} in {state}")
    eligible = int((summary.get("retention") or {}).get("eligible") or 0)
    if eligible:
        issues.append(
            f"{eligible} acknowledged lifecycle intent{'s' if eligible != 1 else ''} exceed retention; "
            "run nautical queue-status --prune-acknowledged"
        )
    for record in data.get("records") or []:
        item: dict[str, Any] = {
            "intent_id": str(record.get("intent_id") or ""),
            "state": str(record.get("state") or ""),
            "stage": str(record.get("stage") or ""),
            "attempts": int(record.get("attempts") or 0),
            "lease_age_s": int(record.get("lease_age_s") or 0),
        }
        failure = record.get("failure")
        if isinstance(failure, dict):
            item["reason"] = str(failure.get("message") or "")
            item["failure_code"] = str(failure.get("code") or "")
        elif record.get("reason"):
            item["reason"] = str(record["reason"])
        summary["sample"].append(item)
    return summary, issues


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
    parser.add_argument(
        "--prune-acknowledged",
        action="store_true",
        help="explicitly remove acknowledged intents older than the retention policy",
    )
    parser.add_argument("--retention-seconds", type=float, default=OUTBOX_ACK_RETENTION_SECONDS)
    parser.add_argument("--maintenance-limit", type=int, default=1000)
    parser.add_argument("--checkpoint", action="store_true", help="request a passive WAL checkpoint during pruning")
    args = parser.parse_args()
    maintenance: dict[str, Any] | None = None
    if args.prune_acknowledged:
        taskdata = Path(args.taskdata).expanduser().resolve()
        result = LifecycleOutboxRepository(taskdata).prune_acknowledged(
            retention_seconds=args.retention_seconds,
            limit=args.maintenance_limit,
            checkpoint=args.checkpoint,
        )
        maintenance = {
            "kind": result.kind.value,
            "ok": result.ok,
            "removed": result.removed,
            "retention_seconds": result.retention_seconds,
            "checkpoint": result.checkpoint,
            "reason": result.reason,
        }
    payload = _status_payload(
        Path(args.taskdata),
        stale_after=max(0.0, float(args.stale_after_seconds)),
        limit=max(0, int(args.limit)),
    )
    if maintenance is not None:
        payload["maintenance"] = maintenance
        if not maintenance["ok"]:
            payload["issues"].append(f"outbox maintenance failed: {maintenance['reason'] or maintenance['kind']}")
            payload["status"] = "error"
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
        retention = outbox.get("retention") or {}
        print(
            "retention:"
            f" acknowledged={retention.get('acknowledged', 0)}"
            f" eligible={retention.get('eligible', 0)}"
            f" oldest_age_s={retention.get('oldest_age_s', 0)}"
            f" policy_s={retention.get('retention_seconds', OUTBOX_ACK_RETENTION_SECONDS)}"
        )
        if maintenance is not None:
            print(
                "maintenance:"
                f" kind={maintenance['kind']} removed={maintenance['removed']}"
                f" checkpoint={maintenance['checkpoint']}"
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
