"""Typed read-only lifecycle outbox status service."""

from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Any

from .lifecycle_outbox import OUTBOX_ACK_RETENTION_SECONDS, OUTBOX_SCHEMA_VERSION, LifecycleOutboxRepository, lifecycle_outbox_path
from .operator_context import OperatorBudgetLedger
from .taskwarrior_client import TaskwarriorClient
from .task_codec import DEFAULT_TASK_CODEC, TaskCodecError


class QueueStatusService:
    """Collect lifecycle outbox health without presentation concerns."""

    def outbox_summary(
        self,
        path: Path,
        *,
        stale_after: float,
        limit: int,
        budget: OperatorBudgetLedger | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        summary: dict[str, Any] = {
            "exists": False,
            "schema": {"status": "absent", "version": 0, "expected_version": OUTBOX_SCHEMA_VERSION},
            "integrity": "not_checked", "states": {}, "stale_claims": 0, "max_attempts": 0,
            "retention": {"retention_seconds": OUTBOX_ACK_RETENTION_SECONDS, "acknowledged": 0, "eligible": 0, "oldest_age_s": 0},
            "sample": [],
        }
        issues: list[str] = []
        if not path.exists():
            return summary, issues
        summary["exists"] = True
        if budget is not None:
            if not budget.consume("sqlite_transactions"):
                return summary, ["operator SQLite transaction budget exhausted"]
            requested_limit = limit
            limit = min(limit, budget.remaining("outbox_rows"))
            if requested_limit > 0 and limit == 0:
                return summary, ["operator outbox row budget exhausted"]
            if limit > 0 and not budget.consume("outbox_rows", limit):
                return summary, ["operator outbox row budget exhausted"]
        result, data = LifecycleOutboxRepository(path.parent.parent).status(limit=limit, stale_after=stale_after)
        summary["integrity"] = str(data.get("integrity") or "not_checked")
        summary["states"] = dict(data.get("states") or {})
        summary["stale_claims"] = int(data.get("stale_claims") or 0)
        summary["max_attempts"] = int(data.get("max_attempts") or 0)
        summary["retention"] = dict(data.get("retention") or summary["retention"])
        schema = summary["schema"]
        version = int(data.get("schema_version") or 0)
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
        for state in ("retry", "manual_review", "quarantined"):
            count = int(summary["states"].get(state, 0))
            if count:
                issues.append(f"{count} lifecycle intent{'s' if count != 1 else ''} in {state}")
        eligible = int(summary["retention"].get("eligible") or 0)
        if eligible:
            issues.append(f"{eligible} acknowledged lifecycle intent{'s' if eligible != 1 else ''} exceed retention; run nautical queue-status --prune-acknowledged")
        for record in data.get("records") or []:
            item: dict[str, Any] = {"intent_id": str(record.get("intent_id") or ""), "state": str(record.get("state") or ""), "stage": str(record.get("stage") or ""), "attempts": int(record.get("attempts") or 0), "lease_age_s": int(record.get("lease_age_s") or 0)}
            failure = record.get("failure")
            if isinstance(failure, dict):
                item["reason"] = str(failure.get("message") or "")
                item["failure_code"] = str(failure.get("code") or "")
            elif record.get("reason"):
                item["reason"] = str(record["reason"])
            summary["sample"].append(item)
        return summary, issues

    def status_payload(
        self,
        taskdata: Path,
        *,
        stale_after: float,
        limit: int,
        budget: OperatorBudgetLedger | None = None,
    ) -> dict[str, Any]:
        resolved = Path(taskdata).expanduser().resolve()
        outbox_path = lifecycle_outbox_path(resolved)
        outbox, issues = self.outbox_summary(outbox_path, stale_after=stale_after, limit=limit, budget=budget)
        status = "error" if outbox["schema"].get("status") == "error" or outbox["integrity"] not in {"ok", "not_checked"} else ("warn" if issues else "ok")
        return {"schema": "nautical.lifecycle_outbox_status", "schema_version": 1, "status": status, "taskdata": str(resolved), "paths": {"state_dir": str(outbox_path.parent), "outbox_db": str(outbox_path)}, "outbox": outbox, "issues": issues}

    def review_payload(
        self,
        taskdata: Path,
        *,
        limit: int = 100,
        intent_id: str | None = None,
        task_binary: str | None = None,
    ) -> dict[str, Any]:
        """Return bounded, read-only evidence for manual-review intents."""
        resolved = Path(taskdata).expanduser().resolve()
        repository = LifecycleOutboxRepository(resolved)
        result, data = repository.status(limit=max(0, int(limit)), intent_id=intent_id)
        if not result.ok:
            return {
                "schema": "nautical.lifecycle_outbox_review",
                "version": 1,
                "status": "unavailable",
                "taskdata": str(resolved),
                "intents": [],
                "failure": {"code": "review_unavailable", "message": result.reason or "outbox read failed"},
            }
        all_records = list(data.get("records", []))
        records = [
            record for record in data.get("records", [])
            if record.get("state") in {"manual_review", "quarantined", "poison"}
        ]
        if intent_id and records:
            record = records[0]
            guard = ((record.get("plan") or {}).get("parent_guard") or {})
            parent_uuid = str(((record.get("plan") or {}).get("parent_uuid") or "")).strip()
            if parent_uuid:
                environment = dict(os.environ)
                environment["TASKDATA"] = str(resolved)
                client = TaskwarriorClient(
                    (task_binary or os.environ.get("NAUTICAL_TASK_BIN") or "task",),
                    env=environment,
                )
                command = client.execute(
                    (f"uuid:{parent_uuid}", "export"),
                    purpose="queue review parent guard",
                    timeout=5.0,
                    attempts=1,
                )
                if command.ok:
                    try:
                        rows = DEFAULT_TASK_CODEC.decode_export(command.stdout, source_query="queue review parent")
                    except (TaskCodecError, ValueError) as exc:
                        record["guard_comparison"] = {"status": "unavailable", "reason": f"parent export could not be decoded: {exc}"}
                    else:
                        current = rows[0].to_mapping() if rows else None
                        if current is None:
                            record["guard_comparison"] = {"status": "unavailable", "reason": "parent task was not found"}
                        else:
                            comparisons = []
                            fields = ["status", "chain", "chainID", "link"]
                            guard_timestamp = "end" if guard.get("end") else "modified" if guard.get("modified") else None
                            if guard_timestamp:
                                fields.append(guard_timestamp)
                            for field in fields:
                                expected = guard.get(field)
                                actual = current.get(field)
                                if field == "link":
                                    try:
                                        expected = int(float(expected))
                                        actual = int(float(actual))
                                    except (TypeError, ValueError, OverflowError):
                                        pass
                                if str(expected if expected is not None else "") != str(actual if actual is not None else ""):
                                    comparisons.append({"field": field, "expected": expected, "actual": actual})
                            record["guard_comparison"] = {"status": "changed" if comparisons else "matches", "differences": comparisons}
                            plan = record.get("plan") or {}
                            if plan.get("action") == "spawn_child":
                                child_uuid = str(plan.get("child_uuid") or "").strip()
                                child_row = None
                                if child_uuid:
                                    child_command = client.execute(
                                        (f"uuid:{child_uuid}", "export"),
                                        purpose="queue review successor verification",
                                        timeout=5.0,
                                        attempts=1,
                                    )
                                    if child_command.ok:
                                        try:
                                            child_rows = DEFAULT_TASK_CODEC.decode_export(
                                                child_command.stdout, source_query="queue review successor"
                                            )
                                            child_row = child_rows[0].to_mapping() if child_rows else None
                                        except (TaskCodecError, ValueError):
                                            child_row = None
                                next_link = str(current.get("nextLink") or "").strip().lower()
                                child_short = child_uuid[:8].lower()
                                if child_row is not None and next_link == child_short:
                                    record["assessment"] = {
                                        "status": "already_applied",
                                        "confidence": "high",
                                        "message": "Parent nextLink and expected successor are present; no spawn is needed.",
                                    }
                                else:
                                    record["assessment"] = {
                                        "status": "needs_review",
                                        "confidence": "insufficient",
                                        "message": "Successor state could not be proven safe for automatic action.",
                                        "options": [
                                            "Inspect the parent nextLink and expected successor before retrying.",
                                            "If the successor is correct, resolve this intent as already applied.",
                                            "If it is absent, rerun reconcile after confirming the parent guard.",
                                        ],
                                    }
                else:
                    record["guard_comparison"] = {"status": "unavailable", "reason": command.stderr.strip() or command.stdout.strip() or "parent export failed"}
        if intent_id and not records and all_records:
            state = str(all_records[0].get("state") or "unknown")
            return {
                "schema": "nautical.lifecycle_outbox_review",
                "version": 1,
                "status": "not_reviewable",
                "taskdata": str(resolved),
                "intents": all_records,
                "failure": {"code": "intent_not_reviewable", "message": f"Intent is in state '{state}', not manual review"},
            }
        if intent_id and not records:
            return {
                "schema": "nautical.lifecycle_outbox_review",
                "version": 1,
                "status": "not_found",
                "taskdata": str(resolved),
                "intents": [],
                "failure": {"code": "intent_not_found", "message": f"No review intent found: {intent_id}"},
            }
        return {
            "schema": "nautical.lifecycle_outbox_review",
            "version": 1,
            "status": "found" if records else "empty",
            "taskdata": str(resolved),
            "intents": records,
            "failure": None,
        }


__all__ = ["QueueStatusService"]
