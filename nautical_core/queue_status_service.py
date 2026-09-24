"""Typed read-only lifecycle outbox status service."""

from __future__ import annotations

from pathlib import Path
import json
import hashlib
import os
from typing import Any

from .lifecycle_outbox import (
    OUTBOX_ACK_RETENTION_SECONDS,
    OUTBOX_SCHEMA_VERSION,
    lifecycle_outbox_path,
    LifecycleOutboxRepository,
)
from .operator_context import OperatorBudgetLedger
from .taskwarrior_client import TaskwarriorClient
from .task_codec import DEFAULT_TASK_CODEC, TaskCodecError
from .lifecycle_models import recurrence_fingerprint
from .integrity_query_service import IntegrityQueryService
from .chain_snapshot import IntegritySnapshotRequest
from .integration_context import IntegrationRuntime
from .manual_review_models import ManualReviewEvidence, ManualReviewItem, ManualReviewUnavailable


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
        runtime: IntegrationRuntime | None = None,
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
        if not records and task_binary and runtime is not None and (not intent_id or str(intent_id).startswith("integrity:")):
            integrity_records = self._integrity_review_records(resolved, task_binary, max(0, int(limit)), runtime)
            records = [
                record for record in integrity_records
                if not intent_id or record.get("intent_id") == intent_id
            ]
        for record in records:
            review_item = self.build_review_item(record, task_binary=task_binary)
            review_payload: dict[str, Any] = review_item.to_dict()
            if isinstance(review_item, ManualReviewItem):
                review_payload["confirmation_token"] = self.review_confirmation_token(review_item)
                review_payload["confirmation_available"] = True
                if task_binary and (limit == 1 or intent_id):
                    context = self._task_context(
                        resolved,
                        task_binary,
                        (review_item.evidence.expected_child_uuid, review_item.evidence.parent_uuid),
                    )
                    if context:
                        review_payload["task_context"] = context
            record["review_item"] = review_payload
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
                                        expected = int(float(str(expected)))
                                        actual = int(float(str(actual)))
                                    except (TypeError, ValueError, OverflowError):
                                        pass
                                if str(expected if expected is not None else "") != str(actual if actual is not None else ""):
                                    comparisons.append({"field": field, "expected": expected, "actual": actual})
                            expected_identity = str(guard.get("recurrence_identity") or "").strip()
                            if expected_identity:
                                try:
                                    actual_identity = recurrence_fingerprint(current)
                                except Exception as exc:
                                    comparisons.append({"field": "recurrence_identity", "expected": expected_identity, "actual": f"unavailable: {exc}"})
                                else:
                                    if actual_identity != expected_identity:
                                        comparisons.append({"field": "recurrence_identity", "expected": expected_identity, "actual": actual_identity})
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

    @staticmethod
    def _task_context(taskdata: Path, task_binary: str, uuids: tuple[str, ...]) -> dict[str, str]:
        """Read one concise date/description context from authoritative rows."""
        environment = dict(os.environ)
        environment["TASKDATA"] = str(taskdata)
        client = TaskwarriorClient((task_binary,), env=environment)
        for uuid in uuids:
            if not str(uuid).strip():
                continue
            command = client.execute(
                (f"uuid:{uuid}", "export"),
                purpose="queue review task context",
                timeout=5,
                attempts=1,
            )
            if not command.ok:
                continue
            try:
                rows = DEFAULT_TASK_CODEC.decode_export(command.stdout, source_query="queue review task context")
            except (TaskCodecError, ValueError):
                continue
            if not rows:
                continue
            row = rows[0].to_mapping()
            when = next((str(row.get(field) or "").strip() for field in ("scheduled", "due", "start", "end") if str(row.get(field) or "").strip()), "")
            description = str(row.get("description") or "").strip()
            context = {}
            if when:
                context["when"] = when
            if description:
                context["description"] = description
            if context:
                return context
        return {}

    @staticmethod
    def _integrity_review_records(taskdata: Path, task_binary: str, limit: int, runtime: IntegrationRuntime) -> list[dict[str, Any]]:
        """Project authoritative integrity findings into review records."""
        environment = dict(os.environ)
        environment["TASKDATA"] = str(taskdata)
        try:
            service = IntegrityQueryService(
                runtime=runtime,
                task_binary=task_binary,
                env=environment,
            )
            payload, _exit_code = service.query(IntegritySnapshotRequest.candidates(complete_chain_history=True))
        except Exception:
            return []
        records: list[dict[str, Any]] = []
        for finding in payload.get("findings") or ():
            if str(finding.get("status") or "") not in {"manual_review", "repairable"}:
                continue
            chain_id = str(finding.get("chain_id") or "").strip()
            evidence = finding.get("evidence") or {}
            subjects = tuple(str(item) for item in (finding.get("subject_uuids") or ()))
            intent_id = "integrity:" + ":".join((chain_id, str(finding.get("invariant_id") or "unknown"), str(finding.get("reason_code") or "unknown")))
            records.append({
                "intent_id": intent_id,
                "state": "manual_review",
                "reason": str(finding.get("message") or finding.get("reason_code") or "integrity finding"),
                "occupants": tuple(evidence.get("occupants") or subjects),
                "plan": {
                    "chainID": chain_id,
                    "source_link": evidence.get("parent_link"),
                    "target_link": evidence.get("child_link"),
                    "parent_uuid": subjects[0] if subjects else "",
                    "child_uuid": subjects[1] if len(subjects) > 1 else "",
                },
                "failure": {"code": str(finding.get("reason_code") or "integrity_review"), "message": str(finding.get("message") or "")},
            })
            if limit and len(records) >= limit:
                break
        return records

    def build_review_item(
        self,
        record: dict[str, Any],
        *,
        task_binary: str | None = None,
    ) -> ManualReviewItem | ManualReviewUnavailable:
        """Project one persisted review record into redacted operator evidence.

        Chain snapshot enrichment is deliberately layered on top by the review
        command; this base projection remains safe when Taskwarrior is
        unavailable.
        """
        plan = record.get("plan") or {}
        parent_guard = plan.get("parent_guard") or {}
        chain_id = str(plan.get("chainID") or parent_guard.get("chainID") or "").strip()
        intent_id = str(record.get("intent_id") or "").strip()
        if not intent_id or not chain_id:
            return ManualReviewUnavailable(intent_id, "review record lacks intent or chain identity")
        failure = record.get("failure") or {}
        occupants = tuple(str(item) for item in (record.get("occupants") or ()))
        evidence = ManualReviewEvidence(
            chain_id=chain_id,
            source_link=plan.get("source_link", parent_guard.get("link")),
            target_link=plan.get("target_link"),
            parent_uuid=str(plan.get("parent_uuid") or parent_guard.get("uuid") or ""),
            expected_child_uuid=str(plan.get("child_uuid") or ""),
            occupants=occupants,
            reason=str(failure.get("message") or record.get("reason") or "manual review required"),
        )
        return ManualReviewItem.from_evidence(intent_id, str(record.get("state") or "manual_review"), evidence)

    def resolve_review(self, taskdata: Path, intent_id: str, reason: str) -> dict[str, Any]:
        result = LifecycleOutboxRepository(Path(taskdata).expanduser().resolve()).resolve_manual_review(
            intent_id=intent_id, reason=reason
        )
        status = "resolved" if result.ok else "already_applied" if result.kind.value == "already_applied" else "error"
        return {"status": status, "reason": result.reason}

    @staticmethod
    def review_confirmation_token(item: ManualReviewItem) -> str:
        canonical = json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def apply_review_action(
        self,
        taskdata: Path,
        intent_id: str,
        action: str,
        confirmation: str,
        *,
        task_binary: str | None = None,
        runtime: IntegrationRuntime | None = None,
    ) -> dict[str, Any]:
        payload = self.review_payload(taskdata, intent_id=intent_id, task_binary=task_binary, runtime=runtime)
        if payload.get("status") != "found" or not payload.get("intents"):
            return {"status": "error", "reason": "review intent is unavailable"}
        record = payload["intents"][0]
        item = self.build_review_item(record, task_binary=task_binary)
        if isinstance(item, ManualReviewUnavailable):
            return {"status": "error", "reason": item.reason}
        expected = self.review_confirmation_token(item)
        if str(confirmation or "").strip() != expected:
            return {"status": "conflict", "reason": f"stale or invalid confirmation token; expected {expected}"}
        normalized = str(action or "").strip().lower()
        if normalized == "skip":
            return {"status": "skipped", "intent_id": intent_id, "reason": "left unresolved by operator"}
        if normalized == "resolve-applied":
            assessment = record.get("assessment") or {}
            if assessment.get("status") != "already_applied":
                return {
                    "status": "conflict",
                    "intent_id": intent_id,
                    "reason": "resolve-applied requires high-confidence successor convergence",
                }
            return self.resolve_review(taskdata, intent_id, "resolved through guided review")
        return {"status": "unsupported", "intent_id": intent_id, "reason": f"action is not executable yet: {normalized}"}


__all__ = ["QueueStatusService"]
