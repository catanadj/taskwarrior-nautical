"""Read-only presentation of the shared chain integrity report."""

from __future__ import annotations

from typing import Any

from .chain_integrity_engine import IntegrityEngineResult
from .operator_findings import FindingActionability, FindingSeverity, OperatorFinding
from .operator_presentation import ordered_records


def _snapshot_payload(snapshot: Any) -> dict[str, Any] | None:
    """Serialize the immutable snapshot model without depending on graph APIs."""
    if snapshot is None:
        return None
    return {
        "snapshot_id": snapshot.snapshot_id,
        "coverage": snapshot.coverage.value,
        "source": snapshot.source,
        "row_count": len(snapshot.rows),
        "configuration_fingerprint": snapshot.configuration_fingerprint,
        "complete_chain_history": snapshot.complete_chain_history,
        "reason": snapshot.reason,
    }


def _finding_payload(finding: Any) -> dict[str, Any]:
    """Serialize an immutable finding model for operator-facing reports."""
    return {
        "invariant_id": finding.invariant_id,
        "status": finding.status.value,
        "severity": finding.severity.value,
        "snapshot_id": finding.snapshot_id,
        "chain_id": finding.chain_id,
        "subject_uuids": list(finding.subject_uuids),
        "reason_code": finding.reason_code,
        "message": finding.message,
        "observed": dict(finding.observed),
        "expected": dict(finding.expected),
        "evidence": dict(finding.evidence),
    }


def _doctor_finding(
    code: str,
    severity: str,
    message: str,
    *,
    guidance: str = "",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Doctor record through the canonical finding contract."""
    canonical = OperatorFinding(
        code=code,
        domain=code.split(".", 1)[0] or "doctor",
        severity=(
            FindingSeverity.ERROR if severity == "error"
            else FindingSeverity.WARNING if severity in {"warn", "warning"}
            else FindingSeverity.INFO
        ),
        actionability=(
            FindingActionability.BLOCKING
            if severity == "error"
            else FindingActionability.INFORMATIONAL
            if severity in {"ok", "info"} and not guidance
            else FindingActionability.ACTIONABLE
        ),
        message=message,
        observed={},
        evidence=details or {},
        guidance=guidance or ("Inspect the reported evidence." if severity == "error" else ""),
    )
    payload = canonical.to_doctor_dict()
    payload["severity"] = severity
    return payload


def doctor_findings(result: IntegrityEngineResult) -> list[dict[str, Any]]:
    """Map stable engine findings to Doctor's presentation contract."""
    findings: list[dict[str, Any]] = []
    task_status = {
        node.task_uuid.lower(): str(node.status or "").strip().lower()
        for node in (result.snapshot.rows if result.snapshot is not None else ())
    }
    historical_statuses = {"completed", "deleted"}
    for finding in result.findings:
        subject_statuses = [task_status.get(uuid.lower()) for uuid in finding.subject_uuids]
        historical = bool(subject_statuses) and all(
            status in historical_statuses for status in subject_statuses
        )
        severity = {
            "healthy": "ok",
            "repairable": "warn",
            "blocked": "error",
            "manual_review": "warn",
            "unavailable": "error",
        }.get(finding.status.value, "error")
        if historical:
            severity = "info"
        details = {
            "invariant_id": finding.invariant_id,
            "reason_code": finding.reason_code,
            "chainID": finding.chain_id or None,
            "subjects": list(finding.subject_uuids),
            "observed": dict(finding.observed),
            "expected": dict(finding.expected),
            "evidence": dict(finding.evidence),
            "snapshot": finding.snapshot_id,
            "historical": historical,
        }
        guidance = (
                "Historical finding retained for audit; no action is required unless the chain is reactivated."
                if historical
                else
                "Review the integrity evidence and run nautical reconcile --apply."
                if finding.status.value == "repairable"
                else "Inspect the invariant evidence before modifying tasks."
            )
        canonical = OperatorFinding(
            code=f"chains.{finding.invariant_id}",
            domain="chains",
            severity=FindingSeverity.INFO if historical else FindingSeverity.ERROR if severity == "error" else FindingSeverity.WARNING,
            actionability=FindingActionability.INFORMATIONAL if historical else FindingActionability.ACTIONABLE,
            message=finding.message,
            affected=tuple(finding.subject_uuids),
            observed=finding.observed if isinstance(finding.observed, dict) else {},
            expected=finding.expected if isinstance(finding.expected, dict) else {},
        evidence=details or {},
            guidance=guidance,
        )
        findings.append(canonical.to_doctor_dict())
    if result.status.value == "unavailable":
        findings.append(_doctor_finding(
            "chains.integrity_unavailable",
            "error",
            "Chain integrity could not be evaluated.",
            guidance="Retry after resolving the reported Taskwarrior or configuration error.",
            details={"reason": result.reason},
        ))
    elif result.status.value == "healthy" and not findings:
        findings.append(_doctor_finding(
            "chains.integrity",
            "ok",
            "Chain integrity is clean.",
            details={"snapshot": result.snapshot.snapshot_id if result.snapshot is not None else None},
        ))
    if result.plans:
        findings.append(_doctor_finding(
            "chains.repair_available",
            "warn",
            f"{len(result.plans)} safe chain repair(s) are available.",
            guidance="Run nautical reconcile --apply after reviewing the dry-run output.",
            details={"repairs": [plan.to_dict() for plan in result.plans[:10]]},
        ))
    if result.refusals:
        findings.append(_doctor_finding(
            "chains.repair_review",
            "warn",
            f"{len(result.refusals)} chain repair issue(s) need review.",
            guidance="Run nautical query integrity --all for evidence, then use reconcile for applicable repairs.",
            details={
                "reasons": {
                    str(item.reason or item.reason_code).strip(): sum(
                        1 for candidate in result.refusals
                        if str(candidate.reason or candidate.reason_code).strip()
                        == str(item.reason or item.reason_code).strip()
                    )
                    for item in result.refusals
                },
                "issues": [
                    {
                        "invariant_id": item.invariant_id,
                        "reason_code": item.reason_code,
                        "message": item.reason,
                        "snapshot": item.snapshot_id,
                        "chainID": item.chain_id,
                        "subjects": list(item.subject_uuids),
                    }
                    for item in result.refusals[:10]
                ],
            },
        ))
    return findings


def summary(result: IntegrityEngineResult) -> dict[str, Any]:
    """Return a stable, presentation-neutral integrity summary."""
    return {
        "status": result.status.value,
        "snapshot": _snapshot_payload(result.snapshot),
        "findings": len(result.findings),
        "plans": len(result.plans),
        "refusals": len(result.refusals),
        "reason": result.reason,
    }


def components(result: IntegrityEngineResult) -> dict[str, Any]:
    """Return the stable evidence components shared by all consumers."""
    finding_rows = ordered_records(
        [_finding_payload(finding) for finding in result.findings],
        keys=("chain_id", "severity", "invariant_id", "reason_code", "snapshot_id"),
    )
    plan_rows = ordered_records(
        [plan.to_dict() for plan in result.plans],
        keys=("chainID", "parent_link", "action", "trigger", "child"),
    )
    refusal_rows = ordered_records(
        [
            {
                "invariant_id": item.invariant_id,
                "reason_code": item.reason_code,
                "reason": item.reason,
                "snapshot_id": item.snapshot_id,
                "chain_id": item.chain_id,
            }
            for item in result.refusals
        ],
        keys=("chain_id", "invariant_id", "reason_code", "snapshot_id"),
    )
    chain_rows = ordered_records(
        [{"chainID": chain_id, "status": status.value} for chain_id, status in result.chain_statuses],
        keys=("chainID", "status"),
    )
    return {
        "status": result.status.value,
        "snapshot": _snapshot_payload(result.snapshot),
        "findings": list(finding_rows),
        "plans": list(plan_rows),
        "refusals": list(refusal_rows),
        "chain_statuses": list(chain_rows),
        "failure": {"message": result.reason} if result.reason else None,
    }


def public_payload(
    result: IntegrityEngineResult,
    *,
    query: dict[str, Any],
    configuration_fingerprint: str,
) -> dict[str, Any]:
    """Serialize one engine result for external read-only consumers."""
    return {
        "schema": "nautical.query.integrity",
        "version": 1,
        "operation": "integrity",
        **components(result),
        "configuration_fingerprint": str(configuration_fingerprint or ""),
        "query": query,
    }


__all__ = ["components", "doctor_findings", "public_payload", "summary"]
