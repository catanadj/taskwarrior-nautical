"""Read-only presentation of the shared chain integrity report."""

from __future__ import annotations

from typing import Any

from .chain_integrity_engine import IntegrityEngineResult


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


def doctor_findings(result: IntegrityEngineResult) -> list[dict[str, Any]]:
    """Map stable engine findings to Doctor's presentation contract."""
    findings: list[dict[str, Any]] = []
    for finding in result.findings:
        severity = {
            "healthy": "ok",
            "repairable": "warn",
            "blocked": "error",
            "manual_review": "warn",
            "unavailable": "error",
        }.get(finding.status.value, "error")
        details = {
            "invariant_id": finding.invariant_id,
            "reason_code": finding.reason_code,
            "chainID": finding.chain_id or None,
            "subjects": list(finding.subject_uuids),
            "observed": dict(finding.observed),
            "expected": dict(finding.expected),
            "evidence": dict(finding.evidence),
            "snapshot": finding.snapshot_id,
        }
        findings.append({
            "id": f"chains.{finding.invariant_id}",
            "severity": severity,
            "message": finding.message,
            "fix": (
                "Review the integrity evidence and run nautical reconcile --apply."
                if finding.status.value == "repairable"
                else "Inspect the invariant evidence before modifying tasks."
            ),
            "details": details,
        })
    if result.status.value == "unavailable":
        findings.append({
            "id": "chains.integrity_unavailable",
            "severity": "error",
            "message": "Chain integrity could not be evaluated.",
            "fix": "Retry after resolving the reported Taskwarrior or configuration error.",
            "details": {"reason": result.reason},
        })
    elif result.status.value == "healthy" and not findings:
        findings.append({
            "id": "chains.integrity",
            "severity": "ok",
            "message": "Chain integrity is clean.",
            "details": {"snapshot": result.snapshot.snapshot_id if result.snapshot is not None else None},
        })
    if result.plans:
        findings.append({
            "id": "chains.repair_available",
            "severity": "warn",
            "message": f"{len(result.plans)} safe chain repair(s) are available.",
            "fix": "Run nautical reconcile --apply after reviewing the dry-run output.",
            "details": {"repairs": [plan.to_dict() for plan in result.plans[:10]]},
        })
    if result.refusals:
        findings.append({
            "id": "chains.repair_review",
            "severity": "warn",
            "message": f"{len(result.refusals)} chain repair issue(s) need review.",
            "fix": "Run nautical query integrity --all for evidence, then use reconcile for applicable repairs.",
            "details": {
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
        })
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
    return {
        "status": result.status.value,
        "snapshot": _snapshot_payload(result.snapshot),
        "findings": [_finding_payload(finding) for finding in result.findings],
        "plans": [plan.to_dict() for plan in result.plans],
        "refusals": [
            {
                "invariant_id": item.invariant_id,
                "reason_code": item.reason_code,
                "reason": item.reason,
                "snapshot_id": item.snapshot_id,
            }
            for item in result.refusals
        ],
        "chain_statuses": [
            {"chainID": chain_id, "status": status.value}
            for chain_id, status in result.chain_statuses
        ],
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
