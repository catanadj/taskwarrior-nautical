"""Read-only presentation of the shared chain integrity report."""

from __future__ import annotations

from typing import Any

from .chain_integrity_engine import IntegrityEngineResult


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
            "fix": "Run nautical chain-repair --apply after reviewing the dry-run output.",
            "details": {"repairs": [plan.to_dict() for plan in result.plans[:10]]},
        })
    if result.refusals:
        findings.append({
            "id": "chains.repair_review",
            "severity": "warn",
            "message": f"{len(result.refusals)} chain repair issue(s) need review.",
            "fix": "Run nautical chain-repair and inspect the refusal evidence.",
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
        "snapshot": result.snapshot.to_dict() if result.snapshot is not None else None,
        "findings": len(result.findings),
        "plans": len(result.plans),
        "refusals": len(result.refusals),
        "reason": result.reason,
    }


__all__ = ["doctor_findings", "summary"]
