"""Typed public document boundary for the Doctor operator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from .operator_models import OperatorFailure, OperatorV2Result, OperatorV2Status


def format_task(task: dict[str, Any]) -> str:
    """Format a task reference for human-facing Doctor details."""
    uuid = str(task.get("uuid") or "")
    short = uuid[:8] if uuid else "unknown"
    description = str(task.get("description") or "").strip() or "(no description)"
    parts = [f"{short} {description}"]
    chain_id = str(task.get("chainID") or "").strip()
    link = task.get("link")
    status = str(task.get("status") or "").strip()
    if chain_id:
        parts.append(f"chain={chain_id}")
    if link not in (None, ""):
        parts.append(f"link={link}")
    if status:
        parts.append(f"status={status}")
    return " | ".join(parts)


def timezone_summary(findings: list[Mapping[str, Any]]) -> str:
    """Extract the concise timezone line used by Doctor's text renderer."""
    for item in findings:
        check_id = item.get("id")
        if check_id == "config.timezone":
            return str(item.get("message") or "").replace("Nautical timezone is available: ", "")
        if check_id in {"config.timezone.missing", "config.timezone.invalid", "config.timezone.unavailable"}:
            details_value = item.get("details")
            details = dict(details_value) if isinstance(details_value, Mapping) else {}
            tz_name = str(details.get("tz") or "?")
            return f"{tz_name} unavailable; UTC fallback active"
    return ""


class DoctorReport(dict[str, Any]):
    """Validated Doctor document with lossless forward-compatible fields."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        if not isinstance(document, Mapping):
            raise ValueError("Doctor report must be an object")
        if document.get("schema") != "nautical.doctor":
            raise ValueError("invalid Doctor report schema")
        if document.get("schema_version") != 1:
            raise ValueError("unsupported Doctor report version")
        if document.get("version") != 2 or document.get("operation") != "diagnose":
            raise ValueError("invalid Doctor report envelope")
        if str(document.get("status") or "") not in {
            "ok", "attention", "repairable", "deferred", "manual_review",
            "unavailable", "partial", "error",
        }:
            raise ValueError("invalid Doctor report status")
        super().__init__(document)

    @classmethod
    def from_mapping(cls, value: object) -> "DoctorReport":
        if not isinstance(value, Mapping):
            raise ValueError("Doctor report must be an object")
        return cls(value)

    def to_dict(self) -> dict[str, Any]:
        return dict(self)


def to_operator_result(payload: Mapping[str, Any]) -> OperatorV2Result:
    """Wrap a Doctor document in the canonical operator envelope."""
    raw_status = str(payload.get("status") or "error")
    status = OperatorV2Status.ATTENTION if raw_status == "warn" else OperatorV2Status(raw_status)
    failure = None
    if status in {OperatorV2Status.ERROR, OperatorV2Status.UNAVAILABLE, OperatorV2Status.INVALID}:
        findings = payload.get("operator_findings")
        finding = next((item for item in findings or () if isinstance(item, Mapping)), {})
        evidence = finding.get("evidence")
        failure = OperatorFailure(
            code=str(finding.get("code") or "doctor_error"),
            message=str(finding.get("message") or "Doctor reported an error"),
            details=cast(Mapping[str, Any], evidence) if isinstance(evidence, Mapping) else {},
        )
    return OperatorV2Result(
        schema="nautical.doctor",
        operation="diagnose",
        status=status,
        payload={key: value for key, value in payload.items() if key not in {"schema", "status"}},
        failure=failure,
    )


def render_finding(item: object) -> dict[str, Any]:
    """Project a canonical finding into Doctor's detail-oriented view."""
    if not isinstance(item, dict):
        return {}
    if "code" not in item:
        return item
    severity = str(item.get("severity") or "info")
    severity = "warn" if severity == "warning" else severity
    evidence = item.get("evidence")
    details = dict(evidence) if isinstance(evidence, dict) else {}
    observed = item.get("observed")
    expected = item.get("expected")
    if isinstance(observed, dict):
        details["observed"] = observed
    if isinstance(expected, dict):
        details["expected"] = expected
    return {
        "id": item.get("code"),
        "severity": severity,
        "message": item.get("message"),
        "fix": item.get("guidance") or "",
        "details": details,
    }


def historical_summaries(findings: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group historical findings for concise human diagnostics."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for item in findings:
        if item.get("severity") != "info":
            continue
        details_value = item.get("details")
        details = dict(details_value) if isinstance(details_value, Mapping) else {}
        if not details.get("historical"):
            continue
        invariant_id = str(details.get("invariant_id") or item.get("id") or "historical")
        observed_value = details.get("observed")
        observed = dict(observed_value) if isinstance(observed_value, Mapping) else {}
        field = str(observed.get("field") or "").strip()
        chain_id = str(details.get("chainID") or "").strip()
        group = groups.setdefault((invariant_id, field), {"count": 0, "chains": set(), "subjects": set()})
        group["count"] += 1
        if chain_id:
            group["chains"].add(chain_id)
        group["subjects"].update(str(value) for value in details.get("subjects") or () if value)
    result: list[dict[str, Any]] = []
    for (invariant_id, field), group in sorted(groups.items()):
        chains = sorted(group["chains"])
        count = int(group["count"])
        label = f" {field}" if field else ""
        result.append({
            "id": "chains.historical_summary",
            "severity": "info",
            "message": f"{count} completed-link{label} observation(s) retained for audit.",
            "fix": "No action is required; current pending-chain findings are reported separately.",
            "details": {
                "invariant_id": invariant_id,
                "historical_count": count,
                "chain_count": len(chains),
                "chains": chains[:8],
                "subjects": sorted(group["subjects"])[:8],
                "detail_command": f"nautical query integrity --chain-id {chains[0]}" if len(chains) == 1 else "nautical query integrity --all",
                "detail_commands": [f"nautical query integrity --chain-id {chain_id}" for chain_id in chains],
            },
        })
    return result


__all__ = ["DoctorReport", "format_task", "historical_summaries", "render_finding", "timezone_summary", "to_operator_result"]
