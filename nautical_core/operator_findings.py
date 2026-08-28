"""Immutable, actionable findings for the operator control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .operator_models import OperatorContractError, OperatorScope, OperatorStatus


class FindingSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FindingActionability(str, Enum):
    INFORMATIONAL = "informational"
    ACTIONABLE = "actionable"
    REPAIRABLE = "repairable"
    RETRYABLE = "retryable"
    DEFERRED = "deferred"
    MANUAL_REVIEW = "manual_review"
    BLOCKING = "blocking"


_SEVERITY_RANK = {
    FindingSeverity.INFO: 0,
    FindingSeverity.WARNING: 1,
    FindingSeverity.ERROR: 2,
}


def _text(value: object, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise OperatorContractError(f"finding {name} is required")
    return text


def _json(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    raise OperatorContractError(f"finding evidence contains non-JSON value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class OperatorFinding:
    """One stable observation or remediation item."""

    code: str
    domain: str
    severity: FindingSeverity
    actionability: FindingActionability
    message: str
    scope: OperatorScope | None = None
    affected: tuple[str, ...] = ()
    observed: Mapping[str, Any] = field(default_factory=dict)
    expected: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    command: str = ""
    guidance: str = ""

    def __post_init__(self) -> None:
        try:
            severity = FindingSeverity(self.severity)
            actionability = FindingActionability(self.actionability)
        except (TypeError, ValueError) as exc:
            raise OperatorContractError("invalid finding severity or actionability") from exc
        if self.scope is not None and not isinstance(self.scope, OperatorScope):
            raise OperatorContractError("finding scope must be an OperatorScope")
        affected = tuple(dict.fromkeys(_text(value, "affected identity") for value in self.affected))
        for name, value in (("observed", self.observed), ("expected", self.expected), ("evidence", self.evidence)):
            if not isinstance(value, Mapping):
                raise OperatorContractError(f"finding {name} must be an object")
            _json(value)
        object.__setattr__(self, "code", _text(self.code, "code"))
        object.__setattr__(self, "domain", _text(self.domain, "domain"))
        object.__setattr__(self, "message", _text(self.message, "message"))
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "actionability", actionability)
        object.__setattr__(self, "affected", affected)
        object.__setattr__(self, "observed", dict(self.observed))
        object.__setattr__(self, "expected", dict(self.expected))
        object.__setattr__(self, "evidence", dict(self.evidence))
        object.__setattr__(self, "command", str(self.command or "").strip())
        object.__setattr__(self, "guidance", str(self.guidance or "").strip())
        if actionability is not FindingActionability.INFORMATIONAL and not (self.command or self.guidance):
            raise OperatorContractError("non-informational finding requires a command or guidance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "domain": self.domain,
            "severity": self.severity.value,
            "actionability": self.actionability.value,
            "message": self.message,
            "scope": None if self.scope is None else self.scope.to_dict(),
            "affected": list(self.affected),
            "observed": _json(self.observed),
            "expected": _json(self.expected),
            "evidence": _json(self.evidence),
            "command": self.command or None,
            "guidance": self.guidance or None,
        }

    def to_doctor_dict(self) -> dict[str, Any]:
        """Serialize for the pre-v2 Doctor envelope during migration."""
        details = dict(self.evidence)
        details.update({"observed": dict(self.observed), "expected": dict(self.expected)})
        return {
            "id": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "fix": self.guidance or None,
            "details": details,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorFinding":
        if not isinstance(value, Mapping):
            raise OperatorContractError("finding must be an object")
        affected = value.get("affected", ())
        if isinstance(affected, str) or not isinstance(affected, (list, tuple)):
            raise OperatorContractError("finding affected must be a list")
        scope = value.get("scope")
        return cls(
            code=value.get("code", ""),
            domain=value.get("domain", ""),
            severity=value.get("severity", ""),
            actionability=value.get("actionability", ""),
            message=value.get("message") or value.get("id") or "Doctor finding",
            scope=None if scope is None else OperatorScope.from_mapping(scope),
            affected=tuple(affected),
            observed=value.get("observed", {}),
            expected=value.get("expected", {}),
            evidence=value.get("evidence", {}),
            command=value.get("command", "") or "",
            guidance=value.get("guidance", "") or "",
        )

    @classmethod
    def from_doctor_mapping(cls, value: object) -> "OperatorFinding":
        """Normalize a legacy Doctor finding into the canonical contract."""
        if not isinstance(value, Mapping):
            raise OperatorContractError("doctor finding must be an object")
        details = value.get("details")
        details_map = dict(details) if isinstance(details, Mapping) else {}
        affected = details_map.get("subjects") or details_map.get("subject_uuids") or ()
        if isinstance(affected, str) or not isinstance(affected, (list, tuple)):
            affected = ()
        severity = str(value.get("severity") or "info").strip().lower()
        guidance = str(value.get("fix") or "").strip()
        return cls(
            code=value.get("id", ""),
            domain=str(value.get("id", "doctor")).split(".", 1)[0] or "doctor",
            severity=(FindingSeverity.ERROR if severity == "error" else FindingSeverity.WARNING
                      if severity in {"warn", "warning"} else FindingSeverity.INFO),
            actionability=(
                FindingActionability.BLOCKING
                if severity == "error"
                else FindingActionability.INFORMATIONAL
                if not guidance
                else FindingActionability.ACTIONABLE
            ),
            message=value.get("message") or value.get("id") or "Doctor finding",
            affected=tuple(affected),
            observed=details_map.get("observed", {}),
            expected=details_map.get("expected", {}),
            evidence=details_map,
            guidance=guidance,
        )


def deduplicate_findings(findings: tuple[OperatorFinding, ...] | list[OperatorFinding]) -> tuple[OperatorFinding, ...]:
    """Collapse identical findings while preserving deterministic ordering."""
    grouped: dict[tuple[object, ...], OperatorFinding] = {}
    for finding in findings:
        if not isinstance(finding, OperatorFinding):
            raise OperatorContractError("finding collection contains an invalid item")
        key = (
            finding.code, finding.domain, finding.severity, finding.actionability,
            finding.message, finding.scope.to_dict() if finding.scope else None,
            repr(finding.observed), repr(finding.expected), repr(finding.evidence),
            finding.command, finding.guidance,
        )
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = finding
        else:
            grouped[key] = OperatorFinding(
                existing.code, existing.domain, existing.severity, existing.actionability,
                existing.message, existing.scope,
                tuple(sorted(set(existing.affected) | set(finding.affected))),
                existing.observed, existing.expected, existing.evidence,
                existing.command, existing.guidance,
            )
    return tuple(grouped[key] for key in sorted(grouped, key=lambda item: repr(item)))


def highest_severity(findings: tuple[OperatorFinding, ...] | list[OperatorFinding]) -> FindingSeverity | None:
    """Return the deterministic highest severity, or ``None`` when empty."""
    normalized = tuple(findings)
    if any(not isinstance(item, OperatorFinding) for item in normalized):
        raise OperatorContractError("finding collection contains an invalid item")
    return max((item.severity for item in normalized), key=lambda value: _SEVERITY_RANK[value], default=None)


def status_for_findings(findings: tuple[OperatorFinding, ...] | list[OperatorFinding]) -> OperatorStatus:
    """Map findings to one stable aggregate operator status."""
    normalized = tuple(findings)
    if any(not isinstance(item, OperatorFinding) for item in normalized):
        raise OperatorContractError("finding collection contains an invalid item")
    if any(item.actionability is FindingActionability.BLOCKING for item in normalized):
        return OperatorStatus.ERROR
    if any(item.actionability is FindingActionability.MANUAL_REVIEW for item in normalized):
        return OperatorStatus.MANUAL_REVIEW
    if any(item.actionability is FindingActionability.RETRYABLE for item in normalized):
        return OperatorStatus.UNAVAILABLE
    if any(item.actionability is FindingActionability.REPAIRABLE for item in normalized):
        return OperatorStatus.REPAIRABLE
    if any(item.actionability is FindingActionability.DEFERRED for item in normalized):
        return OperatorStatus.DEFERRED
    if any(item.severity is FindingSeverity.WARNING for item in normalized):
        return OperatorStatus.ATTENTION
    return OperatorStatus.OK


def sort_findings(findings: tuple[OperatorFinding, ...] | list[OperatorFinding]) -> tuple[OperatorFinding, ...]:
    """Return findings in a stable severity, domain, code, identity order."""
    if any(not isinstance(item, OperatorFinding) for item in findings):
        raise OperatorContractError("finding collection contains an invalid item")
    return tuple(sorted(
        findings,
        key=lambda item: (
            -_SEVERITY_RANK[item.severity], item.domain, item.code,
            item.message, item.affected,
        ),
    ))


__all__ = ["FindingSeverity", "FindingActionability", "OperatorFinding", "deduplicate_findings", "highest_severity", "status_for_findings", "sort_findings"]
