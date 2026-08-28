"""Pure inspector boundaries for operator snapshot evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol, Sequence

from .operator_findings import FindingActionability, FindingSeverity, OperatorFinding, deduplicate_findings
from .operator_models import CoverageKind, CoverageRequirement, OperatorContractError, OperatorLimits, OperatorScope
from .operator_snapshot import OperatorSnapshot


class OperatorInspector(Protocol):
    """Pure observer consuming immutable snapshot evidence only."""

    def inspect(self, snapshot: OperatorSnapshot) -> tuple[OperatorFinding, ...]:
        ...


@dataclass(frozen=True, slots=True)
class ComponentValidityInspector:
    """Named pure inspector for one domain component."""

    component: str
    scope: OperatorScope | None = None

    def inspect(self, snapshot: OperatorSnapshot) -> tuple[OperatorFinding, ...]:
        return inspect_component_validity(snapshot, self.component, scope=self.scope)


class TaskDomainInspector(ComponentValidityInspector):
    def __init__(self, scope: OperatorScope | None = None) -> None:
        super().__init__("task_domain", scope)


class ScheduleAvailabilityInspector(ComponentValidityInspector):
    def __init__(self, scope: OperatorScope | None = None) -> None:
        super().__init__("schedule", scope)


class InstallationInspector(ComponentValidityInspector):
    """Inspect managed runtime/install evidence without probing the filesystem."""

    def __init__(self, scope: OperatorScope | None = None) -> None:
        super().__init__("installation", scope)


class ConfigurationInspector(ComponentValidityInspector):
    """Inspect validated scheduling configuration evidence."""

    def __init__(self, scope: OperatorScope | None = None) -> None:
        super().__init__("configuration", scope)


class DependenciesInspector(ComponentValidityInspector):
    """Inspect resolved runtime dependency evidence."""

    def __init__(self, scope: OperatorScope | None = None) -> None:
        super().__init__("dependencies", scope)


class ChainIntegrityInspector(ComponentValidityInspector):
    """Inspect chain-integrity evidence captured in the snapshot."""

    def __init__(self, scope: OperatorScope | None = None) -> None:
        super().__init__("chain_integrity", scope)


class LifecycleOutboxInspector(ComponentValidityInspector):
    """Inspect lifecycle/outbox evidence captured in the snapshot."""

    def __init__(self, scope: OperatorScope | None = None) -> None:
        super().__init__("lifecycle", scope)


class PerformanceInspector(ComponentValidityInspector):
    """Inspect bounded operational performance evidence."""

    def __init__(self, scope: OperatorScope | None = None) -> None:
        super().__init__("performance", scope)


def inspect_integrity_findings(
    findings: Sequence[Any],
    *,
    scope: OperatorScope | None = None,
) -> tuple[OperatorFinding, ...]:
    """Project typed chain-integrity findings without losing their evidence."""
    from .chain_integrity_models import FindingSeverity as IntegritySeverity, FindingStatus

    result: list[OperatorFinding] = []
    for finding in findings:
        if not hasattr(finding, "to_dict"):
            raise TypeError("integrity findings must be typed model values")
        raw = finding.to_dict()
        status = FindingStatus(raw["status"])
        if status is FindingStatus.HEALTHY:
            continue
        actionability = {
            FindingStatus.REPAIRABLE: FindingActionability.REPAIRABLE,
            FindingStatus.BLOCKED: FindingActionability.BLOCKING,
            FindingStatus.MANUAL_REVIEW: FindingActionability.MANUAL_REVIEW,
            FindingStatus.UNAVAILABLE: FindingActionability.RETRYABLE,
        }[status]
        severity = {
            IntegritySeverity.INFO: FindingSeverity.INFO,
            IntegritySeverity.WARNING: FindingSeverity.WARNING,
            IntegritySeverity.ERROR: FindingSeverity.ERROR,
        }[IntegritySeverity(raw["severity"])]
        result.append(OperatorFinding(
            code=str(raw["reason_code"]),
            domain="chain_integrity",
            severity=severity,
            actionability=actionability,
            message=str(raw["message"]),
            scope=scope,
            affected=tuple(raw.get("subject_uuids", ())),
            observed=raw.get("observed", {}),
            expected=raw.get("expected", {}),
            evidence={"snapshot_id": raw["snapshot_id"], "invariant_id": raw["invariant_id"], **raw.get("evidence", {})},
            guidance="Apply the associated guarded repair plan or inspect the chain evidence.",
        ))
    return deduplicate_findings(result)


def inspect_lifecycle_outcomes(
    outcomes: Sequence[Any],
    *,
    scope: OperatorScope | None = None,
) -> tuple[OperatorFinding, ...]:
    """Project typed lifecycle application outcomes into stable findings."""
    from .lifecycle_application import LifecycleApplicationOutcomeKind

    result: list[OperatorFinding] = []
    for outcome in outcomes:
        if not hasattr(outcome, "kind") or not hasattr(outcome, "identity"):
            raise TypeError("lifecycle outcomes must be typed model values")
        kind = LifecycleApplicationOutcomeKind(outcome.kind)
        if kind in {LifecycleApplicationOutcomeKind.APPLIED, LifecycleApplicationOutcomeKind.ALREADY_APPLIED, LifecycleApplicationOutcomeKind.NOOP}:
            continue
        actionability = {
            LifecycleApplicationOutcomeKind.RETRYABLE: FindingActionability.RETRYABLE,
            LifecycleApplicationOutcomeKind.CONFLICT: FindingActionability.MANUAL_REVIEW,
            LifecycleApplicationOutcomeKind.MANUAL_REVIEW: FindingActionability.MANUAL_REVIEW,
            LifecycleApplicationOutcomeKind.QUARANTINED: FindingActionability.BLOCKING,
        }[kind]
        result.append(OperatorFinding(
            code=f"lifecycle.{kind.value}",
            domain="lifecycle",
            severity=FindingSeverity.ERROR,
            actionability=actionability,
            message=outcome.reason or f"Lifecycle operation is {kind.value}.",
            scope=scope,
            affected=(str(outcome.identity.parent_uuid),),
            evidence={"intent_id": outcome.intent_id, "chainID": outcome.identity.chain_id, "event": outcome.identity.event.value},
            guidance="Retry the lifecycle operation or inspect the durable intent evidence.",
        ))
    return deduplicate_findings(result)


def inspect_occurrence_collection(
    collection: Any,
    *,
    scope: OperatorScope | None = None,
) -> tuple[OperatorFinding, ...]:
    """Project one typed scheduler collection into actionable availability facts."""
    from .occurrence_outcomes import OccurrenceCollectionResult

    if not isinstance(collection, OccurrenceCollectionResult):
        raise TypeError("occurrence collection must be a typed result")
    if collection.status in {"found", "empty"}:
        return ()
    if collection.failure is not None:
        failure = collection.failure
        actionability = (
            FindingActionability.RETRYABLE if failure.status == "unavailable"
            else FindingActionability.ACTIONABLE
        )
        return (OperatorFinding(
            code=f"schedule.{failure.status}",
            domain="schedule",
            severity=FindingSeverity.ERROR,
            actionability=actionability,
            message=failure.reason,
            scope=scope,
            observed={"status": failure.status, "error_type": failure.error_type},
            expected={"status": "found or empty"},
            guidance="Retry with an available, valid scheduler context." if failure.status == "unavailable" else "Correct the recurrence expression and retry.",
        ),)
    terminal = collection.terminal
    return (OperatorFinding(
        code="schedule.exhausted",
        domain="schedule",
        severity=FindingSeverity.WARNING,
        actionability=FindingActionability.ACTIONABLE,
        message="Scheduler search reached its safety limit without a matching occurrence.",
        scope=scope,
        observed={"status": collection.status},
        expected={"status": "found or empty"},
        evidence={} if terminal is None else {"scope": terminal.scope, "kind": terminal.kind, "limit": terminal.limit},
        guidance="Narrow the recurrence range or raise the explicit scheduler limit.",
    ),)


def inspect_snapshot_coverage(
    snapshot: OperatorSnapshot,
    requirement: CoverageRequirement,
    *,
    scope: OperatorScope | None = None,
) -> tuple[OperatorFinding, ...]:
    """Report insufficient or unavailable evidence without treating it as absence."""
    if not isinstance(snapshot, OperatorSnapshot):
        raise TypeError("snapshot coverage inspection requires an OperatorSnapshot")
    if not isinstance(requirement, CoverageRequirement):
        raise TypeError("snapshot coverage inspection requires a CoverageRequirement")
    if snapshot.satisfies(requirement):
        return ()
    unavailable = snapshot.coverage.kind is CoverageKind.UNAVAILABLE
    return (
        OperatorFinding(
            code="snapshot.evidence_unavailable" if unavailable else "snapshot.coverage_insufficient",
            domain="snapshot",
            severity=FindingSeverity.ERROR,
            actionability=FindingActionability.BLOCKING,
            message=(
                snapshot.coverage.reason
                if unavailable and snapshot.coverage.reason
                else f"Snapshot coverage {snapshot.coverage.kind.value} does not satisfy {requirement.minimum.value}."
            ),
            scope=scope,
            observed={"coverage": snapshot.coverage.kind.value},
            expected={"coverage": requirement.minimum.value},
            evidence={"snapshot_id": snapshot.snapshot_id, "source": snapshot.coverage.source},
            guidance="Retry with an authoritative snapshot that satisfies the requested coverage.",
        ),
    )


def inspect_snapshot_consistency(
    snapshot: OperatorSnapshot,
    *,
    scope: OperatorScope | None = None,
) -> tuple[OperatorFinding, ...]:
    """Validate immutable component evidence and report any consistency fault."""
    if not isinstance(snapshot, OperatorSnapshot):
        raise TypeError("snapshot consistency inspection requires an OperatorSnapshot")
    try:
        snapshot.assert_consistent()
    except OperatorContractError as exc:
        return (
            OperatorFinding(
                code="snapshot.inconsistent",
                domain="snapshot",
                severity=FindingSeverity.ERROR,
                actionability=FindingActionability.BLOCKING,
                message=str(exc),
                scope=scope,
                observed={"snapshot_id": snapshot.snapshot_id},
                expected={"components": "same mutation epoch and configuration"},
                evidence={"coverage": snapshot.coverage.kind.value},
                guidance="Refresh the authoritative snapshot before continuing.",
            ),
        )
    return ()


def inspect_snapshot_limits(
    snapshot: OperatorSnapshot,
    limits: OperatorLimits,
    *,
    scope: OperatorScope | None = None,
) -> tuple[OperatorFinding, ...]:
    """Report snapshot indexes exceeding the caller's independent limits."""
    if not isinstance(snapshot, OperatorSnapshot) or not isinstance(limits, OperatorLimits):
        raise TypeError("snapshot limits inspection requires typed snapshot and limits")
    observed = {
        "tasks": len(snapshot.indexes.task_uuids),
        "chains": len(snapshot.indexes.chain_ids),
        "history_links": len(snapshot.indexes.links),
    }
    allowed = {name: getattr(limits, name) for name in observed}
    findings: list[OperatorFinding] = []
    for name in sorted(observed):
        if observed[name] <= allowed[name]:
            continue
        findings.append(
            OperatorFinding(
                code="snapshot.limit_exceeded",
                domain="snapshot",
                severity=FindingSeverity.ERROR,
                actionability=FindingActionability.BLOCKING,
                message=f"Snapshot contains {observed[name]} {name}, exceeding limit {allowed[name]}.",
                scope=scope,
                observed={name: observed[name]},
                expected={name: allowed[name]},
                evidence={"snapshot_id": snapshot.snapshot_id},
                guidance="Narrow the requested scope or raise the explicit operator limit.",
            )
        )
    return tuple(findings)


def inspect_component_availability(
    snapshot: OperatorSnapshot,
    component: str,
    *,
    scope: OperatorScope | None = None,
) -> tuple[OperatorFinding, ...]:
    """Inspect one required snapshot component without performing I/O."""
    if not isinstance(snapshot, OperatorSnapshot):
        raise TypeError("component inspection requires an OperatorSnapshot")
    name = str(component or "").strip()
    if not name:
        raise ValueError("component name is required")
    value = snapshot.components.get(name)
    if isinstance(value, dict) and value.get("available", True):
        return ()
    reason = value.get("reason", "unavailable") if isinstance(value, dict) else "component is absent"
    return (
        OperatorFinding(
            code="component.unavailable",
            domain=name,
            severity=FindingSeverity.ERROR,
            actionability=FindingActionability.BLOCKING,
            message=f"Required {name} evidence is unavailable: {reason}.",
            scope=scope,
            observed={"component": name, "available": False, "reason": reason},
            expected={"component": name, "available": True},
            evidence={"snapshot_id": snapshot.snapshot_id},
            guidance=f"Refresh the authoritative {name} evidence before continuing.",
        ),
    )


def inspect_component_validity(
    snapshot: OperatorSnapshot,
    component: str,
    *,
    scope: OperatorScope | None = None,
) -> tuple[OperatorFinding, ...]:
    """Report a present component that explicitly fails domain validation."""
    if not isinstance(snapshot, OperatorSnapshot):
        raise TypeError("component inspection requires an OperatorSnapshot")
    name = str(component or "").strip()
    if not name:
        raise ValueError("component name is required")
    value = snapshot.components.get(name)
    if not isinstance(value, dict) or value.get("available", True) is False or value.get("valid", True):
        return ()
    reason = str(value.get("reason") or "component validation failed")
    return (
        OperatorFinding(
            code="component.invalid",
            domain=name,
            severity=FindingSeverity.ERROR,
            actionability=FindingActionability.ACTIONABLE,
            message=f"{name} evidence is invalid: {reason}.",
            scope=scope,
            observed={"component": name, "valid": False, "reason": reason},
            expected={"component": name, "valid": True},
            evidence={"snapshot_id": snapshot.snapshot_id},
            guidance=f"Correct the {name} data or configuration, then retry.",
        ),
    )


def run_inspectors(snapshot: OperatorSnapshot, inspectors: Sequence[OperatorInspector]) -> tuple[OperatorFinding, ...]:
    """Run pure inspectors in declaration order and return stable findings."""
    if not isinstance(snapshot, OperatorSnapshot):
        raise TypeError("inspector execution requires an OperatorSnapshot")
    findings: list[OperatorFinding] = []
    for inspector in inspectors:
        inspect = getattr(inspector, "inspect", None)
        if not callable(inspect):
            raise TypeError("inspector must provide an inspect method")
        findings.extend(inspect(snapshot))
    return deduplicate_findings(findings)


def classify_historical(finding: OperatorFinding, *, active: bool) -> OperatorFinding:
    """Keep evidence while making inactive history non-blocking by default."""
    if not isinstance(finding, OperatorFinding):
        raise TypeError("historical classification requires an OperatorFinding")
    if active:
        return finding
    return replace(
        finding,
        severity=FindingSeverity.INFO,
        actionability=FindingActionability.DEFERRED,
        guidance=finding.guidance or "Retained for audit; review if this chain is reactivated.",
    )


def prioritize_findings(
    findings: Sequence[OperatorFinding],
    active_identities: set[str] | frozenset[str],
) -> tuple[OperatorFinding, ...]:
    """Order active findings before historical findings deterministically."""
    if not isinstance(active_identities, (set, frozenset)):
        raise TypeError("active identities must be a set")
    if any(not isinstance(item, OperatorFinding) for item in findings):
        raise TypeError("finding collection contains an invalid item")
    def key(item: OperatorFinding) -> tuple[int, int, str, str]:
        active = bool(set(item.affected) & set(active_identities))
        return (0 if active else 1, -_severity_rank(item.severity), item.domain, item.code)
    return tuple(sorted(findings, key=key))


STANDARD_COMPONENTS = (
    "configuration", "dependencies", "task_domain", "schedule",
    "chain_integrity", "lifecycle", "performance",
)


def inspect_standard_components(
    snapshot: OperatorSnapshot,
    *,
    scope: OperatorScope | None = None,
) -> tuple[OperatorFinding, ...]:
    """Inspect the standard control-plane components in stable order."""
    if not isinstance(snapshot, OperatorSnapshot):
        raise TypeError("standard inspection requires an OperatorSnapshot")
    findings: list[OperatorFinding] = []
    for component in STANDARD_COMPONENTS:
        findings.extend(inspect_component_availability(snapshot, component, scope=scope))
    return tuple(findings)


def aggregate_historical(findings: Sequence[OperatorFinding]) -> tuple[OperatorFinding, ...]:
    """Aggregate deferred findings by invariant while retaining affected IDs."""
    groups: dict[tuple[str, str, str], OperatorFinding] = {}
    counts: dict[tuple[str, str, str], int] = {}
    for finding in findings:
        if not isinstance(finding, OperatorFinding):
            raise TypeError("finding collection contains an invalid item")
        key = (finding.code, finding.domain, finding.message)
        existing = groups.get(key)
        if existing is None:
            groups[key] = finding
            counts[key] = 1
        else:
            groups[key] = replace(
                existing,
                affected=tuple(sorted(set(existing.affected) | set(finding.affected))),
                evidence={**existing.evidence, "aggregated_count": counts[key] + 1},
            )
            counts[key] += 1
    return tuple(groups[key] for key in sorted(groups))


def inspect_snapshot(
    snapshot: OperatorSnapshot,
    requirement: CoverageRequirement,
    limits: OperatorLimits,
    *,
    scope: OperatorScope | None = None,
) -> tuple[OperatorFinding, ...]:
    """Run the core pure snapshot checks in a fixed deterministic order."""
    findings = (
        *inspect_snapshot_coverage(snapshot, requirement, scope=scope),
        *inspect_snapshot_consistency(snapshot, scope=scope),
        *inspect_snapshot_limits(snapshot, limits, scope=scope),
    )
    return deduplicate_findings(findings)


def _severity_rank(value: FindingSeverity) -> int:
    return {FindingSeverity.INFO: 0, FindingSeverity.WARNING: 1, FindingSeverity.ERROR: 2}[value]


__all__ = ["STANDARD_COMPONENTS", "OperatorInspector", "ComponentValidityInspector", "InstallationInspector", "ConfigurationInspector", "DependenciesInspector", "ChainIntegrityInspector", "LifecycleOutboxInspector", "PerformanceInspector", "TaskDomainInspector", "ScheduleAvailabilityInspector", "inspect_integrity_findings", "inspect_lifecycle_outcomes", "inspect_occurrence_collection", "inspect_snapshot", "inspect_snapshot_coverage", "inspect_snapshot_consistency", "inspect_snapshot_limits", "inspect_component_availability", "inspect_component_validity", "inspect_standard_components", "classify_historical", "prioritize_findings", "aggregate_historical", "run_inspectors"]
