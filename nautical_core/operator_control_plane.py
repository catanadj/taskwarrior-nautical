"""Thin composition root for direct typed operator domain flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from collections.abc import Sequence

from .chain_integrity_models import IntegrityFinding, IntegrityRepairPlan
from .chain_repair_planner import IntegrityPlanningResult
from .lifecycle_models import LifecycleEvent, LifecyclePlan, TaskSnapshot
from .lifecycle_planner import CarryValidator, LifecyclePlanner, LifecyclePreflight
from .chain_repair_planner import IntegrityRepairPlanner
from .chain_integrity_engine import ChainIntegrityEngine
from .chain_integrity_recovery import RecoveryAudit
from .chain_generation import ChainGenerationService
from .operator_application import DomainApplicationRegistry
from .operator_domain_planner import OperatorDomainPlanner
from .operator_domain_plans import DomainApplicationAuthorization
from .operator_models import OperatorContractError, OperatorFailure, OperatorPhase, OperatorPhaseResult, OperatorResult
from .operator_inspectors import inspect_occurrence_collection, inspect_operator_snapshot
from .operator_findings import OperatorFinding
from .operator_models import CoverageRequirement, OperatorLimits, OperatorScope
from .operator_snapshot import OperatorSnapshot
from .operator_snapshot import OperatorSnapshotSession, SnapshotReadRequest, SnapshotReader
from .operator_snapshot_provider import OperatorSnapshotProvider
from .operator_context import OperatorBudgetLedger, OperatorInvocationContext
from .occurrence_outcomes import OccurrenceCollectionResult
from .task_models import TaskObservation
from .operator_health_service import OperatorHealthReport, OperatorHealthService


@dataclass(frozen=True, slots=True)
class OperatorControlPlane:
    """One typed planner/application composition root for operator commands."""

    planner: OperatorDomainPlanner
    applications: DomainApplicationRegistry
    configuration: Any | None = None

    @classmethod
    def from_configuration(cls, configuration: object, applications: DomainApplicationRegistry) -> "OperatorControlPlane":
        """Build the planner bundle from one already-validated configuration."""
        if configuration is None:
            raise ValueError("operator control plane requires validated configuration")
        if not isinstance(applications, DomainApplicationRegistry):
            raise TypeError("operator control plane requires a domain application registry")
        return cls(
            OperatorDomainPlanner(LifecyclePlanner(configuration), IntegrityRepairPlanner()),
            applications,
            configuration,
        )

    def plan_recovery(
        self,
        parent: TaskSnapshot,
        *,
        existing_children: tuple[TaskSnapshot, ...] | list[TaskSnapshot],
        hook: object,
        generation: ChainGenerationService | None = None,
    ) -> LifecyclePlan:
        """Build one lifecycle recovery plan through the central control plane."""
        configuration = self.configuration
        if configuration is None:
            raise ValueError("recovery planning requires validated configuration")
        engine = ChainIntegrityEngine.lifecycle_only(
            configuration_fingerprint=str(configuration.fingerprint),
            schedule_fingerprint=str(configuration.scheduler_fingerprint),
        )
        return engine.plan_recovery_plan(
            cast(Any, parent),
            existing_children=cast(Any, existing_children),
            hook=hook,
            generation=generation,
        )

    def plan_recovery_candidates(
        self,
        candidates: Sequence[TaskSnapshot],
        children_for: object,
        *,
        hook: object,
        generation: ChainGenerationService | None = None,
    ) -> tuple[LifecyclePlan, ...]:
        """Plan each recovery candidate through one shared control-plane owner."""
        if not callable(children_for):
            raise TypeError("children_for must be callable")
        return tuple(
            self.plan_recovery(
                parent,
                existing_children=children_for(parent),
                hook=hook,
                generation=generation,
            )
            for parent in candidates
        )

    def drain_integrity(
        self,
        outbox: object,
        *,
        unit_of_work: object,
        executor: object,
        request_factory: object,
        owner: str,
    ) -> tuple[object, ...]:
        """Drain durable integrity work through the control-plane engine."""
        configuration = self.configuration
        if configuration is None:
            raise ValueError("integrity drain requires validated configuration")
        snapshots = OperatorSnapshotProvider.for_unit_of_work(cast(Any, unit_of_work))
        engine = ChainIntegrityEngine(
            snapshots,
            configuration_fingerprint=str(configuration.fingerprint),
            schedule_fingerprint=str(configuration.scheduler_fingerprint),
        )
        return engine.drain(
            cast(Any, outbox),
            owner=owner,
            executor=cast(Any, executor),
            request_factory=cast(Any, request_factory),
        )

    def audit_native_until(self, rows: object, *, predecessor: object, safe_parse_datetime: object,
                           fmt_isoz: object, utc_to_local_naive: object, local_naive_to_utc: object) -> RecoveryAudit:
        """Audit native-until windows through the shared integrity engine."""
        engine = ChainIntegrityEngine.lifecycle_only(
            configuration_fingerprint="reconcile-recovery",
            schedule_fingerprint="reconcile-recovery",
        )
        return engine.audit_native_until(
            rows,
            predecessor=predecessor,
            safe_parse_datetime=safe_parse_datetime,
            fmt_isoz=fmt_isoz,
            utc_to_local_naive=utc_to_local_naive,
            local_naive_to_utc=local_naive_to_utc,
        )

    def apply_native_until(self, candidate: object, previous: object, item: object, **kwargs: object) -> object:
        """Apply one guarded native-until repair through the shared engine."""
        engine = ChainIntegrityEngine.lifecycle_only(
            configuration_fingerprint="reconcile-recovery",
            schedule_fingerprint="reconcile-recovery",
        )
        return engine.apply_native_until_candidate(candidate, previous, item, **kwargs)


    def plan_lifecycle(
        self,
        snapshot: TaskSnapshot,
        event: LifecycleEvent,
        *,
        preflight: LifecyclePreflight | None = None,
        carry_validator: CarryValidator | None = None,
    ) -> LifecyclePlan:
        return self.planner.plan_lifecycle(snapshot, event, preflight=preflight, carry_validator=carry_validator)

    def plan_integrity(self, context: object, findings: tuple[IntegrityFinding, ...]) -> IntegrityPlanningResult:
        return self.planner.plan_integrity(context, findings)

    def apply_domain(self, operation: str, authorization: DomainApplicationAuthorization) -> OperatorResult:
        return self.applications.apply(operation, authorization)

    def apply_domain_phases(
        self,
        operation: str,
        authorization: DomainApplicationAuthorization,
        *,
        budget: OperatorBudgetLedger | None = None,
    ) -> tuple[OperatorPhaseResult, ...]:
        """Apply one authorized domain plan through explicit typed phases."""
        try:
            if not isinstance(authorization, DomainApplicationAuthorization):
                raise OperatorContractError("domain application requires typed authorization")
        except (OperatorContractError, TypeError, ValueError) as exc:
            return (
                OperatorPhaseResult(
                    OperatorPhase.AUTHORIZE,
                    failure=OperatorFailure("invalid_authorization", str(exc), retryable=False),
                ),
            )
        if budget is not None and not budget.effect_started and budget.wall_time_exceeded:
            return (
                OperatorPhaseResult(
                    OperatorPhase.AUTHORIZE,
                    failure=OperatorFailure(
                        "application_limit_exceeded",
                        "operator wall-time budget expired before authorization",
                        retryable=True,
                        details={"resource": "wall_time_seconds"},
                    ),
                ),
            )
        phases = [OperatorPhaseResult(OperatorPhase.AUTHORIZE, value=authorization)]
        try:
            if budget is not None:
                if not callable(getattr(budget, "begin_effect", None)):
                    raise OperatorContractError("effect budget must support begin_effect")
                budget.begin_effect()
            result = self.apply_domain(operation, authorization)
            if not isinstance(result, OperatorResult):
                raise OperatorContractError("domain owner returned an untyped result")
        except Exception as exc:
            phases.append(
                OperatorPhaseResult(
                    OperatorPhase.APPLY,
                    failure=OperatorFailure("application_failed", str(exc), retryable=True),
                )
            )
            return tuple(phases)
        phases.extend((
            OperatorPhaseResult(OperatorPhase.APPLY, value=result),
            OperatorPhaseResult(OperatorPhase.VERIFY, value=result),
            OperatorPhaseResult(OperatorPhase.RESULT, value=result),
        ))
        return tuple(phases)

    def inspect(
        self,
        snapshot: OperatorSnapshot,
        requirement: CoverageRequirement,
        limits: OperatorLimits,
        *,
        scope: OperatorScope | None = None,
    ) -> tuple[OperatorFinding, ...]:
        """Inspect one immutable snapshot through the shared pure pipeline."""
        return inspect_operator_snapshot(snapshot, requirement, limits, scope=scope)

    def inspect_phases(
        self,
        snapshot: OperatorSnapshot,
        requirement: CoverageRequirement,
        limits: OperatorLimits,
        *,
        scope: OperatorScope | None = None,
    ) -> tuple[OperatorPhaseResult, ...]:
        """Run the read-only inspection slice with explicit typed phases."""
        try:
            if not isinstance(snapshot, OperatorSnapshot):
                raise OperatorContractError("inspection requires an OperatorSnapshot")
            if not isinstance(requirement, CoverageRequirement):
                raise OperatorContractError("inspection requires a coverage requirement")
            if not isinstance(limits, OperatorLimits):
                raise OperatorContractError("inspection requires operator limits")
            resolved_scope = scope or OperatorScope.system()
            if not isinstance(resolved_scope, OperatorScope):
                raise OperatorContractError("inspection scope is invalid")
        except (OperatorContractError, TypeError, ValueError) as exc:
            failure = OperatorFailure("invalid_request", str(exc), retryable=False)
            return (OperatorPhaseResult(OperatorPhase.VALIDATE_REQUEST, failure=failure),)

        phases: list[OperatorPhaseResult] = [
            OperatorPhaseResult(OperatorPhase.VALIDATE_REQUEST, value=True),
            OperatorPhaseResult(OperatorPhase.COMPILE_SCOPE, value=resolved_scope),
        ]
        findings = self.inspect(snapshot, requirement, limits, scope=resolved_scope)
        if len(findings) > limits.findings:
            failure = OperatorFailure(
                "inspection_limit_exceeded",
                f"inspection produced {len(findings)} findings, exceeding limit {limits.findings}",
                scope=resolved_scope,
                details={"resource": "findings", "observed": len(findings), "limit": limits.findings},
            )
            phases.append(OperatorPhaseResult(OperatorPhase.INSPECT, failure=failure))
            return tuple(phases)
        phases.extend((
            OperatorPhaseResult(OperatorPhase.INSPECT, value=findings),
            OperatorPhaseResult(OperatorPhase.RESULT, value=findings),
        ))
        return tuple(phases)

    def inspect_request_phases(
        self,
        context: OperatorInvocationContext,
        reader: SnapshotReader,
        request: SnapshotReadRequest,
    ) -> tuple[OperatorPhaseResult, ...]:
        """Acquire and inspect one exact snapshot within one invocation basis."""
        try:
            if not isinstance(context, OperatorInvocationContext):
                raise OperatorContractError("inspection requires an invocation context")
            if not callable(getattr(reader, "read", None)):
                raise OperatorContractError("inspection requires a snapshot reader")
            if not isinstance(request, SnapshotReadRequest):
                raise OperatorContractError("inspection requires a snapshot request")
            if request.scope != context.request.scope:
                raise OperatorContractError("snapshot scope differs from invocation request")
        except (OperatorContractError, TypeError, ValueError) as exc:
            return (
                OperatorPhaseResult(
                    OperatorPhase.VALIDATE_REQUEST,
                    failure=OperatorFailure("invalid_request", str(exc), retryable=False),
                ),
            )
        phases: list[OperatorPhaseResult] = [
            OperatorPhaseResult(OperatorPhase.VALIDATE_REQUEST, value=True),
            OperatorPhaseResult(OperatorPhase.CAPTURE_CONTEXT, value=context),
            OperatorPhaseResult(OperatorPhase.COMPILE_SCOPE, value=request.scope),
        ]
        snapshot = OperatorSnapshotSession(context, reader).read(request)
        if isinstance(snapshot, OperatorFailure):
            phases.append(OperatorPhaseResult(OperatorPhase.ACQUIRE_SNAPSHOT, failure=snapshot))
            return tuple(phases)
        phases.append(OperatorPhaseResult(OperatorPhase.ACQUIRE_SNAPSHOT, value=snapshot))
        findings = self.inspect(snapshot, request.requirement, request.limits, scope=request.scope)
        if len(findings) > request.limits.findings:
            phases.append(
                OperatorPhaseResult(
                    OperatorPhase.INSPECT,
                    failure=OperatorFailure(
                        "inspection_limit_exceeded",
                        f"inspection produced {len(findings)} findings, exceeding limit {request.limits.findings}",
                        scope=request.scope,
                        details={"resource": "findings", "observed": len(findings), "limit": request.limits.findings},
                    ),
                )
            )
            return tuple(phases)
        phases.extend((
            OperatorPhaseResult(OperatorPhase.INSPECT, value=findings),
            OperatorPhaseResult(OperatorPhase.RESULT, value=findings),
        ))
        return tuple(phases)

    def inspect_occurrences(
        self,
        collection: OccurrenceCollectionResult,
        *,
        scope: OperatorScope | None = None,
    ) -> tuple[OperatorFinding, ...]:
        """Project one typed scheduler collection through the operator boundary."""
        return inspect_occurrence_collection(collection, scope=scope)

    def audit_integrity(
        self, unit_of_work: object, rows: Sequence[TaskObservation]
    ) -> tuple[object | None, list[dict[str, object]]]:
        """Audit an authoritative task snapshot through the shared integrity service."""
        from .integrity_audit_service import audit_authoritative_rows

        return audit_authoritative_rows(cast(Any, unit_of_work), rows)

    def diagnose_chains(
        self,
        unit_of_work: object,
        *,
        budget: OperatorBudgetLedger | None = None,
    ) -> tuple[dict[str, int], list[dict[str, object]]]:
        """Export and audit chain state as one read-only diagnosis request."""
        from .integration_models import Absent, Found, Unavailable

        repository = getattr(unit_of_work, "repository", None)
        if repository is None:
            return {"tasks": 0, "nautical_tasks": 0, "chains": 0}, [{
                "id": "chains.export",
                "severity": "error",
                "message": "Task data could not be exported for chain inspection.",
                "details": {"error": "validated repository is unavailable"},
            }]
        from .task_read_repository import ALL_TASK_STATUSES

        repository.configure_commands(timeout=120.0, attempts=2, retry_delay=0.05)
        if budget is not None and not budget.consume("taskwarrior_calls"):
            return {"tasks": 0, "nautical_tasks": 0, "chains": 0}, [{
                "id": "chains.budget",
                "severity": "error",
                "message": "Taskwarrior call budget exhausted before chain export.",
                "details": {"resource": "taskwarrior_calls", "observed": budget.usage("taskwarrior_calls")},
            }]
        read = repository.lifecycle_candidates(
            statuses=ALL_TASK_STATUSES,
            scope_filter=None,
            bounded=False,
        )
        if isinstance(read, Unavailable):
            return {"tasks": 0, "nautical_tasks": 0, "chains": 0}, [{
                "id": "chains.export",
                "severity": "error",
                "message": "Task data could not be exported for chain inspection.",
                "details": {"error": read.evidence.detail},
            }]
        rows: list[TaskObservation] = list(read.value) if isinstance(read, Found) else []
        if not isinstance(read, (Found, Absent)):
            return {"tasks": 0, "nautical_tasks": 0, "chains": 0}, [{
                "id": "chains.export",
                "severity": "error",
                "message": "Task data could not be exported for chain inspection.",
                "details": {"error": "task repository returned an invalid result"},
            }]
        chain_count = len({row.field("chainID").raw_value() for row in rows if row.field("chainID").raw_value()})
        if budget is not None and (
            (len(rows) and not budget.consume("tasks", len(rows)))
            or (chain_count and not budget.consume("chains", chain_count))
        ):
            return {"tasks": len(rows), "nautical_tasks": len(rows), "chains": chain_count}, [{
                "id": "chains.budget",
                "severity": "error",
                "message": "Chain audit resource budget exhausted before integrity inspection.",
                "details": {"resource": "tasks/chains", "observed_tasks": len(rows), "observed_chains": chain_count},
            }]
        _, findings = self.audit_integrity(unit_of_work, rows)
        recurrence_fields = ("cp", "anchor", "anchor_file")
        def value(row: TaskObservation, field: str) -> object:
            raw = row.field(field).value
            return getattr(raw, "value", raw)
        nautical = [
            row for row in rows
            if any(str(value(row, field) or "").strip() for field in recurrence_fields)
            or str(value(row, "chainID") or "").strip()
        ]
        counts = {
            "tasks": len(rows),
            "nautical_tasks": len(nautical),
            "chains": len({str(value(row, "chainID")) for row in nautical if value(row, "chainID")}),
        }
        return counts, findings

    @staticmethod
    def health_report(findings: Sequence[OperatorFinding]) -> OperatorHealthReport:
        """Aggregate already-observed health findings without performing I/O."""
        return OperatorHealthService.report(findings)


__all__ = ["OperatorControlPlane"]
