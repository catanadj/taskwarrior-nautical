"""Thin composition root for direct typed operator domain flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .chain_integrity_models import IntegrityFinding, IntegrityRepairPlan
from .chain_repair_planner import IntegrityPlanningResult
from .lifecycle_models import LifecycleEvent, LifecyclePlan, TaskSnapshot
from .lifecycle_planner import CarryValidator, LifecyclePlanner, LifecyclePreflight
from .chain_repair_planner import IntegrityRepairPlanner
from .chain_integrity_engine import ChainIntegrityEngine
from .chain_generation import ChainGenerationService
from .operator_application import DomainApplicationRegistry
from .operator_domain_planner import OperatorDomainPlanner
from .operator_domain_plans import DomainApplicationAuthorization
from .operator_models import OperatorResult
from .operator_inspectors import inspect_occurrence_collection, inspect_operator_snapshot
from .operator_findings import OperatorFinding
from .operator_models import CoverageRequirement, OperatorLimits, OperatorScope
from .operator_snapshot import OperatorSnapshot
from .occurrence_outcomes import OccurrenceCollectionResult


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
            parent,
            existing_children=existing_children,
            hook=hook,
            generation=generation,
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
        from .chain_snapshot import ChainSnapshotService

        engine = ChainIntegrityEngine(
            ChainSnapshotService(
                unit_of_work,
                configuration_fingerprint=str(configuration.fingerprint),
            ),
            configuration_fingerprint=str(configuration.fingerprint),
            schedule_fingerprint=str(configuration.scheduler_fingerprint),
        )
        return engine.drain(
            outbox,
            owner=owner,
            executor=executor,
            request_factory=request_factory,
        )


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

    def inspect_occurrences(
        self,
        collection: OccurrenceCollectionResult,
        *,
        scope: OperatorScope | None = None,
    ) -> tuple[OperatorFinding, ...]:
        """Project one typed scheduler collection through the operator boundary."""
        return inspect_occurrence_collection(collection, scope=scope)


__all__ = ["OperatorControlPlane"]
