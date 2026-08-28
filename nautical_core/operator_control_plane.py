"""Thin composition root for direct typed operator domain flows."""

from __future__ import annotations

from dataclasses import dataclass

from .chain_integrity_models import IntegrityFinding, IntegrityRepairPlan
from .chain_repair_planner import IntegrityPlanningResult
from .lifecycle_models import LifecycleEvent, LifecyclePlan, TaskSnapshot
from .lifecycle_planner import CarryValidator, LifecyclePlanner, LifecyclePreflight
from .chain_repair_planner import IntegrityRepairPlanner
from .operator_application import DomainApplicationRegistry
from .operator_domain_planner import OperatorDomainPlanner
from .operator_domain_plans import DomainApplicationAuthorization
from .operator_models import OperatorResult
from .operator_inspectors import inspect_operator_snapshot
from .operator_findings import OperatorFinding
from .operator_models import CoverageRequirement, OperatorLimits, OperatorScope
from .operator_snapshot import OperatorSnapshot


@dataclass(frozen=True, slots=True)
class OperatorControlPlane:
    """One typed planner/application composition root for operator commands."""

    planner: OperatorDomainPlanner
    applications: DomainApplicationRegistry

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


__all__ = ["OperatorControlPlane"]
