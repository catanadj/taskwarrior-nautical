"""Typed hand-off from operator orchestration to domain planners."""

from __future__ import annotations

from typing import Any

from .chain_integrity_models import IntegrityFinding, IntegrityRepairPlan
from .chain_repair_planner import IntegrityPlanningResult, IntegrityRepairPlanner
from .lifecycle_models import LifecycleEvent, LifecyclePlan, TaskSnapshot
from .lifecycle_planner import CarryValidator, LifecyclePlanner, LifecyclePreflight
from .operator_domain_plans import DomainEffectPlan, require_domain_effect_plan


class OperatorDomainPlanner:
    """Expose lifecycle and integrity planning through one typed facade."""

    def __init__(self, lifecycle: LifecyclePlanner, integrity: IntegrityRepairPlanner) -> None:
        self._lifecycle = lifecycle
        self._integrity = integrity

    def plan_lifecycle(
        self,
        snapshot: TaskSnapshot,
        event: LifecycleEvent,
        *,
        preflight: LifecyclePreflight | None = None,
        carry_validator: CarryValidator | None = None,
    ) -> LifecyclePlan:
        plan = self._lifecycle.plan(snapshot, event, preflight=preflight, carry_validator=carry_validator)
        if not isinstance(plan, LifecyclePlan):
            raise TypeError("lifecycle planner returned an untyped plan")
        return plan

    def plan_integrity(self, context: Any, findings: tuple[IntegrityFinding, ...]) -> IntegrityPlanningResult:
        result = self._integrity.plan(context, findings)
        if not isinstance(result, IntegrityPlanningResult):
            raise TypeError("integrity planner returned an untyped result")
        if any(not isinstance(plan, IntegrityRepairPlan) for plan in result.plans):
            raise TypeError("integrity planner returned an untyped plan")
        return result

    @staticmethod
    def require_effect_plan(value: object) -> DomainEffectPlan:
        return require_domain_effect_plan(value)


__all__ = ["OperatorDomainPlanner"]
