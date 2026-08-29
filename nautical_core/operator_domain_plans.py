"""Direct typed plan boundary for operator-owned domain effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .chain_integrity_models import IntegrityRepairPlan
from .lifecycle_models import LifecyclePlan
from .operator_models import OperatorContractError, OperatorCoverage, OperatorRequest, OperatorScope


DomainEffectPlan: TypeAlias = LifecyclePlan | IntegrityRepairPlan


@dataclass(frozen=True, slots=True)
class DomainApplicationAuthorization:
    """Evidence envelope carrying a typed domain plan to an effect owner."""

    plan: DomainEffectPlan
    request: OperatorRequest
    snapshot_id: str
    configuration_fingerprint: str
    scope: OperatorScope
    coverage: OperatorCoverage
    schedule_fingerprint: str = ""

    def __post_init__(self) -> None:
        require_domain_effect_plan(self.plan)
        if not isinstance(self.request, OperatorRequest) or not self.request.apply:
            raise OperatorContractError("domain application requires an applying operator request")
        if not str(self.snapshot_id or "").strip() or not str(self.configuration_fingerprint or "").strip():
            raise OperatorContractError("domain application requires snapshot and configuration evidence")
        if not str(self.schedule_fingerprint or "").strip():
            raise OperatorContractError("domain application requires a schedule fingerprint")
        if not isinstance(self.scope, OperatorScope) or self.scope != self.request.scope:
            raise OperatorContractError("domain application scope differs from operator request")
        if not isinstance(self.coverage, OperatorCoverage) or not self.request.coverage.accepts(self.coverage):
            raise OperatorContractError("domain application coverage does not satisfy operator request")


def require_domain_effect_plan(value: object) -> DomainEffectPlan:
    """Reject generic payloads at the domain application boundary."""
    if isinstance(value, (LifecyclePlan, IntegrityRepairPlan)):
        return value
    raise TypeError("operator effect requires a typed LifecyclePlan or IntegrityRepairPlan")


__all__ = ["DomainApplicationAuthorization", "DomainEffectPlan", "require_domain_effect_plan"]
