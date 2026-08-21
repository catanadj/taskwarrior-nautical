"""Guarded application adapter for integrity repair plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .chain_integrity_models import IntegrityOperation, IntegrityRepairPlan, RepairOperationKind
from .integration_models import MutationOutcome, MutationOutcomeKind, MutationRequest


class _MutationExecutor(Protocol):
    def repair_metadata(self, request: MutationRequest) -> MutationOutcome: ...


class _MutationRequestFactory(Protocol):
    def __call__(self, operation: IntegrityOperation) -> MutationRequest: ...


@dataclass(frozen=True, slots=True)
class IntegrityApplicationResult:
    plan_id: str
    operation_id: str
    kind: MutationOutcomeKind
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", str(self.plan_id or "").strip())
        object.__setattr__(self, "operation_id", str(self.operation_id or "").strip())
        if not self.plan_id or not self.operation_id:
            raise ValueError("integrity application result requires plan and operation IDs")
        object.__setattr__(self, "kind", MutationOutcomeKind(self.kind))
        object.__setattr__(self, "reason", str(self.reason or "").strip())


class IntegrityApplicationService:
    """Apply only supported typed operations through the mutation gateway."""

    def apply(
        self,
        plan: IntegrityRepairPlan,
        executor: _MutationExecutor,
        request_factory: _MutationRequestFactory,
    ) -> tuple[IntegrityApplicationResult, ...]:
        if not isinstance(plan, IntegrityRepairPlan):
            raise TypeError("integrity application requires an IntegrityRepairPlan")
        results: list[IntegrityApplicationResult] = []
        for operation in plan.operations:
            if operation.kind is not RepairOperationKind.METADATA_REPAIR:
                results.append(IntegrityApplicationResult(
                    plan.plan_id,
                    operation.operation_id,
                    MutationOutcomeKind.MANUAL_REVIEW,
                    f"integrity operation {operation.kind.value} has no application adapter",
                ))
                continue
            try:
                request = request_factory(operation)
                outcome = executor.repair_metadata(request)
            except Exception as exc:
                results.append(IntegrityApplicationResult(
                    plan.plan_id,
                    operation.operation_id,
                    MutationOutcomeKind.MANUAL_REVIEW,
                    f"mutation adapter failed: {type(exc).__name__}: {exc}",
                ))
                continue
            results.append(IntegrityApplicationResult(
                plan.plan_id,
                operation.operation_id,
                outcome.kind,
                outcome.reason,
            ))
            if outcome.kind not in {MutationOutcomeKind.APPLIED, MutationOutcomeKind.ALREADY_APPLIED}:
                break
        return tuple(results)


__all__ = ["IntegrityApplicationResult", "IntegrityApplicationService"]
