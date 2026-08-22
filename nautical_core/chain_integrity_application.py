"""Guarded application adapter for integrity repair plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .chain_integrity_models import IntegrityOperation, IntegrityRepairPlan, RepairOperationKind, RepairSafety
from .integration_models import (
    MetadataRepairPayload,
    MutationOperation,
    MutationOutcome,
    MutationOutcomeKind,
    MutationRequest,
)


class IntegrityMutationExecutor(Protocol):
    """Typed mutation-capable boundary owned by the integration layer."""

    def repair_metadata(self, request: MutationRequest) -> MutationOutcome: ...


class IntegrityMutationRequestFactory(Protocol):
    def __call__(self, operation: IntegrityOperation) -> MutationRequest: ...


@dataclass(frozen=True, slots=True)
class IntegrityOutboxPersistResult:
    accepted: bool
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "reason", str(self.reason or "").strip())
        if not self.accepted and not self.reason:
            raise ValueError("rejected integrity outbox persistence requires a reason")


class IntegrityOutboxSink(Protocol):
    """Durable owner for multi-operation integrity plans."""

    def persist(self, plan: IntegrityRepairPlan) -> IntegrityOutboxPersistResult: ...


class RepositoryIntegrityOutboxSink:
    """Adapt the repository-owned shared outbox to the planner boundary."""

    def __init__(self, repository: object, *, configuration_fingerprint: str, schedule_fingerprint: str) -> None:
        self._repository = repository
        self._configuration_fingerprint = str(configuration_fingerprint or "").strip()
        self._schedule_fingerprint = str(schedule_fingerprint or "").strip()
        if not self._configuration_fingerprint or not self._schedule_fingerprint:
            raise ValueError("integrity outbox sink requires configuration and schedule fingerprints")

    def persist(self, plan: IntegrityRepairPlan) -> IntegrityOutboxPersistResult:
        from .integrity_outbox_envelope import IntegrityOutboxEnvelope

        enqueue = getattr(self._repository, "enqueue_integrity", None)
        if not callable(enqueue):
            return IntegrityOutboxPersistResult(False, "repository does not support integrity outbox work")
        try:
            result = enqueue(IntegrityOutboxEnvelope(
                plan, self._configuration_fingerprint, self._schedule_fingerprint,
            ))
        except Exception as exc:
            return IntegrityOutboxPersistResult(False, f"integrity outbox persistence failed: {type(exc).__name__}: {exc}")
        return IntegrityOutboxPersistResult(bool(getattr(result, "ok", False)), str(getattr(result, "reason", "")))


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

    @property
    def stale(self) -> bool:
        """Whether the plan's guard became stale rather than malformed."""
        return self.kind is MutationOutcomeKind.CONFLICT and (
            self.reason.startswith("guard ") or "mutation epoch changed" in self.reason
        )


class IntegrityApplicationService:
    """Apply only supported typed operations through the mutation gateway."""

    def apply(
        self,
        plan: IntegrityRepairPlan,
        executor: IntegrityMutationExecutor,
        request_factory: IntegrityMutationRequestFactory,
        outbox: IntegrityOutboxSink | None = None,
    ) -> tuple[IntegrityApplicationResult, ...]:
        if not isinstance(plan, IntegrityRepairPlan):
            raise TypeError("integrity application requires an IntegrityRepairPlan")
        invalid = self.validate_plan(plan)
        if invalid:
            return tuple(IntegrityApplicationResult(
                plan.plan_id,
                operation.operation_id,
                MutationOutcomeKind.MANUAL_REVIEW,
                invalid,
            ) for operation in plan.operations)
        results: list[IntegrityApplicationResult] = []
        if len(plan.operations) > 1:
            if outbox is None:
                return tuple(IntegrityApplicationResult(
                    plan.plan_id,
                    operation.operation_id,
                    MutationOutcomeKind.MANUAL_REVIEW,
                    "multi-operation integrity plan requires outbox persistence",
                ) for operation in plan.operations)
            try:
                persisted = outbox.persist(plan)
            except Exception as exc:
                persisted = IntegrityOutboxPersistResult(False, f"outbox persistence failed: {type(exc).__name__}: {exc}")
            if not persisted.accepted:
                return tuple(IntegrityApplicationResult(
                    plan.plan_id,
                    operation.operation_id,
                    MutationOutcomeKind.MANUAL_REVIEW,
                    persisted.reason,
                ) for operation in plan.operations)
        for operation in plan.operations:
            if operation.kind not in {RepairOperationKind.METADATA_REPAIR, RepairOperationKind.LINK_REPAIR}:
                results.append(IntegrityApplicationResult(
                    plan.plan_id,
                    operation.operation_id,
                    MutationOutcomeKind.MANUAL_REVIEW,
                    f"integrity operation {operation.kind.value} has no application adapter",
                ))
                continue
            try:
                request = request_factory(operation)
                self._validate_request(operation, request)
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

    @staticmethod
    def validate_plan(plan: IntegrityRepairPlan) -> str:
        """Return an actionable refusal for a plan outside the apply contract."""
        if plan.safety is not RepairSafety.SAFE:
            return "integrity application accepts only SAFE repair plans"
        if not plan.configuration_fingerprint:
            return "integrity repair plan has no configuration fingerprint"
        return ""

    @staticmethod
    def _validate_request(operation: IntegrityOperation, request: MutationRequest) -> None:
        if not isinstance(request, MutationRequest):
            raise TypeError("integrity request factory returned an untyped request")
        if request.operation is not MutationOperation.METADATA_REPAIR:
            raise ValueError("integrity structural repair requires a metadata mutation request")
        if request.guard.task_uuid != operation.target_uuid or request.guard.chain_id != operation.chain_id:
            raise ValueError("mutation guard does not match integrity operation target")
        if not isinstance(request.payload, MetadataRepairPayload):
            raise TypeError("integrity metadata repair requires MetadataRepairPayload")
        if request.payload.to_dict() != dict(operation.payload):
            raise ValueError("metadata request payload differs from integrity operation")


__all__ = [
    "IntegrityApplicationResult",
    "IntegrityMutationExecutor",
    "IntegrityApplicationService",
    "IntegrityMutationRequestFactory",
    "IntegrityOutboxPersistResult",
    "IntegrityOutboxSink",
    "RepositoryIntegrityOutboxSink",
]
