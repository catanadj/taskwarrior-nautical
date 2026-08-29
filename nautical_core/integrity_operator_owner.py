"""Operator owner for direct chain-integrity repair plans."""

from __future__ import annotations

from .chain_integrity_application import IntegrityApplicationService, IntegrityMutationRequestFactory, IntegrityOutboxSink, IntegrityMutationExecutor
from .chain_integrity_models import IntegrityRepairPlan
from .operator_domain_plans import DomainApplicationAuthorization, require_domain_effect_plan
from .operator_models import OperatorFailure, OperatorOperation, OperatorResult, OperatorStatus
from .integration_models import GuardTimestamp, GuardTimestampField, MutationGuard, MutationRequest, Found
from .task_changes import TaskPatch
from .task_models import FieldPresence, TaskObservation, TaskUUID
from .lifecycle_models import recurrence_fingerprint


def build_integrity_mutation_request(operation: object, *, unit_of_work: object) -> MutationRequest:
    """Build one fresh guarded metadata request for an integrity operation."""
    repository = getattr(unit_of_work, "repository", None)
    if repository is None:
        raise RuntimeError("integrity repair requires a task repository")
    target_uuid = str(getattr(operation, "target_uuid", "") or "").strip()
    read = repository.by_uuid(target_uuid, refresh=True)
    if not isinstance(read, Found) or not isinstance(read.value, TaskObservation):
        raise RuntimeError(f"integrity target {target_uuid} is unavailable")
    row = read.value

    def field_value(name: str) -> object:
        state = row.field(name)
        return None if state.presence is FieldPresence.ABSENT else state.raw_value()

    modified = str(field_value("modified") or "").strip()
    if not modified:
        raise RuntimeError("integrity target has no modified timestamp")
    link = int(field_value("link") or 0)
    if link < 0:
        raise RuntimeError("integrity target has an invalid link")
    updates = dict(getattr(operation, "payload", {}) or {})
    expected = {key: field_value(key) for key in updates}
    guard = MutationGuard(
        task_uuid=str(field_value("uuid") or target_uuid),
        status=str(field_value("status") or "pending"),
        chain_id=str(field_value("chainID") or getattr(operation, "chain_id", "")),
        link=link,
        recurrence_identity=recurrence_fingerprint(row.to_mapping()),
        timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, modified),),
        expected_mutation_epoch=int(getattr(unit_of_work, "mutation_epoch", 0)),
        chain=str(field_value("chain") or "on"),
    )
    patch = TaskPatch.metadata_repair(TaskUUID(guard.task_uuid), **updates)
    return MutationRequest.metadata_repair(guard, patch, expected=expected)


class IntegrityOperatorOwner:
    """Apply one typed integrity plan through its established application owner."""

    def __init__(
        self,
        service: IntegrityApplicationService,
        *,
        executor: IntegrityMutationExecutor,
        request_factory: IntegrityMutationRequestFactory,
        outbox: IntegrityOutboxSink | None = None,
    ) -> None:
        self._service = service
        self._executor = executor
        self._request_factory = request_factory
        self._outbox = outbox

    def apply(self, authorization: DomainApplicationAuthorization) -> OperatorResult:
        plan = require_domain_effect_plan(authorization.plan)
        if not isinstance(plan, IntegrityRepairPlan):
            raise TypeError("integrity owner requires an IntegrityRepairPlan")
        applications = self._service.apply(
            plan,
            self._executor,
            self._request_factory,
            self._outbox,
        )
        if not applications:
            return OperatorResult(
                OperatorOperation.INTEGRITY,
                OperatorStatus.MANUAL_REVIEW,
                failure=OperatorFailure("integrity_no_result", "integrity application returned no outcome"),
            )
        final = applications[-1]
        if final.kind.value in {"applied", "already_applied"}:
            status = OperatorStatus.OK
            failure = None
        elif final.kind.value in {"conflict", "manual_review"}:
            status = OperatorStatus.MANUAL_REVIEW
            failure = OperatorFailure("integrity_manual_review", final.reason or "integrity repair requires review")
        else:
            status = OperatorStatus.UNAVAILABLE
            failure = OperatorFailure("integrity_retryable", final.reason or "integrity repair can be retried", retryable=True)
        return OperatorResult(
            OperatorOperation.INTEGRITY,
            status,
            data={"plan_id": plan.plan_id, "applications": len(applications)},
            failure=failure,
        )


__all__ = ["IntegrityOperatorOwner", "build_integrity_mutation_request"]
