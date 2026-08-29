"""Operator owner for direct chain-integrity repair plans."""

from __future__ import annotations

from .chain_integrity_application import IntegrityApplicationService, IntegrityMutationRequestFactory, IntegrityOutboxSink, IntegrityMutationExecutor
from .chain_integrity_models import IntegrityRepairPlan
from .operator_domain_plans import DomainApplicationAuthorization, require_domain_effect_plan
from .operator_models import OperatorFailure, OperatorOperation, OperatorResult, OperatorStatus


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


__all__ = ["IntegrityOperatorOwner"]
