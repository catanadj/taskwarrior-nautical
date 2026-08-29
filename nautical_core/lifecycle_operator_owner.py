"""Operator owner for direct lifecycle-plan application."""

from __future__ import annotations

from typing import Protocol

from .lifecycle_application import LifecycleApplicationOutcome, LifecycleApplicationOutcomeKind
from .lifecycle_models import LifecycleAction, LifecyclePlan
from .operator_domain_plans import DomainApplicationAuthorization, require_domain_effect_plan
from .operator_models import OperatorFailure, OperatorOperation, OperatorResult, OperatorStatus


class LifecycleApplicationPort(Protocol):
    def stage(self, plan: LifecyclePlan, *, configuration_fingerprint: str, schedule_fingerprint: str) -> LifecycleApplicationOutcome: ...
    def drain(self, *, configuration_fingerprint: str, schedule_fingerprint: str): ...
    def apply_immediate(self, plan: LifecyclePlan) -> LifecycleApplicationOutcome: ...


class LifecycleOperatorOwner:
    """Stage and drain spawn plans, or apply immediate plans, in one owner."""

    def __init__(self, service: LifecycleApplicationPort) -> None:
        self._service = service

    def apply(self, authorization: DomainApplicationAuthorization) -> OperatorResult:
        plan = require_domain_effect_plan(authorization.plan)
        if not isinstance(plan, LifecyclePlan):
            raise TypeError("lifecycle owner requires a LifecyclePlan")
        if plan.action is LifecycleAction.SPAWN_CHILD:
            staged = self._service.stage(
                plan,
                configuration_fingerprint=authorization.configuration_fingerprint,
                schedule_fingerprint=authorization.schedule_fingerprint,
            )
            if not staged.ok:
                return self._result(staged.kind, staged.reason, staged.intent_id)
            drained = self._service.drain(
                configuration_fingerprint=authorization.configuration_fingerprint,
                schedule_fingerprint=authorization.schedule_fingerprint,
            )
            outcomes = tuple(getattr(drained, "outcomes", ()))
            final = outcomes[-1] if outcomes else staged
            return self._result(final.kind, final.reason, final.intent_id)
        outcome = self._service.apply_immediate(plan)
        return self._result(outcome.kind, outcome.reason, outcome.intent_id)

    @staticmethod
    def _result(kind: LifecycleApplicationOutcomeKind, reason: str, intent_id: str) -> OperatorResult:
        if kind in {LifecycleApplicationOutcomeKind.APPLIED, LifecycleApplicationOutcomeKind.ALREADY_APPLIED, LifecycleApplicationOutcomeKind.NOOP}:
            status = OperatorStatus.OK
            failure = None
        elif kind is LifecycleApplicationOutcomeKind.RETRYABLE:
            status = OperatorStatus.UNAVAILABLE
            failure = OperatorFailure("lifecycle_retryable", reason or "lifecycle application can be retried", retryable=True)
        else:
            status = OperatorStatus.MANUAL_REVIEW
            failure = OperatorFailure("lifecycle_manual_review", reason or "lifecycle application requires review", retryable=False)
        return OperatorResult(
            OperatorOperation.LIFECYCLE,
            status,
            data={"kind": kind.value, "intent_id": intent_id or ""},
            failure=failure,
        )


__all__ = ["LifecycleApplicationPort", "LifecycleOperatorOwner"]
