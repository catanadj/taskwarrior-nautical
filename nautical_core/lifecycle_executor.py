"""Typed, idempotent execution contract for one lifecycle transition.

The executor owns mutation order and compensation policy, while adapters own
Taskwarrior reads/writes.  This keeps on-exit and reconcile on one sequence
without importing either operator or hook implementation details here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from nautical_core.lifecycle_models import (
    ExecutionStage,
    LifecycleAction,
    LifecycleContractError,
    LifecycleOutcome,
    LifecycleOutcomeKind,
    LifecyclePlan,
)


class OperationState(str, Enum):
    APPLIED = "applied"
    ALREADY = "already"
    FOUND = "found"
    ABSENT = "absent"
    UNAVAILABLE = "unavailable"
    CONFLICT = "conflict"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class OperationResult:
    state: OperationState
    value: Any = None
    reason: str = ""

    @property
    def successful(self) -> bool:
        return self.state in {
            OperationState.APPLIED,
            OperationState.ALREADY,
            OperationState.FOUND,
        }


class LifecycleExecutorServices(Protocol):
    def validate_parent(self, plan: LifecyclePlan) -> OperationResult: ...

    def find_equivalent_child(self, plan: LifecyclePlan) -> OperationResult: ...

    def import_child(self, plan: LifecyclePlan) -> OperationResult: ...

    def verify_child(self, plan: LifecyclePlan, child: dict[str, Any]) -> OperationResult: ...

    def apply_parent_patch(self, plan: LifecyclePlan, child: dict[str, Any]) -> OperationResult: ...

    def verify_linkage(self, plan: LifecyclePlan, child: dict[str, Any]) -> OperationResult: ...

    def compensate_child(self, plan: LifecyclePlan, child: dict[str, Any]) -> OperationResult: ...


class LifecycleTerminalServices(Protocol):
    def validate_terminal(self, plan: LifecyclePlan) -> OperationResult: ...

    def disable_chain(self, plan: LifecyclePlan) -> OperationResult: ...

    def verify_terminal(self, plan: LifecyclePlan) -> OperationResult: ...


def _reason(result: OperationResult, fallback: str) -> str:
    return str(result.reason or fallback).strip()


class LifecycleTransitionExecutor:
    """Run one new-format spawn/terminal plan in a fixed, idempotent order."""

    def __init__(self, services: LifecycleExecutorServices) -> None:
        self.services = services

    def _outcome(
        self,
        plan: LifecyclePlan,
        kind: LifecycleOutcomeKind,
        stage: ExecutionStage,
        reason: str = "",
    ) -> LifecycleOutcome:
        return LifecycleOutcome(kind=kind, stage=stage, identity=plan.identity, reason=reason)

    def _failure(
        self,
        plan: LifecyclePlan,
        child: dict[str, Any] | None,
        imported: bool,
        result: OperationResult,
        fallback: str,
    ) -> LifecycleOutcome:
        reason = _reason(result, fallback)
        if result.state is OperationState.UNAVAILABLE:
            return self._outcome(plan, LifecycleOutcomeKind.RETRYABLE, ExecutionStage.RETRYABLE, reason)
        if imported and child is not None:
            compensation = self.services.compensate_child(plan, child)
            if not compensation.successful:
                reason = f"{reason}; compensation failed: {_reason(compensation, 'unknown error')}"
        return self._outcome(plan, LifecycleOutcomeKind.MANUAL_REVIEW, ExecutionStage.MANUAL_REVIEW, reason)

    def execute(self, plan: LifecyclePlan) -> LifecycleOutcome:
        if not isinstance(plan, LifecyclePlan):
            raise LifecycleContractError("lifecycle executor requires a validated plan")
        if plan.action is LifecycleAction.NOOP:
            return self._outcome(plan, LifecycleOutcomeKind.NOOP, ExecutionStage.FINALIZED)
        if plan.action in {LifecycleAction.FINALIZE_CHAIN, LifecycleAction.DISABLE_CHAIN}:
            return self._outcome(plan, LifecycleOutcomeKind.TERMINAL, ExecutionStage.FINALIZED)
        if plan.action is not LifecycleAction.SPAWN_CHILD:
            return self._outcome(
                plan,
                LifecycleOutcomeKind.MANUAL_REVIEW,
                ExecutionStage.MANUAL_REVIEW,
                f"unsupported executor action: {plan.action.value}",
            )

        parent = self.services.validate_parent(plan)
        if not parent.successful:
            return self._failure(plan, None, False, parent, "parent guard failed")

        child: dict[str, Any] | None = None
        imported = False
        lookup = self.services.find_equivalent_child(plan)
        if lookup.state is OperationState.UNAVAILABLE:
            return self._failure(plan, None, False, lookup, "child lookup unavailable")
        if lookup.state is OperationState.FOUND:
            if not isinstance(lookup.value, dict):
                return self._outcome(
                    plan,
                    LifecycleOutcomeKind.MANUAL_REVIEW,
                    ExecutionStage.MANUAL_REVIEW,
                    "equivalent child lookup returned invalid data",
                )
            child = dict(lookup.value)
        elif lookup.state is OperationState.ABSENT:
            imported_result = self.services.import_child(plan)
            if not imported_result.successful:
                return self._failure(plan, None, False, imported_result, "child import failed")
            child = dict(imported_result.value or plan.child_dict())
            imported = True
        else:
            return self._failure(plan, None, False, lookup, "child lookup failed")

        verified_child = self.services.verify_child(plan, child)
        if not verified_child.successful:
            return self._failure(plan, child, imported, verified_child, "child verification failed")

        parent_patch = self.services.apply_parent_patch(plan, child)
        if not parent_patch.successful:
            return self._failure(plan, child, imported, parent_patch, "parent patch failed")

        linkage = self.services.verify_linkage(plan, child)
        if not linkage.successful:
            return self._failure(plan, child, imported, linkage, "parent linkage verification failed")

        return self._outcome(plan, LifecycleOutcomeKind.APPLIED, ExecutionStage.FINALIZED)


class LifecycleTerminalExecutor:
    """Apply one guarded terminal transition and verify chain disablement."""

    def __init__(self, services: LifecycleTerminalServices) -> None:
        self.services = services

    @staticmethod
    def _failure(
        plan: LifecyclePlan,
        result: OperationResult,
        fallback: str,
    ) -> LifecycleOutcome:
        reason = _reason(result, fallback)
        kind = (
            LifecycleOutcomeKind.RETRYABLE
            if result.state is OperationState.UNAVAILABLE
            else LifecycleOutcomeKind.MANUAL_REVIEW
        )
        stage = (
            ExecutionStage.RETRYABLE
            if kind is LifecycleOutcomeKind.RETRYABLE
            else ExecutionStage.MANUAL_REVIEW
        )
        return LifecycleOutcome(kind=kind, stage=stage, identity=plan.identity, reason=reason)

    def execute(self, plan: LifecyclePlan) -> LifecycleOutcome:
        if not isinstance(plan, LifecyclePlan):
            raise LifecycleContractError("terminal executor requires a validated plan")
        if plan.action is LifecycleAction.NOOP:
            return LifecycleOutcome(
                kind=LifecycleOutcomeKind.NOOP,
                stage=ExecutionStage.FINALIZED,
                identity=plan.identity,
            )
        if plan.action not in {LifecycleAction.DISABLE_CHAIN, LifecycleAction.FINALIZE_CHAIN}:
            return LifecycleOutcome(
                kind=LifecycleOutcomeKind.MANUAL_REVIEW,
                stage=ExecutionStage.MANUAL_REVIEW,
                identity=plan.identity,
                reason=f"unsupported terminal action: {plan.action.value}",
            )

        validated = self.services.validate_terminal(plan)
        if not validated.successful:
            return self._failure(plan, validated, "terminal parent guard failed")
        disabled = self.services.disable_chain(plan)
        if not disabled.successful:
            return self._failure(plan, disabled, "chain disablement failed")
        verified = self.services.verify_terminal(plan)
        if not verified.successful:
            return self._failure(plan, verified, "terminal chain verification failed")
        return LifecycleOutcome(
            kind=LifecycleOutcomeKind.APPLIED,
            stage=ExecutionStage.FINALIZED,
            identity=plan.identity,
        )


__all__ = (
    "LifecycleExecutorServices",
    "LifecycleTerminalExecutor",
    "LifecycleTerminalServices",
    "LifecycleTransitionExecutor",
    "OperationResult",
    "OperationState",
)
