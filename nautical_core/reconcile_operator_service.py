"""Typed reconcile recovery coordination boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .chain_generation import ChainGenerationService
from .chain_integrity_lifecycle import is_orphan_deleted_chain_candidate
from .lifecycle.reconciliation import (
    ApplyParentCallback,
    CallbackLifecycleRecoveryOperations,
    LifecycleReconciliationService,
    PlanParentCallback,
    VirtualChildCallback,
)
from .lifecycle.recovery_models import RecoveryResult
from .task_models import TaskObservation, TaskPayload


@dataclass(frozen=True, slots=True)
class ReconcileRecoveryCallbacks:
    """Taskwarrior-specific mechanics supplied to the lifecycle owner."""

    apply_parent: ApplyParentCallback
    plan_parent: PlanParentCallback
    next_child: Callable[[TaskObservation, str], TaskObservation]
    virtual_child: VirtualChildCallback
    terminal_error: Callable[[TaskObservation, Any], str]
    recovery_error: Callable[[TaskPayload, str], RecoveryResult]
    recovery_partial: Callable[[TaskPayload, str], RecoveryResult]
    recovery_manual_review: Callable[[TaskPayload, str], RecoveryResult]
    recovery_terminal: Callable[[TaskPayload, str], RecoveryResult]
    recovery_exception: Callable[[TaskPayload, Exception], RecoveryResult]


@dataclass(frozen=True, slots=True)
class ReconcileRecoveryCoordinator:
    """Submit one typed parent to the shared lifecycle recovery loop."""

    service: LifecycleReconciliationService
    callbacks: ReconcileRecoveryCallbacks

    def recover(
        self,
        parent: TaskPayload,
        *,
        taskdata: Path | None,
        apply: bool,
        max_expiration_hops: int,
        recovery_at: Any,
        lease_held: bool = False,
        generation: ChainGenerationService | None = None,
    ) -> list[tuple[RecoveryResult, str]]:
        callbacks = self.callbacks
        operations = CallbackLifecycleRecoveryOperations(
            apply_parent_callback=callbacks.apply_parent,
            plan_parent_callback=callbacks.plan_parent,
            next_child_callback=callbacks.next_child,
            virtual_child_callback=callbacks.virtual_child,
            terminal_error_callback=callbacks.terminal_error,
            is_orphan_deleted_callback=is_orphan_deleted_chain_candidate,
            recovery_error_callback=callbacks.recovery_error,
            recovery_partial_callback=callbacks.recovery_partial,
            recovery_manual_review_callback=callbacks.recovery_manual_review,
            recovery_terminal_callback=callbacks.recovery_terminal,
            recovery_exception_callback=callbacks.recovery_exception,
        )
        return self.service.recover_candidate(
            parent,
            operations=operations,
            taskdata=taskdata,
            apply=apply,
            max_expiration_hops=max_expiration_hops,
            recovery_at=recovery_at,
            lease_held=lease_held,
            generation=generation,
        )


__all__ = ["ReconcileRecoveryCallbacks", "ReconcileRecoveryCoordinator"]
