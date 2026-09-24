"""Pure lifecycle execution classifications and progress accounting."""

from __future__ import annotations

from enum import Enum

from .integration_models import MutationOutcomeKind
from .lifecycle_models import ExecutionStage
from .lifecycle_outbox import OutboxResultKind


class LifecycleApplicationOutcomeKind(str, Enum):
    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"
    RETRYABLE = "retryable"
    CONFLICT = "conflict"
    MANUAL_REVIEW = "manual_review"
    QUARANTINED = "quarantined"
    NOOP = "noop"


class FailureDisposition(str, Enum):
    RETRY = "retry"
    MANUAL_REVIEW = "manual_review"


SPAWN_STAGE_ORDER = {
    ExecutionStage.PLANNED: 0,
    ExecutionStage.PERSISTED: 0,
    ExecutionStage.CHILD_PRESENT: 1,
    ExecutionStage.PARENT_LINKED: 2,
    ExecutionStage.VERIFIED: 3,
    ExecutionStage.FINALIZED: 4,
}


OUTBOX_TO_APPLICATION = {
    OutboxResultKind.APPLIED: LifecycleApplicationOutcomeKind.APPLIED,
    OutboxResultKind.ALREADY_APPLIED: LifecycleApplicationOutcomeKind.ALREADY_APPLIED,
    OutboxResultKind.RETRYABLE: LifecycleApplicationOutcomeKind.RETRYABLE,
    OutboxResultKind.CONFLICT: LifecycleApplicationOutcomeKind.CONFLICT,
    OutboxResultKind.REJECTED: LifecycleApplicationOutcomeKind.MANUAL_REVIEW,
}


MUTATION_TO_APPLICATION = {
    MutationOutcomeKind.APPLIED: LifecycleApplicationOutcomeKind.APPLIED,
    MutationOutcomeKind.ALREADY_APPLIED: LifecycleApplicationOutcomeKind.ALREADY_APPLIED,
    MutationOutcomeKind.RETRYABLE: LifecycleApplicationOutcomeKind.RETRYABLE,
    MutationOutcomeKind.CONFLICT: LifecycleApplicationOutcomeKind.CONFLICT,
    MutationOutcomeKind.REJECTED: LifecycleApplicationOutcomeKind.MANUAL_REVIEW,
    MutationOutcomeKind.MANUAL_REVIEW: LifecycleApplicationOutcomeKind.MANUAL_REVIEW,
}


def classify_outbox_failure(kind: OutboxResultKind) -> FailureDisposition:
    return (
        FailureDisposition.RETRY
        if kind is OutboxResultKind.RETRYABLE
        else FailureDisposition.MANUAL_REVIEW
    )


def classify_mutation_failure(kind: MutationOutcomeKind) -> FailureDisposition:
    return (
        FailureDisposition.RETRY
        if kind is MutationOutcomeKind.RETRYABLE
        else FailureDisposition.MANUAL_REVIEW
    )

def remaining_drain_work(stage: ExecutionStage) -> int:
    stage_order = SPAWN_STAGE_ORDER[stage]
    if stage_order < SPAWN_STAGE_ORDER[ExecutionStage.CHILD_PRESENT]:
        return 6
    if stage_order < SPAWN_STAGE_ORDER[ExecutionStage.PARENT_LINKED]:
        return 4
    if stage_order < SPAWN_STAGE_ORDER[ExecutionStage.VERIFIED]:
        return 2
    return 1
