"""Typed contract vocabulary for the next-generation hook workflow engine.

This module is deliberately dependency-light.  It defines only closed
classification and result vocabularies; current hook composition roots are
migrated to these types in later passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .task_models import TaskObservation


class HookKind(str, Enum):
    """Taskwarrior hook entry points owned by the workflow engine."""

    ADD = "add"
    MODIFY = "modify"
    EXIT = "exit"


class WorkflowRoute(str, Enum):
    """Explicit route families; no route may fall through implicitly."""

    ORDINARY = "ordinary"
    CP_ACTIVATION = "cp_activation"
    ANCHOR_ACTIVATION = "anchor_activation"
    ANCHOR_FILE_ACTIVATION = "anchor_file_activation"
    RECURRING_EDIT = "recurring_edit"
    COMPLETION = "completion"
    DELETION = "deletion"
    CHAIN_DISABLE = "chain_disable"
    MANUAL_CHAIN_OFF = "manual_chain_off"
    RECURRENCE_REMOVAL = "recurrence_removal"
    RESUME = "resume"
    TERMINAL_STOP = "terminal_stop"
    EXIT_DRAIN = "exit_drain"


class WorkflowOutcomeKind(str, Enum):
    """Closed outcome set shared by add, modify, and exit workflows."""

    PASSTHROUGH = "passthrough"
    ACCEPTED_PATCH = "accepted_patch"
    LIFECYCLE_APPLICATION = "lifecycle_application"
    TERMINAL_TRANSITION = "terminal_transition"
    REJECTED_INPUT = "rejected_input"
    RETRYABLE_UNAVAILABLE = "retryable_unavailable"
    INTERNAL_FAILURE = "internal_failure"


class WorkflowFailureCategory(str, Enum):
    """Operational failure categories exposed by the workflow contract."""

    INVALID_INPUT = "invalid_input"
    INVALID_CONFIGURATION = "invalid_configuration"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"
    SCHEDULER_EXHAUSTED = "scheduler_exhausted"
    LIFECYCLE_CONFLICT = "lifecycle_conflict"
    MANUAL_REVIEW = "manual_review"
    PROGRAMMING_ERROR = "programming_error"


class EvidenceKind(str, Enum):
    """Authoritative evidence requested by a workflow route."""

    TASK_UUIDS = "task_uuids"
    CHAIN_SLOTS = "chain_slots"
    CHAIN_HISTORY = "chain_history"
    CONFIGURATION = "configuration"
    RECURRENCE = "recurrence"


class EvidenceStatus(str, Enum):
    """Evidence state; unavailable is never equivalent to absent."""

    FOUND = "found"
    ABSENT = "absent"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    MALFORMED = "malformed"
    AMBIGUOUS = "ambiguous"


class PatchOperation(str, Enum):
    SET = "set"
    CLEAR = "clear"
    PRESERVE = "preserve"


@dataclass(frozen=True, slots=True)
class AddWorkflowRequest:
    """Validated add input before planning or mutation."""

    task: TaskObservation
    route: WorkflowRoute

    def __post_init__(self) -> None:
        if not isinstance(self.task, TaskObservation):
            raise TypeError("add workflow request requires a TaskObservation")
        allowed = {
            WorkflowRoute.ORDINARY,
            WorkflowRoute.CP_ACTIVATION,
            WorkflowRoute.ANCHOR_ACTIVATION,
            WorkflowRoute.ANCHOR_FILE_ACTIVATION,
        }
        route = WorkflowRoute(self.route)
        if route not in allowed:
            raise ValueError(f"invalid add workflow route: {route.value}")
        object.__setattr__(self, "route", route)


@dataclass(frozen=True, slots=True)
class ModifyWorkflowRequest:
    """Validated old/new task observations and one explicit modify route."""

    old: TaskObservation
    new: TaskObservation
    route: WorkflowRoute

    def __post_init__(self) -> None:
        if not isinstance(self.old, TaskObservation) or not isinstance(self.new, TaskObservation):
            raise TypeError("modify workflow request requires TaskObservation values")
        route = WorkflowRoute(self.route)
        if route in {
            WorkflowRoute.CP_ACTIVATION,
            WorkflowRoute.ANCHOR_ACTIVATION,
            WorkflowRoute.ANCHOR_FILE_ACTIVATION,
            WorkflowRoute.EXIT_DRAIN,
        }:
            raise ValueError(f"invalid modify workflow route: {route.value}")
        object.__setattr__(self, "route", route)


@dataclass(frozen=True, slots=True)
class EvidenceRequest:
    """Bounded request for authoritative Taskwarrior/configuration evidence."""

    kind: EvidenceKind
    uuids: tuple[str, ...] = ()
    chain_id: str | None = None
    links: tuple[int, ...] = ()
    max_rows: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EvidenceKind(self.kind))
        uuids = tuple(str(value).strip() for value in self.uuids if str(value).strip())
        links = tuple(int(value) for value in self.links)
        if any(value <= 0 for value in links):
            raise ValueError("evidence links must be positive")
        if self.max_rows <= 0:
            raise ValueError("evidence max_rows must be positive")
        if not uuids and not str(self.chain_id or "").strip() and not links and self.kind is not EvidenceKind.CONFIGURATION:
            raise ValueError("bounded evidence request requires an identity scope")
        object.__setattr__(self, "uuids", uuids)
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "chain_id", str(self.chain_id).strip() or None)


@dataclass(frozen=True, slots=True)
class EvidenceResult:
    """Typed evidence response preserving unavailable and malformed states."""

    status: EvidenceStatus
    rows: tuple[TaskObservation, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", EvidenceStatus(self.status))
        rows = tuple(self.rows)
        if any(not isinstance(row, TaskObservation) for row in rows):
            raise TypeError("evidence rows must be TaskObservation values")
        if self.status is EvidenceStatus.FOUND and not rows:
            raise ValueError("found evidence requires at least one row")
        if self.status in {EvidenceStatus.UNAVAILABLE, EvidenceStatus.MALFORMED} and not str(self.reason).strip():
            raise ValueError("unavailable or malformed evidence requires a reason")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "reason", str(self.reason).strip())


@dataclass(frozen=True, slots=True)
class TaskPatchOperation:
    """One explicit set/clear/preserve operation with optional CAS evidence."""

    field: str
    operation: PatchOperation
    value: Any = None
    expected_current: Any = None

    def __post_init__(self) -> None:
        field = str(self.field).strip()
        if not field:
            raise ValueError("patch field is required")
        operation = PatchOperation(self.operation)
        if operation is PatchOperation.SET and self.value is None:
            raise ValueError("set operation requires a value")
        if operation is not PatchOperation.SET and self.value is not None:
            raise ValueError(f"{operation.value} operation cannot carry a value")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "operation", operation)


@dataclass(frozen=True, slots=True)
class TaskPatch:
    """Immutable, non-contradictory collection of task field operations."""

    operations: tuple[TaskPatchOperation, ...]

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if any(not isinstance(item, TaskPatchOperation) for item in operations):
            raise TypeError("task patch operations must be TaskPatchOperation values")
        fields = [item.field for item in operations if item.operation is not PatchOperation.PRESERVE]
        if len(fields) != len(set(fields)):
            raise ValueError("task patch contains contradictory operations for one field")
        object.__setattr__(self, "operations", operations)


__all__ = [
    "HookKind",
    "AddWorkflowRequest",
    "EvidenceKind",
    "EvidenceRequest",
    "EvidenceResult",
    "EvidenceStatus",
    "ModifyWorkflowRequest",
    "PatchOperation",
    "TaskPatch",
    "TaskPatchOperation",
    "WorkflowFailureCategory",
    "WorkflowOutcomeKind",
    "WorkflowRoute",
]
