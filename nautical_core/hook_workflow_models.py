"""Typed contract vocabulary for the next-generation hook workflow engine.

This module is deliberately dependency-light.  It defines only closed
classification and result vocabularies; current hook composition roots are
migrated to these types in later passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

from .lifecycle_models import LifecyclePlan, TaskLifecycleState
from .task_models import FrozenValue, TaskObservation, TaskTimestamp


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


class FeedbackFactKind(str, Enum):
    """Stable presentation fact categories; wording remains a renderer concern."""

    PREVIEW = "preview"
    EXPLICIT_TIMING = "explicit_timing"
    CARRY_CHANGE = "carry_change"
    CHAIN_ACTIVATION = "chain_activation"
    UPDATE = "update"
    COMPLETION = "completion"
    RESUME = "resume"
    TERMINAL_STOP = "terminal_stop"
    WARNING = "warning"
    RECOVERY = "recovery"
    MANUAL_REVIEW = "manual_review"


class PatchOperation(str, Enum):
    SET = "set"
    CLEAR = "clear"
    PRESERVE = "preserve"


class OutcomeDisposition(str, Enum):
    """How the hook boundary must complete an accepted workflow result."""

    EMIT_TASK = "emit_task"
    REJECT_OPERATION = "reject_operation"
    DEFER_RECOVERY = "defer_recovery"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class HookOutputContract:
    """Machine-facing stdout/stderr contract for one hook kind."""

    hook_kind: HookKind
    stdout: str
    ensure_ascii: bool = False
    diagnostics_stderr_only_when_enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "hook_kind", HookKind(self.hook_kind))
        if self.stdout not in {"task_json", "empty"}:
            raise ValueError("hook stdout contract must be task_json or empty")
        if not self.ensure_ascii:
            return
        raise ValueError("Nautical hook JSON must preserve Unicode with ensure_ascii=False")


HOOK_OUTPUT_CONTRACTS: dict[HookKind, HookOutputContract] = {
    HookKind.ADD: HookOutputContract(HookKind.ADD, "task_json"),
    HookKind.MODIFY: HookOutputContract(HookKind.MODIFY, "task_json"),
    HookKind.EXIT: HookOutputContract(HookKind.EXIT, "empty"),
}


ROUTE_PRECEDENCE: tuple[WorkflowRoute, ...] = (
    WorkflowRoute.DELETION,
    WorkflowRoute.TERMINAL_STOP,
    WorkflowRoute.MANUAL_CHAIN_OFF,
    WorkflowRoute.RECURRENCE_REMOVAL,
    WorkflowRoute.RESUME,
    WorkflowRoute.COMPLETION,
    WorkflowRoute.RECURRING_EDIT,
    WorkflowRoute.CP_ACTIVATION,
    WorkflowRoute.ANCHOR_FILE_ACTIVATION,
    WorkflowRoute.ANCHOR_ACTIVATION,
    WorkflowRoute.ORDINARY,
)


OUTCOME_DISPOSITION_RULES: dict[WorkflowOutcomeKind, OutcomeDisposition] = {
    WorkflowOutcomeKind.PASSTHROUGH: OutcomeDisposition.EMIT_TASK,
    WorkflowOutcomeKind.ACCEPTED_PATCH: OutcomeDisposition.EMIT_TASK,
    WorkflowOutcomeKind.LIFECYCLE_APPLICATION: OutcomeDisposition.EMIT_TASK,
    WorkflowOutcomeKind.TERMINAL_TRANSITION: OutcomeDisposition.TERMINAL,
    WorkflowOutcomeKind.REJECTED_INPUT: OutcomeDisposition.REJECT_OPERATION,
    WorkflowOutcomeKind.RETRYABLE_UNAVAILABLE: OutcomeDisposition.DEFER_RECOVERY,
    WorkflowOutcomeKind.INTERNAL_FAILURE: OutcomeDisposition.REJECT_OPERATION,
}


@dataclass(frozen=True, slots=True)
class WorkflowOutcome:
    """Structured result at the hook boundary, before rendering."""

    kind: WorkflowOutcomeKind
    disposition: OutcomeDisposition
    route: WorkflowRoute
    exit_code: int = 0
    preserve_input: bool = False
    failure: WorkflowFailureCategory | None = None
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", WorkflowOutcomeKind(self.kind))
        object.__setattr__(self, "disposition", OutcomeDisposition(self.disposition))
        object.__setattr__(self, "route", WorkflowRoute(self.route))
        if OUTCOME_DISPOSITION_RULES[self.kind] is not self.disposition:
            raise ValueError(
                f"outcome {self.kind.value} requires disposition "
                f"{OUTCOME_DISPOSITION_RULES[self.kind].value}"
            )
        if self.exit_code not in (0, 1):
            raise ValueError("workflow exit_code must be 0 or 1")
        failure = self.failure
        if failure is not None:
            object.__setattr__(self, "failure", WorkflowFailureCategory(failure))
        if self.kind in {WorkflowOutcomeKind.REJECTED_INPUT, WorkflowOutcomeKind.RETRYABLE_UNAVAILABLE, WorkflowOutcomeKind.INTERNAL_FAILURE} and failure is None:
            raise ValueError("failed workflow outcomes require a failure category")
        if self.kind in {WorkflowOutcomeKind.PASSTHROUGH, WorkflowOutcomeKind.ACCEPTED_PATCH, WorkflowOutcomeKind.LIFECYCLE_APPLICATION, WorkflowOutcomeKind.TERMINAL_TRANSITION} and failure is not None:
            raise ValueError("successful workflow outcomes cannot carry a failure category")
        if self.disposition is OutcomeDisposition.REJECT_OPERATION and self.exit_code == 0:
            raise ValueError("rejected workflow outcomes require a non-zero exit code")
        if self.disposition is OutcomeDisposition.DEFER_RECOVERY and not self.preserve_input:
            raise ValueError("deferred recovery must preserve the incoming task")
        object.__setattr__(self, "message", str(self.message).strip())


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
    value: FrozenValue = None
    expected_current: FrozenValue = None

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


@dataclass(frozen=True, slots=True)
class LifecycleEffectRef:
    """Typed reference to one lifecycle plan owned by lifecycle application."""

    plan: LifecyclePlan

    def __post_init__(self) -> None:
        if not isinstance(self.plan, LifecyclePlan):
            raise TypeError("lifecycle effect requires a LifecyclePlan")


@dataclass(frozen=True, slots=True)
class TaskPatchEffect:
    """A task patch returned to the single workflow application boundary."""

    patch: TaskPatch

    def __post_init__(self) -> None:
        if not isinstance(self.patch, TaskPatch):
            raise TypeError("task patch effect requires a TaskPatch")


@dataclass(frozen=True, slots=True)
class TerminalStateEffect:
    """A terminal lifecycle state to record after guarded application."""

    task: TaskObservation
    state: TaskLifecycleState
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.task, TaskObservation):
            raise TypeError("terminal state effect requires a TaskObservation")
        object.__setattr__(self, "state", TaskLifecycleState(self.state))
        object.__setattr__(self, "reason", str(self.reason or "").strip())


WorkflowEffect: TypeAlias = TaskPatchEffect | LifecycleEffectRef | TerminalStateEffect


@dataclass(frozen=True, slots=True)
class FeedbackFacts:
    """Presentation-neutral facts shared by panels, diagnostics, and tools."""

    recurrence_kind: str = ""
    natural_explanation: str = ""
    first_occurrence: TaskTimestamp | None = None
    next_occurrence: TaskTimestamp | None = None
    carry_changes: tuple[tuple[str, str], ...] = ()
    limits: tuple[tuple[str, str], ...] = ()
    chain_completed: bool = False
    warnings: tuple[str, ...] = ()
    recovery_guidance: tuple[str, ...] = ()
    fact_kinds: tuple[FeedbackFactKind, ...] = ()

    def __post_init__(self) -> None:
        for name in ("first_occurrence", "next_occurrence"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, TaskTimestamp):
                raise TypeError(f"{name} must be a TaskTimestamp or None")
        object.__setattr__(self, "recurrence_kind", str(self.recurrence_kind or "").strip())
        object.__setattr__(self, "natural_explanation", str(self.natural_explanation or "").strip())
        object.__setattr__(self, "carry_changes", _normalized_pairs(self.carry_changes, "carry_changes"))
        object.__setattr__(self, "limits", _normalized_pairs(self.limits, "limits"))
        object.__setattr__(self, "warnings", _normalized_texts(self.warnings))
        object.__setattr__(self, "recovery_guidance", _normalized_texts(self.recovery_guidance))
        kinds = tuple(FeedbackFactKind(item) for item in self.fact_kinds)
        if len(kinds) != len(set(kinds)):
            raise ValueError("feedback fact kinds must be unique")
        object.__setattr__(self, "fact_kinds", kinds)
        object.__setattr__(self, "chain_completed", bool(self.chain_completed))


def _normalized_texts(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(text for text in (str(value).strip() for value in values) if text)


def _normalized_pairs(values: tuple[tuple[str, str], ...], name: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for pair in values:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"{name} entries must be (name, value) tuples")
        key, value = (str(item).strip() for item in pair)
        if not key:
            raise ValueError(f"{name} keys cannot be empty")
        pairs.append((key, value))
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class WorkflowOperationalResult:
    """Final typed workflow result without rendered strings or JSON mappings."""

    task: TaskObservation
    outcome: WorkflowOutcome
    effects: tuple[WorkflowEffect, ...] = ()
    feedback: FeedbackFacts = FeedbackFacts()

    def __post_init__(self) -> None:
        if not isinstance(self.task, TaskObservation):
            raise TypeError("workflow result requires a final TaskObservation")
        if not isinstance(self.outcome, WorkflowOutcome):
            raise TypeError("workflow result requires a WorkflowOutcome")
        effects = tuple(self.effects)
        if any(not isinstance(effect, (TaskPatchEffect, LifecycleEffectRef, TerminalStateEffect)) for effect in effects):
            raise TypeError("workflow effects must be TaskPatchEffect, LifecycleEffectRef, or TerminalStateEffect values")
        if not isinstance(self.feedback, FeedbackFacts):
            raise TypeError("workflow result feedback must be FeedbackFacts")
        object.__setattr__(self, "effects", effects)


__all__ = [
    "HookKind",
    "HookOutputContract",
    "HOOK_OUTPUT_CONTRACTS",
    "OUTCOME_DISPOSITION_RULES",
    "AddWorkflowRequest",
    "EvidenceKind",
    "EvidenceRequest",
    "EvidenceResult",
    "EvidenceStatus",
    "FeedbackFacts",
    "FeedbackFactKind",
    "LifecycleEffectRef",
    "ModifyWorkflowRequest",
    "OutcomeDisposition",
    "PatchOperation",
    "ROUTE_PRECEDENCE",
    "TaskPatch",
    "TaskPatchOperation",
    "TaskPatchEffect",
    "TerminalStateEffect",
    "WorkflowEffect",
    "WorkflowOperationalResult",
    "WorkflowOutcome",
    "WorkflowFailureCategory",
    "WorkflowOutcomeKind",
    "WorkflowRoute",
]
