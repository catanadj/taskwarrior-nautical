"""Immutable contracts for Nautical's Taskwarrior integration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Generic, TypeAlias, TypeVar


class IntegrationContractError(ValueError):
    """Raised when an integration model violates a boundary invariant."""


class CommandFailureKind(str, Enum):
    SUCCESS = "success"
    ABSENT = "absent"
    TIMEOUT = "timeout"
    BUSY = "busy"
    MISSING_BINARY = "missing_binary"
    INVALID_RESPONSE = "invalid_response"
    REJECTED = "rejected"
    EXECUTION_FAILURE = "execution_failure"


_RETRYABLE_FAILURES = frozenset(
    {
        CommandFailureKind.TIMEOUT,
        CommandFailureKind.BUSY,
        CommandFailureKind.EXECUTION_FAILURE,
    }
)


def _required_text(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise IntegrationContractError(f"{field} is required")
    return text


def _non_negative_float(value: object, field: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise IntegrationContractError(f"{field} must be a finite number") from exc
    if not math.isfinite(number) or number < 0.0 or (positive and number == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise IntegrationContractError(f"{field} must be a finite {qualifier} number")
    return number


@dataclass(frozen=True, slots=True)
class TaskCommand:
    """One bounded Taskwarrior invocation with an observable purpose."""

    argv: tuple[str, ...]
    purpose: str
    timeout: float
    input_text: str | None = None

    def __post_init__(self) -> None:
        argv = tuple(str(arg) for arg in self.argv)
        if not argv or not argv[0].strip():
            raise IntegrationContractError("task command requires an executable")
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "purpose", _required_text(self.purpose, "command purpose"))
        object.__setattr__(self, "timeout", _non_negative_float(self.timeout, "command timeout", positive=True))
        if self.input_text is not None and not isinstance(self.input_text, str):
            raise IntegrationContractError("command input_text must be text or None")


@dataclass(frozen=True, slots=True)
class TaskCommandResult:
    """Lossless evidence from one final Taskwarrior command attempt."""

    command: TaskCommand
    returncode: int
    stdout: str
    stderr: str
    kind: CommandFailureKind
    attempt: int
    duration: float

    def __post_init__(self) -> None:
        if not isinstance(self.command, TaskCommand):
            raise IntegrationContractError("command result requires a TaskCommand")
        if isinstance(self.returncode, bool) or not isinstance(self.returncode, int):
            raise IntegrationContractError("command returncode must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise IntegrationContractError("command output must be text")
        try:
            kind = CommandFailureKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise IntegrationContractError("invalid command failure kind") from exc
        if kind is CommandFailureKind.SUCCESS and self.returncode != 0:
            raise IntegrationContractError("successful command result requires returncode 0")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise IntegrationContractError("command attempt must be a positive integer")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "duration", _non_negative_float(self.duration, "command duration"))

    @property
    def ok(self) -> bool:
        return self.kind is CommandFailureKind.SUCCESS


@dataclass(frozen=True, slots=True)
class FailureEvidence:
    """Structured evidence explaining why an authoritative read is unavailable."""

    command: TaskCommand
    kind: CommandFailureKind
    returncode: int
    attempt: int
    duration: float
    retryable: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.command, TaskCommand):
            raise IntegrationContractError("failure evidence requires a TaskCommand")
        try:
            kind = CommandFailureKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise IntegrationContractError("invalid failure evidence kind") from exc
        if kind in {CommandFailureKind.SUCCESS, CommandFailureKind.ABSENT}:
            raise IntegrationContractError("failure evidence requires an unavailable failure kind")
        if isinstance(self.returncode, bool) or not isinstance(self.returncode, int):
            raise IntegrationContractError("failure returncode must be an integer")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise IntegrationContractError("failure attempt must be a positive integer")
        if not isinstance(self.retryable, bool):
            raise IntegrationContractError("failure retryable flag must be boolean")
        if self.retryable and kind not in _RETRYABLE_FAILURES:
            raise IntegrationContractError(f"{kind.value} failures cannot be marked retryable")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "duration", _non_negative_float(self.duration, "failure duration"))
        object.__setattr__(self, "detail", str(self.detail or "").strip())


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Found(Generic[T]):
    """An authoritative read that contains one non-null domain value."""

    value: T
    query: str

    def __post_init__(self) -> None:
        if self.value is None:
            raise IntegrationContractError("found read cannot contain None")
        object.__setattr__(self, "query", _required_text(self.query, "read query"))


@dataclass(frozen=True, slots=True)
class Absent:
    """An authoritative read proving that its requested value is absent."""

    query: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _required_text(self.query, "read query"))
        object.__setattr__(self, "reason", _required_text(self.reason, "absence reason"))


@dataclass(frozen=True, slots=True)
class Unavailable:
    """A read that cannot authoritatively prove either presence or absence."""

    query: str
    evidence: FailureEvidence

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _required_text(self.query, "read query"))
        if not isinstance(self.evidence, FailureEvidence):
            raise IntegrationContractError("unavailable read requires failure evidence")

    @property
    def retryable(self) -> bool:
        return self.evidence.retryable


TaskRead: TypeAlias = Found[T] | Absent | Unavailable


class GuardTimestampField(str, Enum):
    MODIFIED = "modified"
    DUE = "due"
    UNTIL = "until"
    END = "end"


@dataclass(frozen=True, slots=True)
class GuardTimestamp:
    field: GuardTimestampField
    value: str

    def __post_init__(self) -> None:
        try:
            field = GuardTimestampField(self.field)
        except (TypeError, ValueError) as exc:
            raise IntegrationContractError("invalid guard timestamp field") from exc
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "value", _required_text(self.value, f"{field.value} guard timestamp"))


@dataclass(frozen=True, slots=True)
class MutationGuard:
    """Authoritative task facts that must hold immediately before mutation."""

    task_uuid: str
    status: str
    chain_id: str
    link: int
    recurrence_identity: str
    timestamps: tuple[GuardTimestamp, ...]
    expected_mutation_epoch: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_uuid", _required_text(self.task_uuid, "guard task UUID"))
        object.__setattr__(self, "status", _required_text(self.status, "guard task status"))
        object.__setattr__(self, "chain_id", _required_text(self.chain_id, "guard chainID"))
        object.__setattr__(
            self,
            "recurrence_identity",
            _required_text(self.recurrence_identity, "guard recurrence identity"),
        )
        if isinstance(self.link, bool) or not isinstance(self.link, int) or self.link < 0:
            raise IntegrationContractError("guard link must be a non-negative integer")
        if (
            isinstance(self.expected_mutation_epoch, bool)
            or not isinstance(self.expected_mutation_epoch, int)
            or self.expected_mutation_epoch < 0
        ):
            raise IntegrationContractError("guard mutation epoch must be a non-negative integer")
        timestamps = tuple(self.timestamps)
        if not timestamps or any(not isinstance(item, GuardTimestamp) for item in timestamps):
            raise IntegrationContractError("guard requires typed timestamp evidence")
        fields = tuple(item.field for item in timestamps)
        if len(fields) != len(set(fields)):
            raise IntegrationContractError("guard timestamp fields must be unique")
        if GuardTimestampField.MODIFIED not in fields:
            raise IntegrationContractError("guard requires the task modified timestamp")
        object.__setattr__(self, "timestamps", timestamps)


class MutationOperation(str, Enum):
    CHILD_IMPORT = "child_import"
    PARENT_LINK = "parent_link"
    CHAIN_DISABLE = "chain_disable"
    NATIVE_UNTIL_REPAIR = "native_until_repair"
    METADATA_REPAIR = "metadata_repair"


class MutationPostcondition(str, Enum):
    CHILD_IMPORTED = "child_imported"
    PARENT_LINKED = "parent_linked"
    CHAIN_DISABLED = "chain_disabled"
    NATIVE_UNTIL_REPAIRED = "native_until_repaired"
    METADATA_REPAIRED = "metadata_repaired"


_OPERATION_POSTCONDITION = {
    MutationOperation.CHILD_IMPORT: MutationPostcondition.CHILD_IMPORTED,
    MutationOperation.PARENT_LINK: MutationPostcondition.PARENT_LINKED,
    MutationOperation.CHAIN_DISABLE: MutationPostcondition.CHAIN_DISABLED,
    MutationOperation.NATIVE_UNTIL_REPAIR: MutationPostcondition.NATIVE_UNTIL_REPAIRED,
    MutationOperation.METADATA_REPAIR: MutationPostcondition.METADATA_REPAIRED,
}


class MutationOutcomeKind(str, Enum):
    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"
    RETRYABLE = "retryable"
    REJECTED = "rejected"
    CONFLICT = "conflict"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True, slots=True)
class MutationOutcome:
    """Tagged result of one guarded, named Taskwarrior mutation."""

    operation: MutationOperation
    kind: MutationOutcomeKind
    guard: MutationGuard
    postconditions: tuple[MutationPostcondition, ...] = ()
    reason: str = ""
    failure: FailureEvidence | None = None

    def __post_init__(self) -> None:
        try:
            operation = MutationOperation(self.operation)
            kind = MutationOutcomeKind(self.kind)
            postconditions = tuple(MutationPostcondition(item) for item in self.postconditions)
        except (TypeError, ValueError) as exc:
            raise IntegrationContractError("invalid mutation operation, outcome, or postcondition") from exc
        if not isinstance(self.guard, MutationGuard):
            raise IntegrationContractError("mutation outcome requires a MutationGuard")
        if len(postconditions) != len(set(postconditions)):
            raise IntegrationContractError("mutation postconditions must be unique")
        expected = _OPERATION_POSTCONDITION[operation]
        succeeded = kind in {MutationOutcomeKind.APPLIED, MutationOutcomeKind.ALREADY_APPLIED}
        if succeeded and postconditions != (expected,):
            raise IntegrationContractError(
                f"{kind.value} {operation.value} outcome requires {expected.value} postcondition"
            )
        if not succeeded and expected in postconditions:
            raise IntegrationContractError("unsuccessful mutation cannot claim its expected postcondition")
        reason = str(self.reason or "").strip()
        if not succeeded and not reason:
            raise IntegrationContractError(f"{kind.value} mutation outcome requires a reason")
        if succeeded and self.failure is not None:
            raise IntegrationContractError("successful mutation outcome cannot carry failure evidence")
        if kind is MutationOutcomeKind.RETRYABLE:
            if self.failure is None or not self.failure.retryable:
                raise IntegrationContractError("retryable mutation outcome requires retryable failure evidence")
        elif self.failure is not None and self.failure.retryable:
            raise IntegrationContractError("retryable failure evidence requires a retryable mutation outcome")
        if self.failure is not None and not isinstance(self.failure, FailureEvidence):
            raise IntegrationContractError("mutation failure must be structured evidence")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "postconditions", postconditions)
        object.__setattr__(self, "reason", reason)


__all__ = (
    "Absent",
    "CommandFailureKind",
    "FailureEvidence",
    "Found",
    "GuardTimestamp",
    "GuardTimestampField",
    "IntegrationContractError",
    "MutationGuard",
    "MutationOperation",
    "MutationOutcome",
    "MutationOutcomeKind",
    "MutationPostcondition",
    "TaskCommand",
    "TaskCommandResult",
    "TaskRead",
    "Unavailable",
)
