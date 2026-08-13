"""Immutable contracts for Nautical's Taskwarrior integration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Generic, TypeAlias, TypeVar

from .lifecycle_models import LifecycleIdentity


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


class OutboxStage(str, Enum):
    PERSISTED = "persisted"
    CLAIMED = "claimed"
    APPLYING = "applying"
    VERIFYING = "verifying"
    FINALIZED = "finalized"
    RETRYABLE = "retryable"
    MANUAL_REVIEW = "manual_review"


class OutboxOutcomeKind(str, Enum):
    ADVANCED = "advanced"
    FINALIZED = "finalized"
    RETRYABLE = "retryable"
    MANUAL_REVIEW = "manual_review"


_VALID_INTENT_OPERATIONS = frozenset(
    {
        (MutationOperation.CHILD_IMPORT, MutationOperation.PARENT_LINK),
        (MutationOperation.CHAIN_DISABLE,),
        (MutationOperation.NATIVE_UNTIL_REPAIR,),
        (MutationOperation.METADATA_REPAIR,),
    }
)


@dataclass(frozen=True, slots=True)
class OutboxIntent:
    """Durable mutation work for one deterministic lifecycle transition."""

    identity: LifecycleIdentity
    guard: MutationGuard
    operations: tuple[MutationOperation, ...]
    expected_postconditions: tuple[MutationPostcondition, ...]
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.identity, LifecycleIdentity):
            raise IntegrationContractError("outbox intent requires a LifecycleIdentity")
        if not isinstance(self.guard, MutationGuard):
            raise IntegrationContractError("outbox intent requires a MutationGuard")
        try:
            operations = tuple(MutationOperation(item) for item in self.operations)
            postconditions = tuple(MutationPostcondition(item) for item in self.expected_postconditions)
        except (TypeError, ValueError) as exc:
            raise IntegrationContractError("invalid outbox operation or postcondition") from exc
        if operations not in _VALID_INTENT_OPERATIONS:
            raise IntegrationContractError("outbox intent has an unsupported mutation sequence")
        expected = tuple(_OPERATION_POSTCONDITION[operation] for operation in operations)
        if postconditions != expected:
            raise IntegrationContractError("outbox postconditions must exactly match its mutation sequence")
        if self.identity.parent_uuid != self.guard.task_uuid:
            raise IntegrationContractError("outbox identity and guard UUID differ")
        if self.identity.chain_id != self.guard.chain_id:
            raise IntegrationContractError("outbox identity and guard chainID differ")
        if self.identity.source_link != self.guard.link:
            raise IntegrationContractError("outbox identity and guard link differ")
        if MutationOperation.CHILD_IMPORT in operations and self.identity.target_link is None:
            raise IntegrationContractError("child import intent requires a target lifecycle link")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise IntegrationContractError("outbox max_attempts must be a positive integer")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "expected_postconditions", postconditions)

    @property
    def intent_id(self) -> str:
        digest = hashlib.sha256(self.identity.key.encode("utf-8")).hexdigest()[:24]
        return f"ob1-{digest}"


@dataclass(frozen=True, slots=True)
class OutboxOutcome:
    """Validated durable progress for one outbox intent."""

    intent: OutboxIntent
    stage: OutboxStage
    kind: OutboxOutcomeKind
    mutations: tuple[MutationOutcome, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.intent, OutboxIntent):
            raise IntegrationContractError("outbox outcome requires an OutboxIntent")
        try:
            stage = OutboxStage(self.stage)
            kind = OutboxOutcomeKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise IntegrationContractError("invalid outbox stage or outcome kind") from exc
        mutations = tuple(self.mutations)
        if any(not isinstance(item, MutationOutcome) for item in mutations):
            raise IntegrationContractError("outbox outcome requires typed mutation outcomes")
        operations = tuple(item.operation for item in mutations)
        if len(operations) != len(set(operations)):
            raise IntegrationContractError("outbox outcome contains duplicate mutation operations")
        if any(item.operation not in self.intent.operations or item.guard != self.intent.guard for item in mutations):
            raise IntegrationContractError("outbox mutation outcome does not belong to its intent")

        required_stage = {
            OutboxOutcomeKind.FINALIZED: OutboxStage.FINALIZED,
            OutboxOutcomeKind.RETRYABLE: OutboxStage.RETRYABLE,
            OutboxOutcomeKind.MANUAL_REVIEW: OutboxStage.MANUAL_REVIEW,
        }.get(kind)
        if required_stage is not None and stage is not required_stage:
            raise IntegrationContractError(f"{kind.value} outbox outcome requires {required_stage.value} stage")
        if kind is OutboxOutcomeKind.ADVANCED and stage not in {
            OutboxStage.CLAIMED,
            OutboxStage.APPLYING,
            OutboxStage.VERIFYING,
        }:
            raise IntegrationContractError("advanced outbox outcome requires an active processing stage")

        success_kinds = {MutationOutcomeKind.APPLIED, MutationOutcomeKind.ALREADY_APPLIED}
        if kind is OutboxOutcomeKind.FINALIZED:
            if operations != self.intent.operations or any(item.kind not in success_kinds for item in mutations):
                raise IntegrationContractError("finalized outbox outcome requires every mutation to succeed")
        elif kind is OutboxOutcomeKind.RETRYABLE:
            if not any(item.kind is MutationOutcomeKind.RETRYABLE for item in mutations):
                raise IntegrationContractError("retryable outbox outcome requires a retryable mutation")
        elif kind is OutboxOutcomeKind.MANUAL_REVIEW:
            review_kinds = {
                MutationOutcomeKind.REJECTED,
                MutationOutcomeKind.CONFLICT,
                MutationOutcomeKind.MANUAL_REVIEW,
            }
            if not any(item.kind in review_kinds for item in mutations):
                raise IntegrationContractError("manual-review outbox outcome requires review evidence")

        reason = str(self.reason or "").strip()
        if kind in {OutboxOutcomeKind.RETRYABLE, OutboxOutcomeKind.MANUAL_REVIEW} and not reason:
            raise IntegrationContractError(f"{kind.value} outbox outcome requires a reason")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "mutations", mutations)
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
    "OutboxIntent",
    "OutboxOutcome",
    "OutboxOutcomeKind",
    "OutboxStage",
    "TaskCommand",
    "TaskCommandResult",
    "TaskRead",
    "Unavailable",
)
