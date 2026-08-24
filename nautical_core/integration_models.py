"""Immutable contracts for Nautical's Taskwarrior integration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Generic, Mapping, Protocol, TypeAlias, TypeVar

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


FrozenValue: TypeAlias = object
FrozenPairs: TypeAlias = tuple[tuple[str, FrozenValue], ...]


def _freeze_value(value: object) -> FrozenValue:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))
    return value


def _freeze_pairs(value: Mapping[str, object]) -> FrozenPairs:
    return tuple(sorted((str(key), _freeze_value(item)) for key, item in value.items()))


def _thaw(value: FrozenValue) -> object:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw(item) for key, item in value}
        return [_thaw(item) for item in value]
    return value


def _non_negative_float(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise IntegrationContractError(f"{field} must be a finite number")
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
    chain: str = "on"

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_uuid", _required_text(self.task_uuid, "guard task UUID"))
        object.__setattr__(self, "status", _required_text(self.status, "guard task status"))
        object.__setattr__(self, "chain_id", _required_text(self.chain_id, "guard chainID"))
        chain = _required_text(self.chain, "guard chain state").lower()
        if chain not in {"on", "off"}:
            raise IntegrationContractError("guard chain state must be on or off")
        object.__setattr__(self, "chain", chain)
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
        try:
            timestamps = tuple(self.timestamps)
        except TypeError as exc:
            raise IntegrationContractError("guard requires typed timestamp evidence") from exc
        if not timestamps or any(not isinstance(item, GuardTimestamp) for item in timestamps):
            raise IntegrationContractError("guard requires typed timestamp evidence")
        fields = tuple(item.field for item in timestamps)
        if len(fields) != len(set(fields)):
            raise IntegrationContractError("guard timestamp fields must be unique")
        if not {GuardTimestampField.MODIFIED, GuardTimestampField.END}.intersection(fields):
            raise IntegrationContractError("guard requires the task modified or end timestamp")
        object.__setattr__(self, "timestamps", timestamps)


class MutationOperation(str, Enum):
    CHILD_IMPORT = "child_import"
    CHILD_COMPENSATION = "child_compensation"
    PARENT_LINK = "parent_link"
    PARENT_LINK_CLEAR = "parent_link_clear"
    CHAIN_DISABLE = "chain_disable"
    NATIVE_UNTIL_REPAIR = "native_until_repair"
    METADATA_REPAIR = "metadata_repair"


class MutationPostcondition(str, Enum):
    CHILD_IMPORTED = "child_imported"
    CHILD_COMPENSATED = "child_compensated"
    PARENT_LINKED = "parent_linked"
    PARENT_LINK_CLEARED = "parent_link_cleared"
    CHAIN_DISABLED = "chain_disabled"
    NATIVE_UNTIL_REPAIRED = "native_until_repaired"
    METADATA_REPAIRED = "metadata_repaired"


_OPERATION_POSTCONDITION = {
    MutationOperation.CHILD_IMPORT: MutationPostcondition.CHILD_IMPORTED,
    MutationOperation.CHILD_COMPENSATION: MutationPostcondition.CHILD_COMPENSATED,
    MutationOperation.PARENT_LINK: MutationPostcondition.PARENT_LINKED,
    MutationOperation.PARENT_LINK_CLEAR: MutationPostcondition.PARENT_LINK_CLEARED,
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
        if self.failure is not None and not isinstance(self.failure, FailureEvidence):
            raise IntegrationContractError("mutation failure must be structured evidence")
        if succeeded and self.failure is not None:
            raise IntegrationContractError("successful mutation outcome cannot carry failure evidence")
        if kind is MutationOutcomeKind.RETRYABLE:
            if self.failure is None or not self.failure.retryable:
                raise IntegrationContractError("retryable mutation outcome requires retryable failure evidence")
        elif self.failure is not None and self.failure.retryable:
            raise IntegrationContractError("retryable failure evidence requires a retryable mutation outcome")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "postconditions", postconditions)
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class ChildImportPayload:
    """Immutable child document and identity supplied to the import gateway."""

    parent_uuid: str
    child_uuid: str
    chain_id: str
    target_link: int
    fields: FrozenPairs

    def __post_init__(self) -> None:
        parent_uuid = _required_text(self.parent_uuid, "child parent UUID")
        child_uuid = _required_text(self.child_uuid, "child UUID")
        chain_id = _required_text(self.chain_id, "child chainID")
        try:
            import uuid

            parsed_uuid = uuid.UUID(child_uuid)
        except (ValueError, AttributeError) as exc:
            raise IntegrationContractError("child UUID must be a valid full UUID") from exc
        if str(parsed_uuid) != child_uuid.lower():
            raise IntegrationContractError("child UUID must use canonical UUID form")
        if isinstance(self.target_link, bool) or not isinstance(self.target_link, int) or self.target_link <= 0:
            raise IntegrationContractError("child target link must be a positive integer")
        fields = tuple(self.fields)
        if any(
            not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str)
            for item in fields
        ):
            raise IntegrationContractError("child payload fields must be frozen key/value pairs")
        field_map = dict(fields)
        if str(field_map.get("uuid") or "").strip().lower() != child_uuid.lower():
            raise IntegrationContractError("child payload UUID does not match its identity")
        if str(field_map.get("chainID") or "").strip() != chain_id:
            raise IntegrationContractError("child payload chainID does not match its identity")
        if _coerce_payload_link(field_map.get("link")) != self.target_link:
            raise IntegrationContractError("child payload link does not match its identity")
        if not str(field_map.get("prevLink") or "").strip():
            raise IntegrationContractError("child payload requires prevLink")
        object.__setattr__(self, "parent_uuid", parent_uuid)
        object.__setattr__(self, "child_uuid", str(parsed_uuid))
        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "fields", fields)

    @classmethod
    def from_draft(cls, draft: object, *, parent_uuid: str) -> "ChildImportPayload":
        """Create an import payload only from a validated child draft."""
        from .task_models import TaskDraft

        if not isinstance(draft, TaskDraft):
            raise IntegrationContractError("child import requires a validated TaskDraft")
        return cls(
            parent_uuid=parent_uuid,
            child_uuid=draft.identity.task_uuid.value,
            chain_id=draft.identity.chain_id.value,
            target_link=draft.identity.link.value,
            fields=_freeze_pairs(draft.to_mapping()),
        )

    def to_dict(self) -> dict[str, object]:
        return {key: _thaw(value) for key, value in self.fields}


@dataclass(frozen=True, slots=True)
class ChildCompensationPayload:
    """Mark one imported child deleted when guarded linkage cannot complete."""

    task_uuid: str
    expected_status: str = "pending"

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_uuid", _required_text(self.task_uuid, "compensation task UUID"))
        status = _required_text(self.expected_status, "compensation task status").lower()
        if status not in {"pending", "waiting", "recurring"}:
            raise IntegrationContractError("child compensation requires a mutable task status")
        object.__setattr__(self, "expected_status", status)


def _coerce_payload_link(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


@dataclass(frozen=True, slots=True)
class ParentLinkPayload:
    """Expected parent-link update, separated from its task guard."""

    parent_uuid: str
    child_short_uuid: str
    expected_next_link: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_uuid", _required_text(self.parent_uuid, "parent UUID"))
        object.__setattr__(self, "child_short_uuid", _required_text(self.child_short_uuid, "child short UUID"))
        object.__setattr__(self, "expected_next_link", str(self.expected_next_link or "").strip())


@dataclass(frozen=True, slots=True)
class ParentLinkClearPayload:
    """Clear one optimistic parent link only when its expected value remains."""

    parent_uuid: str
    expected_next_link: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_uuid", _required_text(self.parent_uuid, "parent UUID"))
        object.__setattr__(self, "expected_next_link", _required_text(self.expected_next_link, "expected parent nextLink"))


@dataclass(frozen=True, slots=True)
class ChainDisablePayload:
    """Chain state transition requested for one guarded parent task."""

    task_uuid: str
    expected_chain: str = "on"
    target_chain: str = "off"

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_uuid", _required_text(self.task_uuid, "chain task UUID"))
        expected = str(self.expected_chain or "").strip().lower()
        target = str(self.target_chain or "").strip().lower()
        if expected != "on" or target != "off":
            raise IntegrationContractError("chain disablement must transition chain:on to chain:off")
        object.__setattr__(self, "expected_chain", expected)
        object.__setattr__(self, "target_chain", target)


@dataclass(frozen=True, slots=True)
class NativeUntilRepairPayload:
    """Native-until replacement with its expected prior value."""

    task_uuid: str
    expected_until: str
    replacement_until: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_uuid", _required_text(self.task_uuid, "native-until task UUID"))
        expected = _required_text(self.expected_until, "expected native until")
        replacement = _required_text(self.replacement_until, "replacement native until")
        if expected == replacement:
            raise IntegrationContractError("native-until repair must change the value")
        object.__setattr__(self, "expected_until", expected)
        object.__setattr__(self, "replacement_until", replacement)


@dataclass(frozen=True, slots=True)
class MetadataRepairPayload:
    """Explicit UDA updates for one guarded lifecycle task."""

    task_uuid: str
    updates: FrozenPairs
    expected: FrozenPairs = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_uuid", _required_text(self.task_uuid, "metadata task UUID"))
        updates = tuple(self.updates)
        if not updates or any(
            not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str)
            for item in updates
        ):
            raise IntegrationContractError("metadata repair requires non-empty frozen updates")
        if any(not key.strip() or key in {"uuid", "status"} for key, _value in updates):
            raise IntegrationContractError("metadata repair cannot update UUID or status")
        expected = tuple(self.expected)
        if any(
            not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str)
            for item in expected
        ):
            raise IntegrationContractError("metadata repair expected values must be frozen key/value pairs")
        update_keys = {key for key, _value in updates}
        if any(key not in update_keys for key, _value in expected):
            raise IntegrationContractError("metadata repair expected values must name updated fields")
        object.__setattr__(self, "updates", updates)
        object.__setattr__(self, "expected", expected)

    def to_dict(self) -> dict[str, object]:
        return {key: _thaw(value) for key, value in self.updates}

    def expected_dict(self) -> dict[str, object]:
        return {key: _thaw(value) for key, value in self.expected}


MutationPayload: TypeAlias = (
    ChildImportPayload
    | ChildCompensationPayload
    | ParentLinkPayload
    | ParentLinkClearPayload
    | ChainDisablePayload
    | NativeUntilRepairPayload
    | MetadataRepairPayload
)


@dataclass(frozen=True, slots=True)
class MutationRequest:
    """One named mutation with a typed guard and operation payload."""

    operation: MutationOperation
    guard: MutationGuard
    payload: MutationPayload

    @classmethod
    def child_import(cls, guard: MutationGuard, draft: object) -> "MutationRequest":
        """Build a guarded child-import request from a validated TaskDraft."""
        if not isinstance(guard, MutationGuard):
            raise IntegrationContractError("child import requires a MutationGuard")
        payload = ChildImportPayload.from_draft(draft, parent_uuid=guard.task_uuid)
        return cls(MutationOperation.CHILD_IMPORT, guard, payload)

    @classmethod
    def parent_link(cls, guard: MutationGuard, patch: object) -> "MutationRequest":
        """Build a guarded parent-link request from a typed TaskPatch."""
        from .task_changes import PatchOperation, TaskPatch

        if not isinstance(guard, MutationGuard):
            raise IntegrationContractError("parent link requires a MutationGuard")
        if not isinstance(patch, TaskPatch) or patch.operation is not PatchOperation.PARENT_LINK:
            raise IntegrationContractError("parent link requires a PARENT_LINK TaskPatch")
        if patch.target.value.lower() != guard.task_uuid.lower():
            raise IntegrationContractError("parent-link patch target differs from guard")
        values = patch.set_values()
        child_short = str(values.get("nextLink") or "").strip()
        if not child_short:
            raise IntegrationContractError("parent-link patch requires nextLink")
        return cls(
            MutationOperation.PARENT_LINK,
            guard,
            ParentLinkPayload(guard.task_uuid, child_short),
        )

    @classmethod
    def chain_disable(cls, guard: MutationGuard, patch: object) -> "MutationRequest":
        """Build a guarded chain-disable request from a typed TaskPatch."""
        from .task_changes import PatchOperation, TaskPatch

        if not isinstance(guard, MutationGuard):
            raise IntegrationContractError("chain disable requires a MutationGuard")
        if not isinstance(patch, TaskPatch) or patch.operation is not PatchOperation.CHAIN_DISABLE:
            raise IntegrationContractError("chain disable requires a CHAIN_DISABLE TaskPatch")
        if patch.target.value.lower() != guard.task_uuid.lower():
            raise IntegrationContractError("chain-disable patch target differs from guard")
        values = patch.set_values()
        if str(values.get("chain") or "").strip().lower() != "off":
            raise IntegrationContractError("chain-disable patch must set chain=off")
        return cls(
            MutationOperation.CHAIN_DISABLE,
            guard,
            ChainDisablePayload(guard.task_uuid, expected_chain=guard.chain, target_chain="off"),
        )

    @classmethod
    def native_until_repair(cls, guard: MutationGuard, patch: object) -> "MutationRequest":
        """Build a native-until repair from typed patch and guard evidence."""
        from .task_changes import PatchOperation, TaskPatch

        if not isinstance(guard, MutationGuard):
            raise IntegrationContractError("native-until repair requires a MutationGuard")
        if not isinstance(patch, TaskPatch) or patch.operation is not PatchOperation.NATIVE_UNTIL_REPAIR:
            raise IntegrationContractError("native-until repair requires a NATIVE_UNTIL_REPAIR TaskPatch")
        if patch.target.value.lower() != guard.task_uuid.lower():
            raise IntegrationContractError("native-until patch target differs from guard")
        replacement = str(patch.set_values().get("until") or "").strip()
        expected = next(
            (item.value for item in guard.timestamps if item.field is GuardTimestampField.UNTIL),
            "",
        )
        if not expected:
            raise IntegrationContractError("native-until repair requires guarded until evidence")
        return cls(
            MutationOperation.NATIVE_UNTIL_REPAIR,
            guard,
            NativeUntilRepairPayload(guard.task_uuid, expected, replacement),
        )

    @classmethod
    def metadata_repair(
        cls,
        guard: MutationGuard,
        patch: object,
        *,
        expected: Mapping[str, object] | None = None,
    ) -> "MutationRequest":
        """Build a guarded metadata repair from a typed TaskPatch."""
        from .task_changes import PatchOperation, TaskPatch

        if not isinstance(guard, MutationGuard):
            raise IntegrationContractError("metadata repair requires a MutationGuard")
        if not isinstance(patch, TaskPatch) or patch.operation is not PatchOperation.METADATA_REPAIR:
            raise IntegrationContractError("metadata repair requires a METADATA_REPAIR TaskPatch")
        if patch.target.value.lower() != guard.task_uuid.lower():
            raise IntegrationContractError("metadata patch target differs from guard")
        values = dict(patch.set_values())
        cleared = patch.clear_fields()
        if cleared:
            values.update({field: None for field in cleared})
        return cls(
            MutationOperation.METADATA_REPAIR,
            guard,
            MetadataRepairPayload(guard.task_uuid, _freeze_pairs(values), _freeze_pairs(expected or {})),
        )

    @classmethod
    def ordinary_carry(cls, guard: MutationGuard, patch: object) -> "MutationRequest":
        """Build a metadata-gateway request from an ordinary-carry patch."""
        from .task_changes import PatchOperation, TaskPatch

        if not isinstance(guard, MutationGuard):
            raise IntegrationContractError("ordinary carry requires a MutationGuard")
        if not isinstance(patch, TaskPatch) or patch.operation is not PatchOperation.ORDINARY_CARRY:
            raise IntegrationContractError("ordinary carry requires an ORDINARY_CARRY TaskPatch")
        if patch.target.value.lower() != guard.task_uuid.lower():
            raise IntegrationContractError("ordinary-carry patch target differs from guard")
        values = dict(patch.set_values())
        values.update({field: None for field in patch.clear_fields()})
        return cls(
            MutationOperation.METADATA_REPAIR,
            guard,
            MetadataRepairPayload(guard.task_uuid, _freeze_pairs(values)),
        )

    def __post_init__(self) -> None:
        try:
            operation = MutationOperation(self.operation)
        except (TypeError, ValueError) as exc:
            raise IntegrationContractError("invalid mutation request operation") from exc
        if not isinstance(self.guard, MutationGuard):
            raise IntegrationContractError("mutation request requires a MutationGuard")
        expected = {
            MutationOperation.CHILD_IMPORT: ChildImportPayload,
            MutationOperation.CHILD_COMPENSATION: ChildCompensationPayload,
            MutationOperation.PARENT_LINK: ParentLinkPayload,
            MutationOperation.PARENT_LINK_CLEAR: ParentLinkClearPayload,
            MutationOperation.CHAIN_DISABLE: ChainDisablePayload,
            MutationOperation.NATIVE_UNTIL_REPAIR: NativeUntilRepairPayload,
            MutationOperation.METADATA_REPAIR: MetadataRepairPayload,
        }[operation]
        if not isinstance(self.payload, expected):
            raise IntegrationContractError(
                f"{operation.value} request requires {expected.__name__}"
            )
        task_uuid = getattr(self.payload, "task_uuid", None)
        if task_uuid is None and isinstance(self.payload, (ChildImportPayload, ParentLinkPayload, ParentLinkClearPayload)):
            task_uuid = self.payload.parent_uuid
        if str(task_uuid).strip().lower() != self.guard.task_uuid.lower():
            raise IntegrationContractError("mutation payload and guard task UUID differ")
        if isinstance(self.payload, ChildImportPayload):
            if self.payload.chain_id != self.guard.chain_id:
                raise IntegrationContractError("child payload and guard chainID differ")
            if self.payload.target_link <= self.guard.link:
                raise IntegrationContractError("child target link must follow the guarded parent link")
        object.__setattr__(self, "operation", operation)


class TaskwarriorMutationPort(Protocol):
    """Named mutation surface shared by lifecycle and operator adapters."""

    def apply(self, request: MutationRequest) -> MutationOutcome: ...

    def import_child(self, request: MutationRequest) -> MutationOutcome: ...

    def compensate_child(self, request: MutationRequest) -> MutationOutcome: ...

    def compensate_imported_child(self, request: MutationRequest) -> MutationOutcome: ...

    def link_parent(self, request: MutationRequest) -> MutationOutcome: ...

    def clear_parent_link(self, request: MutationRequest) -> MutationOutcome: ...

    def disable_chain(self, request: MutationRequest) -> MutationOutcome: ...

    def repair_native_until(self, request: MutationRequest) -> MutationOutcome: ...

    def repair_metadata(self, request: MutationRequest) -> MutationOutcome: ...


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
        try:
            mutations = tuple(self.mutations)
        except TypeError as exc:
            raise IntegrationContractError("outbox outcome requires typed mutation outcomes") from exc
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
        elif any(item.kind not in success_kinds for item in mutations):
            raise IntegrationContractError("advanced outbox outcome can contain only successful mutations")

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
    "ChainDisablePayload",
    "ChildCompensationPayload",
    "ChildImportPayload",
    "FailureEvidence",
    "Found",
    "GuardTimestamp",
    "GuardTimestampField",
    "IntegrationContractError",
    "MutationGuard",
    "MutationOperation",
    "MutationOutcome",
    "MutationOutcomeKind",
    "MutationPayload",
    "MutationPostcondition",
    "MutationRequest",
    "MetadataRepairPayload",
    "NativeUntilRepairPayload",
    "OutboxIntent",
    "OutboxOutcome",
    "OutboxOutcomeKind",
    "OutboxStage",
    "ParentLinkPayload",
    "ParentLinkClearPayload",
    "TaskwarriorMutationPort",
    "TaskCommand",
    "TaskCommandResult",
    "TaskRead",
    "Unavailable",
)
