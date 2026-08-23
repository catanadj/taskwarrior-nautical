"""Explicit set/clear mutation semantics for the task domain model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

from .task_models import ChainID, TaskTimestamp, TaskUUID


class TaskChangeError(ValueError):
    """A task patch is ambiguous, unsafe, or violates its operation policy."""


def timestamp_equal(left: object, right: object) -> bool:
    """Compare Taskwarrior compact and ISO timestamps by UTC instant."""
    def parse(value: object) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    left_value = parse(left)
    right_value = parse(right)
    return left_value is not None and right_value is not None and left_value == right_value


class ChangeAction(str, Enum):
    SET = "set"
    CLEAR = "clear"


class PatchOperation(str, Enum):
    PARENT_LINK = "parent_link"
    CHAIN_DISABLE = "chain_disable"
    NATIVE_UNTIL_REPAIR = "native_until_repair"
    METADATA_REPAIR = "metadata_repair"
    RECURRENCE_ACTIVATION = "recurrence_activation"
    ORDINARY_CARRY = "ordinary_carry"


_VOLATILE_FIELDS = frozenset({"id", "urgency", "modified", "end"})
_IMMUTABLE_FIELDS = frozenset({"uuid", "chainID", "link", "prevLink"})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (TaskUUID, ChainID, TaskTimestamp)):
        return str(value.value) if hasattr(value, "value") else value
    raise TaskChangeError(f"unsupported patch value: {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw(item) for key, item in value}
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TaskChange:
    field: str
    action: ChangeAction
    value: Any = None

    def __post_init__(self) -> None:
        field = str(self.field).strip()
        if not field:
            raise TaskChangeError("task change requires a field")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "action", ChangeAction(self.action))
        if self.action is ChangeAction.SET:
            object.__setattr__(self, "value", _freeze(self.value))
        elif self.value is not None:
            raise TaskChangeError(f"clearing {field} cannot carry a value")


@dataclass(frozen=True, slots=True, eq=False)
class TaskPatch:
    target: TaskUUID
    operation: PatchOperation
    changes: tuple[TaskChange, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.target, TaskUUID):
            raise TaskChangeError("task patch requires a typed target UUID")
        object.__setattr__(self, "operation", PatchOperation(self.operation))
        changes = tuple(self.changes)
        if not changes or any(not isinstance(item, TaskChange) for item in changes):
            raise TaskChangeError("task patch requires typed changes")
        fields = [item.field for item in changes]
        if len(fields) != len(set(fields)):
            raise TaskChangeError("a task patch cannot change one field more than once")
        forbidden = _VOLATILE_FIELDS.intersection(fields)
        if forbidden:
            raise TaskChangeError(f"volatile fields cannot be patched: {', '.join(sorted(forbidden))}")
        immutable = _IMMUTABLE_FIELDS.intersection(fields)
        if immutable and self.operation not in {PatchOperation.PARENT_LINK, PatchOperation.CHAIN_DISABLE}:
            raise TaskChangeError(f"immutable fields require a named structural operation: {', '.join(sorted(immutable))}")
        object.__setattr__(self, "changes", changes)

    @classmethod
    def set(cls, target: TaskUUID, operation: PatchOperation, **fields: Any) -> "TaskPatch":
        if not fields:
            raise TaskChangeError("set patch requires at least one field")
        return cls(target, operation, tuple(TaskChange(key, ChangeAction.SET, value) for key, value in fields.items()))

    @classmethod
    def clear(cls, target: TaskUUID, operation: PatchOperation, *fields: str) -> "TaskPatch":
        if not fields:
            raise TaskChangeError("clear patch requires at least one field")
        return cls(target, operation, tuple(TaskChange(field, ChangeAction.CLEAR) for field in fields))

    @classmethod
    def parent_link(cls, parent: TaskUUID, child: TaskUUID) -> "TaskPatch":
        return cls.set(parent, PatchOperation.PARENT_LINK, nextLink=str(child))

    @classmethod
    def chain_disable(cls, target: TaskUUID) -> "TaskPatch":
        return cls.set(target, PatchOperation.CHAIN_DISABLE, chain="off")

    @classmethod
    def native_until_repair(cls, target: TaskUUID, until: TaskTimestamp) -> "TaskPatch":
        if not isinstance(until, TaskTimestamp):
            raise TaskChangeError("native-until repair requires a typed timestamp")
        return cls.set(target, PatchOperation.NATIVE_UNTIL_REPAIR, until=until.value.isoformat().replace("+00:00", "Z"))

    @classmethod
    def metadata_repair(cls, target: TaskUUID, **fields: Any) -> "TaskPatch":
        return cls.set(target, PatchOperation.METADATA_REPAIR, **fields)

    @classmethod
    def recurrence_activation(cls, target: TaskUUID, **fields: Any) -> "TaskPatch":
        return cls.set(target, PatchOperation.RECURRENCE_ACTIVATION, **fields)

    @classmethod
    def ordinary_carry(cls, target: TaskUUID, **fields: Any) -> "TaskPatch":
        return cls.set(target, PatchOperation.ORDINARY_CARRY, **fields)

    @property
    def fingerprint(self) -> str:
        payload = {
            "target": self.target.value,
            "operation": self.operation.value,
            "changes": [
                (item.field, item.action.value, _thaw(item.value))
                for item in sorted(self.changes, key=lambda change: change.field)
            ],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:24]

    def set_values(self) -> Mapping[str, Any]:
        return MappingProxyType({item.field: _thaw(item.value) for item in self.changes if item.action is ChangeAction.SET})

    def clear_fields(self) -> tuple[str, ...]:
        return tuple(item.field for item in self.changes if item.action is ChangeAction.CLEAR)

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned, lossless patch envelope for persistence."""
        return {
            "schema_version": 1,
            "target": self.target.value,
            "operation": self.operation.value,
            "changes": [
                {
                    "field": item.field,
                    "action": item.action.value,
                    **({"value": _thaw(item.value)} if item.action is ChangeAction.SET else {}),
                }
                for item in self.changes
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskPatch":
        """Decode only the current strict patch schema."""
        if not isinstance(value, Mapping) or value.get("schema_version") != 1:
            raise TaskChangeError("unsupported task patch schema")
        try:
            target = TaskUUID(str(value["target"]))
            operation = PatchOperation(value["operation"])
            changes = tuple(
                TaskChange(
                    str(item["field"]),
                    ChangeAction(item["action"]),
                    item.get("value"),
                )
                for item in value["changes"]
                if isinstance(item, Mapping)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TaskChangeError(f"invalid task patch: {exc}") from exc
        if not changes:
            raise TaskChangeError("task patch requires changes")
        return cls(target, operation, changes)


__all__ = (
    "ChangeAction",
    "PatchOperation",
    "TaskChange",
    "TaskChangeError",
    "TaskPatch",
    "timestamp_equal",
)
