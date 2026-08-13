"""Typed, invocation-scoped Taskwarrior task reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from .integration_models import TaskCommandResult


TaskRow: TypeAlias = Mapping[str, Any]
TaskSlot: TypeAlias = tuple[str, int]


class TaskQueryKind(str, Enum):
    BROAD = "broad"
    UUID = "uuid"
    CHILD_SLOT = "child_slot"
    PREDECESSOR_SLOT = "predecessor_slot"
    CHAIN = "chain"
    ACTIVE_ROOTS = "active_roots"
    LIFECYCLE_CANDIDATES = "lifecycle_candidates"
    VERIFICATION = "verification"


@dataclass(frozen=True, slots=True)
class TaskSnapshotScope:
    """The exact state for which an exported snapshot is authoritative."""

    kind: TaskQueryKind
    identity: str
    statuses: tuple[str, ...]
    complete_chain_history: bool = False

    def __post_init__(self) -> None:
        try:
            kind = TaskQueryKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid task snapshot scope") from exc
        identity = str(self.identity or "").strip()
        if not identity:
            raise ValueError("task snapshot scope requires an identity")
        statuses = tuple(sorted({str(item).strip().lower() for item in self.statuses if str(item).strip()}))
        if not statuses:
            raise ValueError("task snapshot scope requires included statuses")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "complete_chain_history", bool(self.complete_chain_history))


def _link_number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(float(str(value)))
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _append_index(index: dict[Any, list[TaskRow]], key: Any, row: TaskRow) -> None:
    if key not in (None, "", ("", None)):
        index.setdefault(key, []).append(row)


def _freeze_index(index: dict[Any, list[TaskRow]]) -> Mapping[Any, tuple[TaskRow, ...]]:
    return MappingProxyType({key: tuple(values) for key, values in index.items()})


@dataclass(frozen=True, slots=True)
class AuthoritativeTaskSnapshot:
    """One parsed export plus indexes that preserve its authority scope."""

    scope: TaskSnapshotScope
    rows: tuple[TaskRow, ...]
    command_result: TaskCommandResult
    by_uuid: Mapping[str, tuple[TaskRow, ...]] = field(init=False, repr=False)
    by_short_uuid: Mapping[str, tuple[TaskRow, ...]] = field(init=False, repr=False)
    by_chain: Mapping[str, tuple[TaskRow, ...]] = field(init=False, repr=False)
    by_slot: Mapping[TaskSlot, tuple[TaskRow, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scope, TaskSnapshotScope):
            raise TypeError("authoritative snapshot requires a TaskSnapshotScope")
        if not isinstance(self.command_result, TaskCommandResult) or not self.command_result.ok:
            raise ValueError("authoritative snapshot requires a successful command result")
        copied_rows: list[TaskRow] = []
        uuid_index: dict[str, list[TaskRow]] = {}
        short_index: dict[str, list[TaskRow]] = {}
        chain_index: dict[str, list[TaskRow]] = {}
        slot_index: dict[TaskSlot, list[TaskRow]] = {}
        for raw_row in self.rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError("authoritative snapshot contains a non-object row")
            row: TaskRow = MappingProxyType(dict(raw_row))
            copied_rows.append(row)
            uuid_value = str(row.get("uuid") or "").strip().lower()
            chain_id = str(row.get("chainID") or "").strip()
            link = _link_number(row.get("link"))
            _append_index(uuid_index, uuid_value, row)
            _append_index(short_index, uuid_value[:8], row)
            _append_index(chain_index, chain_id, row)
            if chain_id and link is not None:
                _append_index(slot_index, (chain_id, link), row)
        object.__setattr__(self, "rows", tuple(copied_rows))
        object.__setattr__(self, "by_uuid", _freeze_index(uuid_index))
        object.__setattr__(self, "by_short_uuid", _freeze_index(short_index))
        object.__setattr__(self, "by_chain", _freeze_index(chain_index))
        object.__setattr__(self, "by_slot", _freeze_index(slot_index))

    def uuid_matches(self, uuid_value: str) -> tuple[TaskRow, ...]:
        identity = str(uuid_value or "").strip().lower()
        if not identity:
            return ()
        if len(identity) >= 32 or "-" in identity:
            return self.by_uuid.get(identity, ())
        if len(identity) == 8:
            return self.by_short_uuid.get(identity, ())
        return tuple(row for row in self.rows if str(row.get("uuid") or "").lower().startswith(identity))

    def chain_rows(self, chain_id: str) -> tuple[TaskRow, ...]:
        return self.by_chain.get(str(chain_id or "").strip(), ())

    def slot_rows(self, chain_id: str, link: int) -> tuple[TaskRow, ...]:
        return self.by_slot.get((str(chain_id or "").strip(), int(link)), ())


__all__ = (
    "AuthoritativeTaskSnapshot",
    "TaskQueryKind",
    "TaskRow",
    "TaskSlot",
    "TaskSnapshotScope",
)
