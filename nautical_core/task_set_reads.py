"""Authoritative bounded set-read contracts for Taskwarrior state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence, cast
from uuid import UUID

from .integration_models import Absent, FailureEvidence, Found, TaskCommandResult, TaskRead, Unavailable
from .task_models import FieldPresence, TaskObservation


_CHAIN_ID_RE = re.compile(r"^[0-9a-fA-F]{8,64}$")
_DEFAULT_STATUSES = ("completed", "deleted", "pending", "recurring", "waiting")


class SetReadStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    DUPLICATE = "duplicate"
    CONTRADICTORY = "contradictory"
    AMBIGUOUS = "ambiguous"
    MALFORMED = "malformed"
    TRUNCATED = "truncated"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


def _statuses(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(sorted({str(value).strip().lower() for value in values if str(value).strip()}))
    if not normalized:
        raise ValueError("set read requires at least one status")
    return normalized


def _uuid(value: object) -> str:
    try:
        return str(UUID(str(value).strip())).lower()
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"set read requires a full UUID: {value!r}") from exc


def _chain_id(value: object) -> str:
    text = str(value or "").strip().lower()
    if not _CHAIN_ID_RE.fullmatch(text):
        raise ValueError(f"set read requires a valid chainID: {value!r}")
    return text


@dataclass(frozen=True, slots=True)
class UUIDSetRequest:
    uuids: tuple[str, ...]
    statuses: tuple[str, ...] = _DEFAULT_STATUSES
    refresh: bool = False
    expected_mutation_epoch: int | None = None
    max_chunk_size: int = 32
    max_query_length: int = 4096

    def __post_init__(self) -> None:
        identities = tuple(sorted({_uuid(value) for value in self.uuids}))
        if not identities:
            raise ValueError("UUID set read requires at least one identity")
        if self.expected_mutation_epoch is not None and (
            isinstance(self.expected_mutation_epoch, bool) or self.expected_mutation_epoch < 0
        ):
            raise ValueError("expected mutation epoch must be non-negative")
        if self.max_chunk_size <= 0 or self.max_query_length < 256:
            raise ValueError("set read chunk limits are invalid")
        object.__setattr__(self, "uuids", identities)
        object.__setattr__(self, "statuses", _statuses(self.statuses))
        object.__setattr__(self, "refresh", bool(self.refresh))


@dataclass(frozen=True, slots=True)
class ChainSlot:
    chain_id: str
    link: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain_id", _chain_id(self.chain_id))
        if isinstance(self.link, bool) or int(self.link) <= 0:
            raise ValueError("chain slot link must be positive")
        object.__setattr__(self, "link", int(self.link))


@dataclass(frozen=True, slots=True)
class ChainSlotSetRequest:
    slots: tuple[ChainSlot, ...]
    statuses: tuple[str, ...] = _DEFAULT_STATUSES
    expected_predecessors: Mapping[ChainSlot, str] = field(default_factory=dict)
    complete_chain_history: bool = False
    refresh: bool = False
    expected_mutation_epoch: int | None = None
    max_chunk_size: int = 32
    max_query_length: int = 4096

    def __post_init__(self) -> None:
        slots = tuple(sorted({slot if isinstance(slot, ChainSlot) else ChainSlot(*slot) for slot in self.slots}, key=lambda item: (item.chain_id, item.link)))
        if not slots:
            raise ValueError("chain-slot set read requires at least one slot")
        if self.expected_mutation_epoch is not None and (
            isinstance(self.expected_mutation_epoch, bool) or self.expected_mutation_epoch < 0
        ):
            raise ValueError("expected mutation epoch must be non-negative")
        if self.max_chunk_size <= 0 or self.max_query_length < 256:
            raise ValueError("set read chunk limits are invalid")
        predecessors = {
            slot: str(value).strip().lower()
            for slot, value in self.expected_predecessors.items()
            if str(value).strip()
        }
        if any(slot not in slots for slot in predecessors):
            raise ValueError("expected predecessor contains an unrequested slot")
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "statuses", _statuses(self.statuses))
        object.__setattr__(self, "expected_predecessors", MappingProxyType(predecessors))
        object.__setattr__(self, "refresh", bool(self.refresh))
        object.__setattr__(self, "complete_chain_history", bool(self.complete_chain_history))


@dataclass(frozen=True, slots=True)
class SetReadResult:
    status: SetReadStatus
    requested: tuple[object, ...]
    found: Mapping[object, TaskObservation] = field(default_factory=dict)
    absent: tuple[object, ...] = ()
    evidence: tuple[str, ...] = ()
    mutation_epoch: int = 0
    complete_for_requested_identities: bool = False
    failures: tuple[FailureEvidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SetReadStatus(self.status))
        object.__setattr__(self, "requested", tuple(self.requested))
        object.__setattr__(self, "found", MappingProxyType(dict(self.found)))
        object.__setattr__(self, "absent", tuple(self.absent))
        object.__setattr__(self, "evidence", tuple(str(item) for item in self.evidence if str(item)))
        object.__setattr__(self, "complete_for_requested_identities", bool(self.complete_for_requested_identities))
        failures = tuple(self.failures)
        if any(not isinstance(item, FailureEvidence) for item in failures):
            raise TypeError("set read failures require FailureEvidence")
        object.__setattr__(self, "failures", failures)


class _Repository(Protocol):
    @property
    def mutation_epoch(self) -> int: ...

    def broad_snapshot(
        self,
        *,
        identity: str,
        filters: Sequence[str],
        statuses: Sequence[str],
        complete_chain_history: bool = False,
        refresh: bool = False,
    ) -> TaskRead[object]: ...


class _Snapshot(Protocol):
    rows: tuple[TaskObservation, ...]
    truncated: bool
    command_result: TaskCommandResult


def _row_text(row: TaskObservation, field_name: str) -> str:
    value = row.field(field_name)
    if value.presence is not FieldPresence.VALUE:
        return ""
    return str(getattr(value.value, "value", value.value) or "").strip()


def _or_filters(groups: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    tokens: list[str] = ["("]
    for index, group in enumerate(groups):
        if index:
            tokens.append("or")
        tokens.extend(("(", *group, ")"))
    tokens.append(")")
    return tuple(tokens)


def uuid_set_filters(uuids: tuple[str, ...]) -> tuple[str, ...]:
    """Build a bounded boolean Taskwarrior filter for canonical UUIDs."""
    return _or_filters(tuple((f"uuid:{_uuid(value)}",) for value in uuids))


def chain_slot_set_filters(slots: tuple[ChainSlot, ...]) -> tuple[str, ...]:
    """Build a bounded boolean filter for typed chain/link slots."""
    normalized = tuple(slot if isinstance(slot, ChainSlot) else ChainSlot(*slot) for slot in slots)
    return _or_filters(tuple((f"chainID:{slot.chain_id}", f"link:{slot.link}") for slot in normalized))


class AuthoritativeSetReadService:
    """Resolve bounded identity sets without widening their authority."""

    def __init__(self, repository: _Repository) -> None:
        self._repository = repository

    def _epoch_ok(self, expected: int | None) -> bool:
        return expected is None or int(self._repository.mutation_epoch) == expected

    @staticmethod
    def _chunks(values: tuple[object, ...], request: UUIDSetRequest | ChainSlotSetRequest) -> tuple[tuple[object, ...], ...]:
        chunks: list[tuple[object, ...]] = []
        current: list[object] = []
        length = 0
        for value in values:
            rendered = str(value)
            extra = len(rendered) + 18
            if current and (len(current) >= request.max_chunk_size or length + extra > request.max_query_length):
                chunks.append(tuple(current))
                current, length = [], 0
            current.append(value)
            length += extra
        if current:
            chunks.append(tuple(current))
        return tuple(chunks)

    def read_uuids(self, request: UUIDSetRequest) -> SetReadResult:
        if not isinstance(request, UUIDSetRequest):
            raise TypeError("UUID set read requires UUIDSetRequest")
        if not self._epoch_ok(request.expected_mutation_epoch):
            return SetReadResult(SetReadStatus.STALE, request.uuids, evidence=("mutation epoch changed before read",), mutation_epoch=int(self._repository.mutation_epoch))
        found: dict[object, TaskObservation] = {}
        absent: list[object] = []
        evidence: list[str] = []
        chunks = self._chunks(request.uuids, request)
        for index, raw_chunk in enumerate(chunks):
            chunk = tuple(str(value) for value in raw_chunk)
            read = self._repository.broad_snapshot(
                identity=f"uuid-set:{index}:{len(chunk)}",
                filters=uuid_set_filters(tuple(str(value) for value in chunk)),
                statuses=request.statuses,
                refresh=request.refresh,
            )
            if isinstance(read, Unavailable):
                evidence.append(f"chunk {index + 1}/{len(chunks)} unavailable: {read.evidence.detail}")
                return SetReadResult(SetReadStatus.PARTIAL if found else SetReadStatus.UNAVAILABLE, request.uuids, found, tuple(absent), tuple(evidence), int(self._repository.mutation_epoch), False, (read.evidence,))
            if isinstance(read, Absent):
                absent.extend(chunk)
                continue
            snapshot = cast(_Snapshot, read.value)
            if snapshot.truncated:
                return SetReadResult(SetReadStatus.TRUNCATED, request.uuids, found, tuple(absent), (f"chunk {index + 1} was truncated",), int(self._repository.mutation_epoch))
            for row in snapshot.rows:
                identity = _row_text(row, "uuid").lower()
                if not identity or identity not in chunk:
                    return SetReadResult(SetReadStatus.MALFORMED, request.uuids, found, tuple(absent), ("set export contained a malformed or unrelated row",), int(self._repository.mutation_epoch))
                if identity in found:
                    return SetReadResult(SetReadStatus.DUPLICATE, request.uuids, found, tuple(absent), (f"duplicate UUID {identity}",), int(self._repository.mutation_epoch))
                found[identity] = row
            absent.extend(str(identity) for identity in chunk if identity not in found and identity not in absent)
        if not self._epoch_ok(request.expected_mutation_epoch):
            return SetReadResult(SetReadStatus.STALE, request.uuids, found, tuple(absent), ("mutation epoch changed during read",), int(self._repository.mutation_epoch))
        return SetReadResult(SetReadStatus.COMPLETE, request.uuids, found, tuple(absent), tuple(evidence), int(self._repository.mutation_epoch), True)

    def read_slots(self, request: ChainSlotSetRequest) -> SetReadResult:
        if not isinstance(request, ChainSlotSetRequest):
            raise TypeError("chain-slot set read requires ChainSlotSetRequest")
        if not self._epoch_ok(request.expected_mutation_epoch):
            return SetReadResult(SetReadStatus.STALE, request.slots, evidence=("mutation epoch changed before read",), mutation_epoch=int(self._repository.mutation_epoch))
        found: dict[object, TaskObservation] = {}
        absent: list[ChainSlot] = []
        slots = request.slots
        chunks = self._chunks(slots, request)
        for index, raw_chunk in enumerate(chunks):
            chunk = tuple(value for value in raw_chunk if isinstance(value, ChainSlot))
            filters = chain_slot_set_filters(tuple(slot for slot in chunk if isinstance(slot, ChainSlot)))
            if request.complete_chain_history:
                read = self._repository.broad_snapshot(
                    identity=f"slot-set:{index}:{len(chunk)}",
                    filters=filters,
                    statuses=request.statuses,
                    complete_chain_history=True,
                    refresh=request.refresh,
                )
            else:
                read = self._repository.broad_snapshot(
                    identity=f"slot-set:{index}:{len(chunk)}",
                    filters=filters,
                    statuses=request.statuses,
                    refresh=request.refresh,
                )
            if isinstance(read, Unavailable):
                detail = f"chunk {index + 1}/{len(chunks)} unavailable: {read.evidence.detail}"
                return SetReadResult(SetReadStatus.PARTIAL if found else SetReadStatus.UNAVAILABLE, slots, found, tuple(absent), (detail,), int(self._repository.mutation_epoch), False, (read.evidence,))
            if isinstance(read, Absent):
                absent.extend(chunk)
                continue
            snapshot = cast(_Snapshot, read.value)
            if snapshot.truncated:
                return SetReadResult(SetReadStatus.TRUNCATED, slots, found, tuple(absent), (f"chunk {index + 1} was truncated",), int(self._repository.mutation_epoch))
            for row in snapshot.rows:
                chain_id = _row_text(row, "chainID").lower()
                try:
                    link = int(float(_row_text(row, "link")))
                except ValueError:
                    link = 0
                slot = ChainSlot(chain_id, link) if chain_id and link > 0 else None
                if slot is None or slot not in chunk:
                    return SetReadResult(SetReadStatus.MALFORMED, slots, found, tuple(absent), ("set export contained a malformed or unrelated slot",), int(self._repository.mutation_epoch))
                if slot in found:
                    return SetReadResult(SetReadStatus.DUPLICATE, slots, found, tuple(absent), (f"duplicate slot {slot.chain_id}:{slot.link}",), int(self._repository.mutation_epoch))
                expected = request.expected_predecessors.get(slot)
                if expected and _row_text(row, "prevLink").lower() != expected:
                    return SetReadResult(SetReadStatus.CONTRADICTORY, slots, found, tuple(absent), (f"predecessor mismatch for {slot.chain_id}:{slot.link}",), int(self._repository.mutation_epoch))
                found[slot] = row
            absent.extend(slot for slot in chunk if isinstance(slot, ChainSlot) and slot not in found and slot not in absent)
        if not self._epoch_ok(request.expected_mutation_epoch):
            return SetReadResult(SetReadStatus.STALE, slots, found, tuple(absent), ("mutation epoch changed during read",), int(self._repository.mutation_epoch))
        return SetReadResult(SetReadStatus.COMPLETE, slots, found, tuple(absent), (), int(self._repository.mutation_epoch), True)


__all__ = [
    "AuthoritativeSetReadService",
    "ChainSlot",
    "ChainSlotSetRequest",
    "SetReadResult",
    "SetReadStatus",
    "UUIDSetRequest",
    "chain_slot_set_filters",
    "uuid_set_filters",
]
