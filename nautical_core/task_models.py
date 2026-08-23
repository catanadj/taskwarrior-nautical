"""Immutable Taskwarrior task observations and domain-facing value types.

This module deliberately does not decide whether a task is operationally valid.
It records what Taskwarrior returned, including malformed known fields, so later
validation and integrity tooling can make that decision with complete evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias
from uuid import UUID


FrozenValue: TypeAlias = object
_MISSING = object()
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$")
_SHORT_REF_RE = re.compile(r"^[0-9a-fA-F]{1,8}$")
_KNOWN_STATUSES = frozenset({"pending", "waiting", "completed", "deleted", "recurring"})
_SEMANTIC_EXCLUSIONS = frozenset({"id", "urgency"})


class FieldPresence(str, Enum):
    ABSENT = "absent"
    NULL = "null"
    VALUE = "value"


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TaskStatus(str, Enum):
    PENDING = "pending"
    WAITING = "waiting"
    COMPLETED = "completed"
    DELETED = "deleted"
    RECURRING = "recurring"


def _freeze(value: Any) -> FrozenValue:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported task field value: {type(value).__name__}")


def _thaw(value: FrozenValue) -> Any:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw(item) for key, item in value}
        return [_thaw(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _semantic_value(value: Any) -> Any:
    """Convert typed scalar wrappers to stable JSON-compatible values."""
    if isinstance(value, (TaskUUID, ShortUUIDRef, ChainID)):
        return value.value
    if isinstance(value, TaskLink):
        return value.value
    if isinstance(value, TaskTimestamp):
        return value.value.isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        return {str(key): _semantic_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TaskUUID:
    """A canonical full Taskwarrior UUID."""

    value: str

    def __post_init__(self) -> None:
        text = str(self.value).strip().lower()
        if not _UUID_RE.fullmatch(text):
            raise ValueError("task UUID must be a canonical full UUID")
        object.__setattr__(self, "value", str(UUID(text)))

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ShortUUIDRef:
    """A short hexadecimal Taskwarrior UUID reference."""

    value: str

    def __post_init__(self) -> None:
        text = str(self.value).strip().lower()
        if not _SHORT_REF_RE.fullmatch(text):
            raise ValueError("short UUID reference must contain one to eight hexadecimal characters")
        object.__setattr__(self, "value", text)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ChainID:
    value: str

    def __post_init__(self) -> None:
        text = str(self.value).strip()
        if not text:
            raise ValueError("chainID cannot be empty")
        object.__setattr__(self, "value", text)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class TaskLink:
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value <= 0:
            raise ValueError("task link must be a positive integer")


@dataclass(frozen=True, slots=True)
class TaskTimestamp:
    """An aware UTC timestamp while the original encoding remains in FieldState."""

    value: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.value, datetime) or self.value.tzinfo is None:
            raise ValueError("task timestamp must be timezone-aware")
        object.__setattr__(self, "value", self.value.astimezone(timezone.utc))


@dataclass(frozen=True, slots=True)
class FieldState:
    """Lossless presence/raw/normalized value for one task field."""

    presence: FieldPresence
    raw: FrozenValue = None
    value: FrozenValue = None

    def __post_init__(self) -> None:
        presence = FieldPresence(self.presence)
        object.__setattr__(self, "presence", presence)
        if presence is FieldPresence.ABSENT:
            object.__setattr__(self, "raw", None)
            object.__setattr__(self, "value", None)

    @classmethod
    def absent(cls) -> "FieldState":
        return cls(FieldPresence.ABSENT)

    @classmethod
    def from_raw(cls, raw: Any, value: FrozenValue = _MISSING) -> "FieldState":
        if raw is None:
            return cls(FieldPresence.NULL)
        frozen_raw = _freeze(raw)
        return cls(FieldPresence.VALUE, frozen_raw, frozen_raw if value is _MISSING else value)

    def raw_value(self) -> Any:
        return _thaw(self.raw) if self.presence is FieldPresence.VALUE else None


@dataclass(frozen=True, slots=True)
class DecodeIssue:
    field: str
    code: str
    message: str
    severity: IssueSeverity = IssueSeverity.ERROR
    raw: FrozenValue = None

    def __post_init__(self) -> None:
        for name in ("field", "code", "message"):
            text = str(getattr(self, name)).strip()
            if not text:
                raise ValueError(f"decode issue {name} is required")
            object.__setattr__(self, name, text)
        object.__setattr__(self, "severity", IssueSeverity(self.severity))


@dataclass(frozen=True, slots=True)
class ObservationProvenance:
    source_query: str
    snapshot_id: str = ""
    mutation_epoch: int = 0
    command_count: int = 0

    def __post_init__(self) -> None:
        query = str(self.source_query).strip()
        if not query:
            raise ValueError("observation provenance requires a source query")
        if isinstance(self.mutation_epoch, bool) or self.mutation_epoch < 0:
            raise ValueError("observation mutation epoch must be non-negative")
        if isinstance(self.command_count, bool) or self.command_count < 0:
            raise ValueError("observation command count must be non-negative")
        object.__setattr__(self, "source_query", query)
        object.__setattr__(self, "snapshot_id", str(self.snapshot_id or "").strip())


def _timestamp(value: Any) -> TaskTimestamp:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "T" not in text and len(text) == 8 and text.isdigit():
        text = text[:4] + "-" + text[4:6] + "-" + text[6:] + "T00:00:00+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return TaskTimestamp(parsed)


def _known_value(field: str, raw: Any) -> tuple[FrozenValue, DecodeIssue | None]:
    text = str(raw).strip()
    try:
        if field == "uuid":
            return TaskUUID(text), None
        if field in {"prevLink", "nextLink"}:
            return ShortUUIDRef(text), None
        if field == "chainID":
            return ChainID(text), None
        if field == "link":
            if isinstance(raw, bool):
                raise ValueError("link must be a positive integer")
            number = float(raw)
            if not math.isfinite(number) or number <= 0 or not number.is_integer():
                raise ValueError("link must be a positive integer")
            return TaskLink(int(number)), None
        if field in {"due", "scheduled", "wait", "until", "entry", "modified", "end"}:
            return _timestamp(raw), None
        if field == "status":
            value = text.lower()
            issue = None if value in _KNOWN_STATUSES else DecodeIssue(field, "unknown_status", f"unknown Taskwarrior status: {text}", IssueSeverity.WARNING, _freeze(raw))
            return (TaskStatus(value) if issue is None else value), issue
        if field in {"chain", "anchor", "anchor_file", "anchor_mode", "cp", "chainMax", "chainUntil", "omit", "omit_file", "bc"}:
            return text, None
    except (TypeError, ValueError) as exc:
        return _freeze(raw), DecodeIssue(field, "invalid_value", str(exc), IssueSeverity.ERROR, _freeze(raw))
    return _freeze(raw), None


_KNOWN_FIELDS = frozenset(
    {
        "uuid", "status", "chainID", "link", "prevLink", "nextLink", "due", "scheduled", "wait", "until",
        "entry", "modified", "end", "chain", "anchor", "anchor_file", "anchor_mode", "cp", "chainMax",
        "chainUntil", "omit", "omit_file", "bc",
    }
)


@dataclass(frozen=True, slots=True, eq=False)
class TaskObservation:
    """Immutable, lossless interpretation of one Taskwarrior JSON row."""

    fields: Mapping[str, FieldState]
    arbitrary: Mapping[str, FrozenValue]
    issues: tuple[DecodeIssue, ...]
    provenance: ObservationProvenance
    _fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, ObservationProvenance):
            raise TypeError("task observation requires provenance")
        fields = MappingProxyType(dict(self.fields))
        arbitrary = MappingProxyType({str(key): _freeze(value) for key, value in self.arbitrary.items()})
        if any(not isinstance(key, str) or not isinstance(value, FieldState) for key, value in fields.items()):
            raise TypeError("task observation fields must be FieldState values")
        if any(not isinstance(issue, DecodeIssue) for issue in self.issues):
            raise TypeError("task observation issues must be DecodeIssue values")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "arbitrary", arbitrary)
        object.__setattr__(self, "issues", tuple(self.issues))

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, Any],
        *,
        source_query: str,
        snapshot_id: str = "",
        mutation_epoch: int = 0,
        command_count: int = 0,
    ) -> "TaskObservation":
        if not isinstance(row, Mapping):
            raise TypeError("task observation requires a JSON object")
        fields: dict[str, FieldState] = {}
        arbitrary: dict[str, FrozenValue] = {}
        issues: list[DecodeIssue] = []
        for key, raw in row.items():
            name = str(key)
            if name in _KNOWN_FIELDS:
                if raw is None:
                    fields[name] = FieldState(FieldPresence.NULL)
                else:
                    value, issue = _known_value(name, raw)
                    fields[name] = FieldState.from_raw(raw, value)
                    if issue is not None:
                        issues.append(issue)
            else:
                try:
                    arbitrary[name] = _freeze(raw)
                except TypeError as exc:
                    issues.append(DecodeIssue(name, "unsupported_value", str(exc), IssueSeverity.ERROR))
        for name in _KNOWN_FIELDS:
            fields.setdefault(name, FieldState.absent())
        provenance = ObservationProvenance(source_query, snapshot_id, mutation_epoch, command_count)
        semantic = {
            "fields": {
                key: _semantic_value(value.value)
                for key, value in sorted(fields.items())
                if key not in _SEMANTIC_EXCLUSIONS and value.presence is not FieldPresence.ABSENT
            },
            "arbitrary": {
                key: _thaw(value)
                for key, value in sorted(arbitrary.items())
                if key not in _SEMANTIC_EXCLUSIONS
            },
            "provenance": {
                "source_query": provenance.source_query,
                "snapshot_id": provenance.snapshot_id,
                "mutation_epoch": provenance.mutation_epoch,
            },
        }
        fingerprint = hashlib.sha256(_canonical_json(semantic).encode("utf-8")).hexdigest()[:24]
        return cls(fields, arbitrary, tuple(issues), provenance, fingerprint)

    def field(self, name: str) -> FieldState:
        return self.fields.get(str(name), FieldState.absent())

    @property
    def semantic_fingerprint(self) -> str:
        return self._fingerprint

    def to_mapping(self) -> dict[str, Any]:
        """Thaw a copy only at an explicit external serialization boundary."""
        result = {key: value.raw_value() for key, value in self.fields.items() if value.presence is not FieldPresence.ABSENT}
        result.update({key: _thaw(value) for key, value in self.arbitrary.items()})
        return result

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TaskObservation) and (
            self.semantic_fingerprint == other.semantic_fingerprint
            and self.issues == other.issues
            and self.provenance == other.provenance
        )


__all__ = (
    "ChainID",
    "DecodeIssue",
    "FieldPresence",
    "FieldState",
    "IssueSeverity",
    "ObservationProvenance",
    "ShortUUIDRef",
    "TaskStatus",
    "TaskLink",
    "TaskObservation",
    "TaskTimestamp",
    "TaskUUID",
)
