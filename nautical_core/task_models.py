"""Immutable Taskwarrior task observations and domain-facing value types.

This module deliberately does not decide whether a task is operationally valid.
It records what Taskwarrior returned, including malformed known fields, so later
validation and integrity tooling can make that decision with complete evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, TypeAlias
from uuid import UUID

try:
    from .task_field_policy import DRAFT_FORBIDDEN_FIELDS, draft_field_may_be_supplied
except ImportError:  # standalone thin-hook helper loading
    DRAFT_FORBIDDEN_FIELDS = frozenset(
        {"uuid", "chainID", "link", "prevLink", "id", "status", "modified", "end"}
    )

    def draft_field_may_be_supplied(field: str) -> bool:
        return str(field) not in DRAFT_FORBIDDEN_FIELDS




FrozenValue: TypeAlias = object
# Mutable Taskwarrior JSON at the hook protocol edge.
TaskPayload: TypeAlias = MutableMapping[str, Any]
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
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("non-finite task field value")
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
            return (TaskUUID(text) if _UUID_RE.fullmatch(text) else ShortUUIDRef(text)), None
        if field == "chainID":
            return ChainID(text), None
        if field == "link":
            if isinstance(raw, bool):
                raise ValueError("link must be a positive integer")
            number = float(raw)
            if not math.isfinite(number) or number <= 0 or not number.is_integer():
                raise ValueError("link must be a positive integer")
            return TaskLink(int(number)), None
        if field == "chainMax":
            number = float(raw)
            if not math.isfinite(number) or number <= 0 or not number.is_integer():
                raise ValueError("chainMax must be a positive integer")
            return int(number), None
        if field in {"due", "scheduled", "wait", "until", "entry", "modified", "end", "chainUntil"}:
            return _timestamp(raw), None
        if field == "status":
            value = text.lower()
            issue = None if value in _KNOWN_STATUSES else DecodeIssue(field, "unknown_status", f"unknown Taskwarrior status: {text}", IssueSeverity.WARNING, _freeze(raw))
            return (TaskStatus(value) if issue is None else value), issue
        if field in {"chain", "anchor", "anchor_file", "anchor_mode", "cp", "chainUntil", "omit", "omit_file", "bc", "description", "project"}:
            return text, None
    except (TypeError, ValueError) as exc:
        return _freeze(raw), DecodeIssue(field, "invalid_value", str(exc), IssueSeverity.ERROR, _freeze(raw))
    return _freeze(raw), None


_KNOWN_FIELDS = frozenset(
    {
        "uuid", "status", "chainID", "link", "prevLink", "nextLink", "due", "scheduled", "wait", "until",
        "entry", "modified", "end", "chain", "anchor", "anchor_file", "anchor_mode", "cp", "chainMax",
        "chainUntil", "omit", "omit_file", "bc", "description", "project",
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
    _projection_cache: dict[str, object] = field(default_factory=dict, init=False, repr=False, compare=False)

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
        object.__setattr__(self, "_projection_cache", {})

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

    def get(self, name: str, default: Any = None) -> Any:
        """Read a raw field value for typed consumers migrating from mappings."""
        state = self.field(name)
        return default if state.presence is FieldPresence.ABSENT else state.raw_value()

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

    def cached_projection(self, key: str) -> object | None:
        """Return an invocation-local validated projection, if already built."""
        return self._projection_cache.get(str(key))

    def cache_projection(self, key: str, value: object) -> object:
        """Store a projection without exposing mutable state as part of the model."""
        self._projection_cache[str(key)] = value
        return value

class RecurrenceKind(str, Enum):
    CP = "cp"
    ANCHOR = "anchor"


class ChainState(str, Enum):
    ENABLED = "on"
    DISABLED = "off"


class TaskOperation(str, Enum):
    SCHEDULE = "schedule"
    COMPLETION = "completion"
    EXPIRATION = "expiration"
    DELETION = "deletion"
    REPAIR = "repair"
    QUERY = "query"


@dataclass(frozen=True, slots=True)
class ChainIdentity:
    task_uuid: TaskUUID
    chain_id: ChainID
    link: TaskLink
    previous: TaskUUID | ShortUUIDRef | None = None
    next: TaskUUID | ShortUUIDRef | None = None
    state: ChainState = ChainState.ENABLED

    def __post_init__(self) -> None:
        if not isinstance(self.task_uuid, TaskUUID):
            raise TypeError("chain identity requires a full task UUID")
        if not isinstance(self.chain_id, ChainID) or not isinstance(self.link, TaskLink):
            raise TypeError("chain identity requires typed chainID and link")
        object.__setattr__(self, "state", ChainState(self.state))
        for name in ("previous", "next"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, (TaskUUID, ShortUUIDRef)):
                raise TypeError(f"chain identity {name} reference is invalid")


@dataclass(frozen=True, slots=True)
class TemporalState:
    due: TaskTimestamp | None = None
    scheduled: TaskTimestamp | None = None
    wait: TaskTimestamp | None = None
    until: TaskTimestamp | None = None
    entry: TaskTimestamp | None = None
    modified: TaskTimestamp | None = None
    end: TaskTimestamp | None = None
    presence: Mapping[str, FieldPresence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = (self.due, self.scheduled, self.wait, self.until, self.entry, self.modified, self.end)
        if any(value is not None and not isinstance(value, TaskTimestamp) for value in values):
            raise TypeError("temporal state values must be typed timestamps")
        object.__setattr__(self, "presence", MappingProxyType(dict(self.presence)))

    def reference(self) -> TaskTimestamp | None:
        return self.due or self.scheduled


@dataclass(frozen=True, slots=True)
class RecurrenceState:
    kind: RecurrenceKind
    spec: Any
    anchor_mode: str
    chain_max: int | None
    chain_until: TaskTimestamp | None
    business_calendar: str = ""
    omit: str = ""
    omit_file: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", RecurrenceKind(self.kind))
        mode = str(self.anchor_mode).strip().lower()
        if mode not in {"skip", "all", "flex"}:
            raise ValueError("anchor_mode must be skip, all, or flex")
        object.__setattr__(self, "anchor_mode", mode)
        if self.chain_max is not None and (isinstance(self.chain_max, bool) or self.chain_max <= 0):
            raise ValueError("chainMax must be positive")
        if self.chain_until is not None and not isinstance(self.chain_until, TaskTimestamp):
            raise TypeError("chainUntil must be a typed timestamp")


@dataclass(frozen=True, slots=True)
class NauticalTask:
    observation: TaskObservation
    identity: ChainIdentity
    status: TaskStatus
    temporal: TemporalState
    recurrence: RecurrenceState
    description: str = ""

    @classmethod
    def from_observation(cls, observation: TaskObservation) -> "NauticalTask":
        if not isinstance(observation, TaskObservation):
            raise TypeError("NauticalTask requires a TaskObservation")
        cached = observation.cached_projection("nautical_task")
        if isinstance(cached, cls):
            return cached
        errors = [issue for issue in observation.issues if issue.severity is IssueSeverity.ERROR]
        uuid = observation.field("uuid").value
        status = observation.field("status").value
        chain_id = observation.field("chainID").value
        link = observation.field("link").value
        if not isinstance(uuid, TaskUUID):
            errors.append(DecodeIssue("uuid", "missing_or_invalid", "a full UUID is required"))
        if not isinstance(status, TaskStatus):
            errors.append(DecodeIssue("status", "missing_or_invalid", "a recognized status is required"))
        if not isinstance(chain_id, ChainID):
            errors.append(DecodeIssue("chainID", "missing_or_invalid", "a chainID is required"))
        if not isinstance(link, TaskLink):
            errors.append(DecodeIssue("link", "missing_or_invalid", "a positive link is required"))
        if errors:
            raise ValueError("; ".join(f"{issue.field}: {issue.message}" for issue in errors))
        state = ChainState.ENABLED if str(observation.field("chain").value or "on").lower() == "on" else ChainState.DISABLED
        identity = ChainIdentity(uuid, chain_id, link, observation.field("prevLink").value, observation.field("nextLink").value, state)
        temporal_values: dict[str, TaskTimestamp | None] = {}
        presence: dict[str, FieldPresence] = {}
        for name in ("due", "scheduled", "wait", "until", "entry", "modified", "end"):
            field = observation.field(name)
            presence[name] = field.presence
            temporal_values[name] = field.value if isinstance(field.value, TaskTimestamp) else None
        temporal = TemporalState(**temporal_values, presence=presence)
        anchor = observation.field("anchor").value
        anchor_file = observation.field("anchor_file").value
        cp = observation.field("cp").value
        if bool(cp) and (bool(anchor) or bool(anchor_file)):
            raise ValueError("recurrence cannot contain both cp and anchor fields")
        if not cp and not anchor and not anchor_file:
            raise ValueError("recurrence requires cp, anchor, or anchor_file")
        kind = RecurrenceKind.CP if cp else RecurrenceKind.ANCHOR
        mode = observation.field("anchor_mode").value or "skip"
        chain_max = observation.field("chainMax").value
        if chain_max is not None and not isinstance(chain_max, int):
            raise ValueError("chainMax is malformed")
        chain_until = observation.field("chainUntil").value
        from .recurrence_context import RecurrenceContext
        from .recurrence_spec import RecurrenceSpec

        spec = RecurrenceSpec(
            context=RecurrenceContext(chain_id=chain_id.value),
            anchor=str(anchor or ""), anchor_file=str(anchor_file or ""),
            omit=str(observation.field("omit").value or ""),
            omit_file=str(observation.field("omit_file").value or ""), cp=str(cp or ""),
            anchor_mode=str(mode), chain_max=chain_max,
            chain_until=chain_until.value.isoformat().replace("+00:00", "Z") if isinstance(chain_until, TaskTimestamp) else "",
        )
        recurrence = RecurrenceState(
            kind, spec, str(mode), chain_max,
            chain_until if isinstance(chain_until, TaskTimestamp) else None,
            str(observation.field("bc").value or ""),
            str(observation.field("omit").value or ""),
            str(observation.field("omit_file").value or ""),
        )
        return observation.cache_projection(
            "nautical_task",
            cls(observation, identity, status, temporal, recurrence, str(observation.field("description").value or "")),
        )


@dataclass(frozen=True, slots=True)
class TaskDraft:
    """A complete, immutable child task intent before Taskwarrior encoding."""

    identity: ChainIdentity
    description: str
    recurrence: RecurrenceState
    target: TaskTimestamp
    fields: Mapping[str, FrozenValue] = field(default_factory=dict)
    target_field: str = "due"

    @classmethod
    def from_task(cls, task: NauticalTask, *, target_field: str | None = None) -> "TaskDraft":
        """Project a validated task into a complete draft-shaped payload."""
        if not isinstance(task, NauticalTask):
            raise TypeError("task draft requires a validated NauticalTask")
        field = target_field or ("due" if task.temporal.due is not None else "scheduled")
        target = task.temporal.due if field == "due" else task.temporal.scheduled
        if target is None:
            raise ValueError("task draft requires a due or scheduled target")
        excluded = {
            "id", "uuid", "status", "modified", "end", "chainID", "link", "prevLink", "nextLink",
            "description", "chain", "anchor", "anchor_file", "anchor_mode", "cp", "omit", "omit_file",
            "bc", "chainMax", "chainUntil",
        }
        # Keep the non-target temporal field so relative scheduled/wait carries
        # survive a due-target child (and vice versa).
        excluded.add(field)
        values = {
            key: state.raw_value()
            for key, state in task.observation.fields.items()
            if state.presence is FieldPresence.VALUE
        }
        values.update({key: _thaw(value) for key, value in task.observation.arbitrary.items()})
        return cls(
            identity=task.identity,
            description=task.description,
            recurrence=task.recurrence,
            target=target,
            fields={key: value for key, value in values.items() if key not in excluded},
            target_field=field,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ChainIdentity):
            raise TypeError("task draft requires chain identity")
        if not isinstance(self.recurrence, RecurrenceState):
            raise TypeError("task draft requires recurrence state")
        if not isinstance(self.target, TaskTimestamp):
            raise TypeError("task draft requires a typed target timestamp")
        description = str(self.description).strip()
        if not description:
            raise ValueError("task draft requires a description")
        target_field = str(self.target_field).strip().lower()
        if target_field not in {"due", "scheduled"}:
            raise ValueError("task draft target field must be due or scheduled")
        copied = {str(key): _freeze(value) for key, value in self.fields.items()}
        overlap = {key for key in copied if not draft_field_may_be_supplied(key)}
        if overlap:
            raise ValueError(f"task draft cannot supply generated, identity, or policy-owned fields: {', '.join(sorted(overlap))}")
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "target_field", target_field)
        object.__setattr__(self, "fields", MappingProxyType(copied))

    @property
    def fingerprint(self) -> str:
        from hashlib import sha256
        payload = self.to_mapping()
        return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:24]

    def to_mapping(self) -> dict[str, Any]:
        recurrence_spec = self.recurrence.spec
        result = {key: _thaw(value) for key, value in self.fields.items()}
        result.update(
            {
                "uuid": self.identity.task_uuid.value,
                "description": self.description,
                "chain": self.identity.state.value,
                "chainID": self.identity.chain_id.value,
                "link": self.identity.link.value,
                "status": "pending",
                "prevLink": str(self.identity.previous) if self.identity.previous is not None else None,
                self.target_field: self.target.value.isoformat().replace("+00:00", "Z"),
                "anchor": str(getattr(recurrence_spec, "anchor", "") or ""),
                "anchor_file": str(getattr(recurrence_spec, "anchor_file", "") or ""),
                "omit": self.recurrence.omit,
                "omit_file": self.recurrence.omit_file,
                "cp": str(getattr(recurrence_spec, "cp", "") or ""),
                "anchor_mode": self.recurrence.anchor_mode,
            }
        )
        if self.recurrence.chain_max is not None:
            result["chainMax"] = self.recurrence.chain_max
        if self.recurrence.chain_until is not None:
            result["chainUntil"] = self.recurrence.chain_until.value.isoformat().replace("+00:00", "Z")
        if self.recurrence.business_calendar:
            result["bc"] = self.recurrence.business_calendar
        return result

    def field_value(self, name: str, default: Any = None) -> Any:
        """Read an optional carried field without serializing the full draft."""
        key = str(name)
        if key == "description":
            return self.description
        if key == self.target_field:
            return self.target.value.isoformat().replace("+00:00", "Z")
        value = self.fields.get(key)
        return default if value is None else _thaw(value)


@dataclass(frozen=True, slots=True)
class ValidatedTask:
    operation: TaskOperation
    task: NauticalTask


@dataclass(frozen=True, slots=True)
class InvalidTask:
    operation: TaskOperation
    observation: TaskObservation
    issues: tuple[DecodeIssue, ...]


TaskValidation = ValidatedTask | InvalidTask


def validate_task(observation: TaskObservation, operation: TaskOperation) -> TaskValidation:
    """Validate one observation for an operation without hiding its evidence."""
    operation = TaskOperation(operation)
    try:
        task = NauticalTask.from_observation(observation)
    except (TypeError, ValueError) as exc:
        return InvalidTask(operation, observation, (DecodeIssue("task", f"invalid_for_{operation.value}", str(exc)),))
    if operation in {TaskOperation.SCHEDULE, TaskOperation.COMPLETION} and task.temporal.reference() is None:
        issue = DecodeIssue("due", "missing_reference", f"{operation.value} requires due or scheduled")
        return InvalidTask(operation, observation, (issue,))
    return ValidatedTask(operation, task)


__all__ = (
    "ChainIdentity",
    "ChainState",
    "ChainID",
    "DecodeIssue",
    "FieldPresence",
    "FieldState",
    "IssueSeverity",
    "InvalidTask",
    "NauticalTask",
    "ObservationProvenance",
    "RecurrenceKind",
    "RecurrenceState",
    "ShortUUIDRef",
    "TaskStatus",
    "TaskLink",
    "TaskObservation",
    "TaskDraft",
    "TaskOperation",
    "TaskTimestamp",
    "TaskUUID",
    "TaskValidation",
    "TemporalState",
    "ValidatedTask",
    "validate_task",
)
