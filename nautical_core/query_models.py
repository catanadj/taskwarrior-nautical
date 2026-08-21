"""Versioned, read-only models for Nautical's local query contract.

The models in this module deliberately contain no Taskwarrior, scheduler, or
presentation logic.  They validate the public request/response shape so the
CLI and the query service share one contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Literal, Mapping


QUERY_API_VERSION = 1
OCCURRENCES_SCHEMA = "nautical.query.occurrences"
NEXT_SCHEMA = "nautical.query.next"
CAPABILITIES_SCHEMA = "nautical.query.capabilities"
OCCURRENCE_OPERATION = "occurrences"
NEXT_OPERATION = "next"

DEFAULT_MAX_TASKS = 100
DEFAULT_MAX_OCCURRENCES = 1000
DEFAULT_MAX_TOTAL_OCCURRENCES = 10000
DEFAULT_MAX_ITERATIONS = 512
DEFAULT_MAX_FILE_SKIPS = 512
HARD_MAX_TASKS = 1000
HARD_MAX_OCCURRENCES = 10000
HARD_MAX_TOTAL_OCCURRENCES = 100000
HARD_MAX_ITERATIONS = 10000
HARD_MAX_FILE_SKIPS = 10000

QueryStatus = Literal["found", "empty", "exhausted", "absent", "unavailable", "invalid"]
OmissionPolicy = Literal["exclude", "include", "report"]


class QueryContractError(ValueError):
    """Raised when a public query contract value is invalid."""


def _text(value: object, field: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise QueryContractError(f"{field} is required")
    return text


def _positive_int(value: object, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, str))
        or (isinstance(value, float) and not value.is_integer())
    ):
        raise QueryContractError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise QueryContractError(f"{field} must be a positive integer") from exc
    if result <= 0 or result > maximum:
        raise QueryContractError(f"{field} must be between 1 and {maximum}")
    return result


def _bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise QueryContractError(f"{field} must be boolean")
    return value


def _parse_boundary(value: object, field: str) -> "QueryBoundary":
    if isinstance(value, QueryBoundary):
        return value
    text = _text(value, field)
    try:
        if "T" not in text and "t" not in text:
            return QueryBoundary(date.fromisoformat(text), date_only=True)
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise QueryContractError(
            f"{field} must be an RFC 3339 timestamp with offset or an ISO date"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise QueryContractError(f"{field} timestamp must include an explicit timezone offset")
    return QueryBoundary(parsed, date_only=False)


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


@dataclass(frozen=True, slots=True)
class QueryBoundary:
    """A date or timezone-aware timestamp supplied by a query consumer."""

    value: date | datetime
    date_only: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.value, datetime):
            if self.value.tzinfo is None or self.value.utcoffset() is None:
                raise QueryContractError("query timestamp must include an explicit timezone offset")
            if self.date_only:
                raise QueryContractError("datetime query boundary cannot be date-only")
        elif not isinstance(self.value, date) or not self.date_only:
            raise QueryContractError("query boundary must be a date-only value or aware timestamp")

    def to_text(self) -> str:
        if self.date_only:
            return self.value.isoformat()
        assert isinstance(self.value, datetime)
        return self.value.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class QuerySelector:
    """One explicit task selection mode."""

    uuids: tuple[str, ...] = ()
    chain_id: str = ""
    all_tasks: bool = False

    def __post_init__(self) -> None:
        uuids = tuple(dict.fromkeys(_text(item, "task UUID").lower() for item in self.uuids))
        chain_id = _text(self.chain_id, "chainID", required=False).lower()
        if not isinstance(self.all_tasks, bool):
            raise QueryContractError("all_tasks must be boolean")
        modes = bool(uuids) + bool(chain_id) + self.all_tasks
        if modes != 1:
            raise QueryContractError("query selector requires exactly one of UUIDs, chainID, or all_tasks")
        if len(uuids) > HARD_MAX_TASKS:
            raise QueryContractError(f"query selector cannot contain more than {HARD_MAX_TASKS} UUIDs")
        object.__setattr__(self, "uuids", uuids)
        object.__setattr__(self, "chain_id", chain_id)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QuerySelector":
        if not isinstance(value, Mapping):
            raise QueryContractError("query selector must be an object")
        raw_uuids = value.get("uuids", ())
        if isinstance(raw_uuids, str):
            raw_uuids = (raw_uuids,)
        if not isinstance(raw_uuids, (list, tuple)):
            raise QueryContractError("query selector uuids must be a list")
        return cls(
            uuids=tuple(str(item) for item in raw_uuids),
            chain_id=str(value.get("chain_id", value.get("chainID", "")) or ""),
            all_tasks=_bool(value.get("all_tasks", False), "all_tasks"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "uuids": list(self.uuids),
            "chainID": self.chain_id or None,
            "all_tasks": self.all_tasks,
        }


@dataclass(frozen=True, slots=True)
class OccurrenceQueryRequest:
    """Validated request for a bounded schedule occurrence collection."""

    selector: QuerySelector
    start: QueryBoundary
    end: QueryBoundary | None = None
    count: int | None = None
    start_inclusive: bool = True
    omission_policy: OmissionPolicy = "exclude"
    max_tasks: int = DEFAULT_MAX_TASKS
    max_occurrences: int = DEFAULT_MAX_OCCURRENCES
    max_total_occurrences: int = DEFAULT_MAX_TOTAL_OCCURRENCES
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_file_skips: int = DEFAULT_MAX_FILE_SKIPS
    version: int = QUERY_API_VERSION
    operation: str = OCCURRENCE_OPERATION

    def __post_init__(self) -> None:
        if not isinstance(self.selector, QuerySelector):
            raise QueryContractError("occurrence request requires a query selector")
        if not isinstance(self.start, QueryBoundary):
            raise QueryContractError("occurrence request requires a start boundary")
        if self.end is not None and not isinstance(self.end, QueryBoundary):
            raise QueryContractError("query end must be a query boundary")
        if self.end is not None:
            start_value = self.start.value
            end_value = self.end.value
            if type(start_value) is not type(end_value):
                raise QueryContractError("query boundaries must both be dates or both be timestamps")
            if end_value < start_value:
                raise QueryContractError("query end must not precede query start")
        if self.end is None and self.count is None:
            raise QueryContractError("occurrence query requires an end boundary or count")
        count = None if self.count is None else _positive_int(self.count, "count", HARD_MAX_OCCURRENCES)
        start_inclusive = _bool(self.start_inclusive, "start_inclusive")
        if self.omission_policy not in {"exclude", "include", "report"}:
            raise QueryContractError("omission_policy must be exclude, include, or report")
        normalized_limits: dict[str, int] = {}
        for field, value, maximum in (
            ("max_tasks", self.max_tasks, HARD_MAX_TASKS),
            ("max_occurrences", self.max_occurrences, HARD_MAX_OCCURRENCES),
            ("max_total_occurrences", self.max_total_occurrences, HARD_MAX_TOTAL_OCCURRENCES),
            ("max_iterations", self.max_iterations, HARD_MAX_ITERATIONS),
            ("max_file_skips", self.max_file_skips, HARD_MAX_FILE_SKIPS),
        ):
            normalized_limits[field] = _positive_int(value, field, maximum)
        if isinstance(self.version, bool) or self.version != QUERY_API_VERSION:
            raise QueryContractError(f"unsupported query API version: {self.version!r}")
        if self.operation not in {OCCURRENCE_OPERATION, NEXT_OPERATION}:
            raise QueryContractError(f"unsupported query operation: {self.operation!r}")
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "start_inclusive", start_inclusive)
        for field, normalized in normalized_limits.items():
            object.__setattr__(self, field, normalized)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OccurrenceQueryRequest":
        if not isinstance(value, Mapping):
            raise QueryContractError("occurrence request must be an object")
        return cls(
            selector=QuerySelector.from_mapping(value.get("selector", {})),
            start=_parse_boundary(value.get("from", value.get("start")), "from"),
            end=(
                _parse_boundary(value["to"], "to")
                if value.get("to") is not None
                else None
            ),
            count=value.get("count"),
            start_inclusive=_bool(value.get("start_inclusive", True), "start_inclusive"),
            omission_policy=str(value.get("omission_policy", "exclude")),
            max_tasks=value.get("max_tasks", DEFAULT_MAX_TASKS),
            max_occurrences=value.get("max_occurrences", DEFAULT_MAX_OCCURRENCES),
            max_total_occurrences=value.get("max_total_occurrences", DEFAULT_MAX_TOTAL_OCCURRENCES),
            max_iterations=value.get("max_iterations", DEFAULT_MAX_ITERATIONS),
            max_file_skips=value.get("max_file_skips", DEFAULT_MAX_FILE_SKIPS),
            version=value.get("version", QUERY_API_VERSION),
            operation=str(value.get("operation", OCCURRENCE_OPERATION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "operation": self.operation,
            "selector": self.selector.to_dict(),
            "from": self.start.to_text(),
            "to": self.end.to_text() if self.end is not None else None,
            "count": self.count,
            "start_inclusive": self.start_inclusive,
            "omission_policy": self.omission_policy,
            "max_tasks": self.max_tasks,
            "max_occurrences": self.max_occurrences,
            "max_total_occurrences": self.max_total_occurrences,
            "max_iterations": self.max_iterations,
            "max_file_skips": self.max_file_skips,
        }


@dataclass(frozen=True, slots=True)
class QueryFailure:
    """Stable machine-readable failure evidence."""

    code: str
    message: str
    retryable: bool = False
    task_uuid: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _text(self.code, "failure code"))
        object.__setattr__(self, "message", _text(self.message, "failure message"))
        if not isinstance(self.retryable, bool):
            raise QueryContractError("failure retryable must be boolean")
        if self.task_uuid is not None:
            object.__setattr__(self, "task_uuid", _text(self.task_uuid, "failure task UUID"))
        if not isinstance(self.details, Mapping):
            raise QueryContractError("failure details must be an object")
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "task_uuid": self.task_uuid,
            "details": _json_value(self.details),
        }


@dataclass(frozen=True, slots=True)
class TaskIdentity:
    """Documented task identity and opaque recurrence metadata."""

    uuid: str
    chain_id: str
    link: int | None = None
    description: str = ""
    recurrence_kind: str = ""
    expression: str = ""
    schedule_fingerprint: str = ""
    current_due: str = ""
    current_scheduled: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "uuid", _text(self.uuid, "task UUID"))
        object.__setattr__(self, "chain_id", _text(self.chain_id, "task chainID"))
        if self.link is not None and (isinstance(self.link, bool) or not isinstance(self.link, int) or self.link < 0):
            raise QueryContractError("task link must be a non-negative integer")
        for field in (
            "description",
            "recurrence_kind",
            "expression",
            "schedule_fingerprint",
            "current_due",
            "current_scheduled",
        ):
            object.__setattr__(self, field, str(getattr(self, field) or ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "chainID": self.chain_id,
            "link": self.link,
            "description": self.description,
            "recurrence_kind": self.recurrence_kind,
            "expression": self.expression,
            "schedule_fingerprint": self.schedule_fingerprint,
            "current_due": self.current_due or None,
            "current_scheduled": self.current_scheduled or None,
        }


@dataclass(frozen=True, slots=True)
class OccurrenceRecord:
    """One ordered local/UTC occurrence returned to an external consumer."""

    local: datetime
    utc: datetime
    timezone: str
    source: str
    description: str = ""
    omitted: bool = False
    omission_reason: str = ""

    def __post_init__(self) -> None:
        for field in ("local", "utc"):
            value = getattr(self, field)
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise QueryContractError(f"occurrence {field} must be timezone-aware")
        if self.utc.astimezone(timezone.utc) != self.local.astimezone(timezone.utc):
            raise QueryContractError("occurrence local and UTC timestamps identify different instants")
        object.__setattr__(self, "timezone", _text(self.timezone, "occurrence timezone"))
        object.__setattr__(self, "source", _text(self.source, "occurrence source"))
        if not isinstance(self.omitted, bool):
            raise QueryContractError("occurrence omitted must be boolean")
        object.__setattr__(self, "description", str(self.description or ""))
        object.__setattr__(self, "omission_reason", str(self.omission_reason or ""))

    def to_dict(self) -> dict[str, Any]:
        offset = self.local.utcoffset()
        return {
            "local": self.local.isoformat(),
            "utc": self.utc.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "timezone": self.timezone,
            "utc_offset_minutes": int(offset.total_seconds() // 60) if offset is not None else None,
            "fold": self.local.fold,
            "source": self.source,
            "description": self.description,
            "omitted": self.omitted,
            "omission_reason": self.omission_reason or None,
        }


@dataclass(frozen=True, slots=True)
class TaskOccurrenceResult:
    """Occurrences and status for one resolved task."""

    task: TaskIdentity | None
    status: QueryStatus
    occurrences: tuple[OccurrenceRecord, ...] = ()
    omitted_occurrences: tuple[OccurrenceRecord, ...] = ()
    failure: QueryFailure | None = None
    terminal: Mapping[str, Any] | None = None
    chain: Mapping[str, Any] = field(default_factory=dict)
    lifecycle: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task is not None and not isinstance(self.task, TaskIdentity):
            raise QueryContractError("task occurrence result identity is invalid")
        if self.status not in {"found", "empty", "exhausted", "absent", "unavailable", "invalid"}:
            raise QueryContractError(f"invalid query status: {self.status!r}")
        if any(not isinstance(item, OccurrenceRecord) for item in (*self.occurrences, *self.omitted_occurrences)):
            raise QueryContractError("task occurrence result contains an invalid occurrence")
        if self.failure is not None and not isinstance(self.failure, QueryFailure):
            raise QueryContractError("task occurrence result failure is invalid")
        if self.terminal is not None and not isinstance(self.terminal, Mapping):
            raise QueryContractError("task occurrence terminal evidence must be an object")
        if not isinstance(self.chain, Mapping) or not isinstance(self.lifecycle, Mapping):
            raise QueryContractError("task occurrence metadata must be objects")
        object.__setattr__(self, "chain", dict(self.chain))
        object.__setattr__(self, "lifecycle", dict(self.lifecycle))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict() if self.task is not None else None,
            "status": self.status,
            "occurrences": [item.to_dict() for item in self.occurrences],
            "omitted_occurrences": [item.to_dict() for item in self.omitted_occurrences],
            "failure": self.failure.to_dict() if self.failure is not None else None,
            "terminal": _json_value(self.terminal) if self.terminal is not None else None,
            "chain": _json_value(self.chain),
            "lifecycle": _json_value(self.lifecycle),
        }


@dataclass(frozen=True, slots=True)
class OccurrenceQueryResponse:
    """Versioned response envelope emitted by the local query CLI."""

    request: OccurrenceQueryRequest
    timezone: str
    results: tuple[TaskOccurrenceResult, ...] = ()
    status: QueryStatus = "empty"
    configuration_fingerprint: str = ""
    failure: QueryFailure | None = None
    schema: str = OCCURRENCES_SCHEMA
    version: int = QUERY_API_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.request, OccurrenceQueryRequest):
            raise QueryContractError("query response requires its request")
        object.__setattr__(self, "timezone", _text(self.timezone, "response timezone"))
        if self.status not in {"found", "empty", "exhausted", "absent", "unavailable", "invalid"}:
            raise QueryContractError(f"invalid query response status: {self.status!r}")
        if any(not isinstance(item, TaskOccurrenceResult) for item in self.results):
            raise QueryContractError("query response contains an invalid task result")
        if self.failure is not None and not isinstance(self.failure, QueryFailure):
            raise QueryContractError("query response failure is invalid")
        expected_schema = OCCURRENCES_SCHEMA if self.request.operation == OCCURRENCE_OPERATION else NEXT_SCHEMA
        if self.schema == OCCURRENCES_SCHEMA and expected_schema == NEXT_SCHEMA:
            object.__setattr__(self, "schema", NEXT_SCHEMA)
        if self.schema != expected_schema or self.version != QUERY_API_VERSION:
            raise QueryContractError("unsupported occurrence response schema")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "operation": self.request.operation,
            "status": self.status,
            "basis": "schedule" if self.request.operation == OCCURRENCE_OPERATION else "next",
            "timezone": self.timezone,
            "query": self.request.to_dict(),
            "configuration_fingerprint": self.configuration_fingerprint or None,
            "results": [item.to_dict() for item in self.results],
            "failure": self.failure.to_dict() if self.failure is not None else None,
        }


__all__ = (
    "CAPABILITIES_SCHEMA",
    "DEFAULT_MAX_FILE_SKIPS",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_OCCURRENCES",
    "DEFAULT_MAX_TOTAL_OCCURRENCES",
    "DEFAULT_MAX_TASKS",
    "HARD_MAX_FILE_SKIPS",
    "HARD_MAX_ITERATIONS",
    "HARD_MAX_OCCURRENCES",
    "HARD_MAX_TOTAL_OCCURRENCES",
    "HARD_MAX_TASKS",
    "OCCURRENCES_SCHEMA",
    "NEXT_SCHEMA",
    "NEXT_OPERATION",
    "OccurrenceQueryRequest",
    "OccurrenceQueryResponse",
    "OccurrenceRecord",
    "QueryBoundary",
    "QueryContractError",
    "QueryFailure",
    "QuerySelector",
    "TaskIdentity",
    "TaskOccurrenceResult",
)
