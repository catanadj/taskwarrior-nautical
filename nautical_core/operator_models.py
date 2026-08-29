"""Versioned contracts shared by Nautical's operator control plane.

This module is deliberately domain-neutral.  It describes what an operator
requested and what the control plane observed; established scheduler,
lifecycle, Taskwarrior, and integrity services remain the owners of decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Mapping, cast


OPERATOR_API_VERSION = 1
OPERATOR_RESULT_VERSION = 2


class OperatorContractError(ValueError):
    """Raised when an operator request or result violates its contract."""


class OperatorOperation(str, Enum):
    CAPABILITIES = "capabilities"
    INSPECT = "inspect"
    HEALTH = "health"
    OCCURRENCES = "occurrences"
    CHAIN = "chain"
    INTEGRITY = "integrity"
    LIFECYCLE = "lifecycle"
    QUEUE = "queue"
    DIAGNOSE = "diagnose"
    PLAN = "plan"
    APPLY = "apply"
    VERIFY = "verify"
    HOUSEKEEPING = "housekeeping"


class OperatorPhase(str, Enum):
    """Typed phases used by the internal operator control-plane pipeline."""

    VALIDATE_REQUEST = "validate_request"
    CAPTURE_CONTEXT = "capture_context"
    COMPILE_SCOPE = "compile_scope"
    ACQUIRE_SNAPSHOT = "acquire_snapshot"
    INSPECT = "inspect"
    PLAN = "plan"
    AUTHORIZE = "authorize"
    APPLY = "apply"
    REFRESH = "refresh"
    VERIFY = "verify"
    RESULT = "result"


@dataclass(frozen=True, slots=True)
class OperatorPhaseResult:
    """One immutable phase outcome; failures stop later phases."""

    phase: OperatorPhase
    value: Any = None
    failure: "OperatorFailure | None" = None

    def __post_init__(self) -> None:
        try:
            phase = OperatorPhase(self.phase)
        except (TypeError, ValueError) as exc:
            raise OperatorContractError("invalid operator phase") from exc
        if self.failure is not None and not isinstance(self.failure, OperatorFailure):
            raise OperatorContractError("operator phase failure must be OperatorFailure")
        if self.failure is not None and self.value is not None:
            raise OperatorContractError("failed operator phase cannot contain a value")
        if self.failure is None and self.value is None:
            raise OperatorContractError("successful operator phase requires a value")
        object.__setattr__(self, "phase", phase)


class OperatorStatus(str, Enum):
    OK = "ok"
    ATTENTION = "attention"
    WARN = "warn"
    DEGRADED = "degraded"
    REPAIRABLE = "repairable"
    DEFERRED = "deferred"
    MANUAL_REVIEW = "manual_review"
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"
    ERROR = "error"


class OperatorV2Status(str, Enum):
    """Unified status vocabulary for the public operator contract."""

    OK = "ok"
    FOUND = "found"
    EMPTY = "empty"
    ABSENT = "absent"
    EXHAUSTED = "exhausted"
    ATTENTION = "attention"
    REPAIRABLE = "repairable"
    DEFERRED = "deferred"
    MANUAL_REVIEW = "manual_review"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"
    ERROR = "error"


def exit_code_for_v2_status(status: OperatorV2Status | str) -> OperatorExitCode:
    """Map v2 statuses to the stable operator process contract."""
    try:
        normalized = OperatorV2Status(status)
    except (TypeError, ValueError) as exc:
        raise OperatorContractError("invalid operator v2 status") from exc
    if normalized in {OperatorV2Status.OK, OperatorV2Status.FOUND, OperatorV2Status.EMPTY, OperatorV2Status.ABSENT}:
        return OperatorExitCode.SUCCESS
    if normalized is OperatorV2Status.EXHAUSTED:
        return OperatorExitCode.PARTIAL
    if normalized in {OperatorV2Status.ATTENTION, OperatorV2Status.REPAIRABLE, OperatorV2Status.DEFERRED}:
        return OperatorExitCode.FINDINGS
    if normalized is OperatorV2Status.INVALID:
        return OperatorExitCode.INVALID_REQUEST
    if normalized is OperatorV2Status.UNAVAILABLE:
        return OperatorExitCode.UNAVAILABLE
    if normalized is OperatorV2Status.PARTIAL:
        return OperatorExitCode.PARTIAL
    if normalized is OperatorV2Status.MANUAL_REVIEW:
        return OperatorExitCode.MANUAL_REVIEW
    return OperatorExitCode.INTERNAL_FAILURE


class OperatorExitCode(IntEnum):
    """Stable process outcomes for operator composition roots."""

    SUCCESS = 0
    FINDINGS = 1
    INVALID_REQUEST = 2
    UNAVAILABLE = 3
    PARTIAL = 4
    MANUAL_REVIEW = 5
    INTERNAL_FAILURE = 6


def exit_code_for_status(status: OperatorStatus | str) -> OperatorExitCode:
    """Map a typed result status to the control-plane process contract."""

    try:
        normalized = OperatorStatus(status)
    except (TypeError, ValueError) as exc:
        raise OperatorContractError("invalid operator status") from exc
    if normalized is OperatorStatus.OK:
        return OperatorExitCode.SUCCESS
    if normalized in {OperatorStatus.ATTENTION, OperatorStatus.REPAIRABLE, OperatorStatus.DEFERRED}:
        return OperatorExitCode.FINDINGS
    if normalized is OperatorStatus.UNAVAILABLE:
        return OperatorExitCode.UNAVAILABLE
    if normalized is OperatorStatus.PARTIAL:
        return OperatorExitCode.PARTIAL
    if normalized is OperatorStatus.MANUAL_REVIEW:
        return OperatorExitCode.MANUAL_REVIEW
    return OperatorExitCode.INTERNAL_FAILURE


class OperatorScopeKind(str, Enum):
    SYSTEM = "system"
    ACTIVE_TASKS = "active_tasks"
    CHAIN = "chain"
    CHAINS = "chains"
    UUID = "uuid"
    UUIDS = "uuids"
    LIFECYCLE_CANDIDATES = "lifecycle_candidates"
    INTEGRITY_CANDIDATES = "integrity_candidates"
    TEMPORAL_RANGE = "temporal_range"
    CURSOR = "cursor"


class CoverageKind(str, Enum):
    COMPLETE = "complete"
    BOUNDED = "bounded"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


_COVERAGE_RANK = {
    CoverageKind.UNAVAILABLE: 0,
    CoverageKind.PARTIAL: 1,
    CoverageKind.BOUNDED: 2,
    CoverageKind.COMPLETE: 3,
}


def _text(value: object, field_name: str, *, required: bool = False) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise OperatorContractError(f"{field_name} is required")
    return text


def _positive(value: object, field_name: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise OperatorContractError(f"{field_name} must be a positive integer")
    try:
        result = int(value) if isinstance(value, (int, str, bytes, bytearray)) else int(str(value))
    except (TypeError, ValueError) as exc:
        raise OperatorContractError(f"{field_name} must be a positive integer") from exc
    if result < 1 or result > maximum:
        raise OperatorContractError(f"{field_name} must be between 1 and {maximum}")
    return result


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OperatorContractError(f"{field_name} must be an object")
    return value


def _json_value(value: object, _seen: set[int] | None = None) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    seen = _seen if _seen is not None else set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in seen:
            raise OperatorContractError("cyclic JSON value is not supported")
        seen.add(marker)
        try:
            return {str(key): _json_value(item, seen) for key, item in value.items()}
        finally:
            seen.remove(marker)
    if isinstance(value, (list, tuple, set, frozenset)):
        marker = id(value)
        if marker in seen:
            raise OperatorContractError("cyclic JSON value is not supported")
        seen.add(marker)
        try:
            values = [_json_value(item, seen) for item in value]
        finally:
            seen.remove(marker)
        if isinstance(value, (set, frozenset)):
            return sorted(values, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
        return values
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict(), seen)
    raise OperatorContractError(f"value of type {type(value).__name__} is not JSON-native")


def _freeze_json_value(value: object) -> object:
    """Validate JSON-native data and recursively detach mutable containers."""
    _json_value(value)
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple, set, frozenset)):
        frozen = tuple(_freeze_json_value(item) for item in value)
        if isinstance(value, (set, frozenset)):
            return tuple(sorted(frozen, key=lambda item: json.dumps(_json_value(item), sort_keys=True, ensure_ascii=False, separators=(",", ":"))))
        return frozen
    if isinstance(value, Enum):
        return value.value
    return value


@dataclass(frozen=True, slots=True)
class OperatorScope:
    """One explicit, non-broadening operator scope."""

    kind: OperatorScopeKind
    values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            kind = OperatorScopeKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise OperatorContractError("invalid operator scope kind") from exc
        values = tuple(dict.fromkeys(_text(item, "scope value", required=True) for item in self.values))
        needs_value = kind in {
            OperatorScopeKind.CHAIN,
            OperatorScopeKind.CHAINS,
            OperatorScopeKind.UUID,
            OperatorScopeKind.UUIDS,
            OperatorScopeKind.TEMPORAL_RANGE,
            OperatorScopeKind.CURSOR,
        }
        if needs_value and not values:
            raise OperatorContractError(f"scope {kind.value} requires at least one value")
        if kind in {OperatorScopeKind.SYSTEM, OperatorScopeKind.ACTIVE_TASKS} and values:
            raise OperatorContractError(f"scope {kind.value} does not accept values")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "values": list(self.values)}

    @classmethod
    def system(cls) -> "OperatorScope":
        return cls(OperatorScopeKind.SYSTEM)

    @classmethod
    def from_selector(
        cls,
        *,
        chain_id: str | None = None,
        uuid: str | None = None,
        all_tasks: bool = False,
    ) -> "OperatorScope":
        """Normalize one CLI selector into the shared scope contract."""
        selected = sum(bool(value) for value in (chain_id, uuid, all_tasks))
        if selected != 1:
            raise OperatorContractError(
                "scope selector requires exactly one of chain_id, uuid, or all_tasks"
            )
        if chain_id:
            return cls(OperatorScopeKind.CHAIN, (chain_id,))
        if uuid:
            return cls(OperatorScopeKind.UUID, (uuid,))
        return cls(OperatorScopeKind.SYSTEM)

    @classmethod
    def chains(cls, values: tuple[str, ...] | list[str]) -> "OperatorScope":
        return cls(OperatorScopeKind.CHAINS, tuple(values))

    @classmethod
    def uuids(cls, values: tuple[str, ...] | list[str]) -> "OperatorScope":
        return cls(OperatorScopeKind.UUIDS, tuple(values))

    def split(self) -> tuple["OperatorScope", ...]:
        """Split a multi-value scope into equivalent single-value scopes."""
        if self.kind in {OperatorScopeKind.CHAINS, OperatorScopeKind.UUIDS}:
            single = OperatorScopeKind.CHAIN if self.kind is OperatorScopeKind.CHAINS else OperatorScopeKind.UUID
            return tuple(OperatorScope(single, (value,)) for value in self.values)
        return (self,)

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorScope":
        raw = _mapping(value, "scope")
        values = raw.get("values", ())
        if isinstance(values, str) or not isinstance(values, (list, tuple)):
            raise OperatorContractError("scope values must be a list")
        return cls(raw.get("kind", ""), tuple(values))


@dataclass(frozen=True, slots=True)
class OperatorCoverage:
    """Proof describing exactly how much of a requested scope was observed."""

    kind: CoverageKind
    source: str
    reason: str = ""
    observed: tuple[str, ...] = ()
    omitted_count: int = 0
    snapshot_id: str = ""
    mutation_epoch: str = ""

    def __post_init__(self) -> None:
        try:
            kind = CoverageKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise OperatorContractError("invalid coverage kind") from exc
        source = _text(self.source, "coverage source", required=True)
        observed = tuple(dict.fromkeys(_text(item, "observed identity", required=True) for item in self.observed))
        omitted = self.omitted_count
        if isinstance(omitted, bool) or not isinstance(omitted, int) or omitted < 0:
            raise OperatorContractError("coverage omitted_count must be a non-negative integer")
        if kind is CoverageKind.COMPLETE and omitted:
            raise OperatorContractError("complete coverage cannot omit identities")
        if kind is CoverageKind.UNAVAILABLE and not _text(self.reason, "coverage reason", required=True):
            raise OperatorContractError("unavailable coverage requires a reason")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "reason", _text(self.reason, "coverage reason"))
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "snapshot_id", _text(self.snapshot_id, "snapshot_id"))
        object.__setattr__(self, "mutation_epoch", _text(self.mutation_epoch, "mutation_epoch"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "source": self.source,
            "reason": self.reason,
            "observed": list(self.observed),
            "omitted_count": self.omitted_count,
            "snapshot_id": self.snapshot_id or None,
            "mutation_epoch": self.mutation_epoch or None,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorCoverage":
        raw = _mapping(value, "coverage")
        observed = raw.get("observed", ())
        if isinstance(observed, str) or not isinstance(observed, (list, tuple)):
            raise OperatorContractError("coverage observed must be a list")
        return cls(
            kind=raw.get("kind", ""),
            source=raw.get("source", ""),
            reason=raw.get("reason", "") or "",
            observed=tuple(observed),
            omitted_count=raw.get("omitted_count", 0),
            snapshot_id=raw.get("snapshot_id", "") or "",
            mutation_epoch=raw.get("mutation_epoch", "") or "",
        )


@dataclass(frozen=True, slots=True)
class CoverageRequirement:
    """Minimum evidence quality required by an inspection or effect plan."""

    minimum: CoverageKind = CoverageKind.COMPLETE

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "minimum", CoverageKind(self.minimum))
        except (TypeError, ValueError) as exc:
            raise OperatorContractError("invalid coverage requirement") from exc

    def accepts(self, coverage: OperatorCoverage) -> bool:
        if not isinstance(coverage, OperatorCoverage):
            raise OperatorContractError("coverage requirement expects OperatorCoverage")
        return _COVERAGE_RANK[coverage.kind] >= _COVERAGE_RANK[self.minimum]

    def to_dict(self) -> dict[str, str]:
        return {"minimum": self.minimum.value}

    @classmethod
    def from_mapping(cls, value: object) -> "CoverageRequirement":
        raw = _mapping(value, "coverage requirement")
        return cls(raw.get("minimum", ""))


@dataclass(frozen=True, slots=True)
class OperatorCursor:
    """Deterministic page position bound to immutable observation evidence."""

    snapshot_id: str
    configuration_fingerprint: str
    mutation_epoch: str
    position: int = 0
    page_size: int = 100

    def __post_init__(self) -> None:
        snapshot_id = _text(self.snapshot_id, "cursor snapshot_id", required=True)
        fingerprint = _text(self.configuration_fingerprint, "cursor configuration_fingerprint", required=True)
        epoch = _text(self.mutation_epoch, "cursor mutation_epoch", required=True)
        if isinstance(self.position, bool) or not isinstance(self.position, int) or self.position < 0:
            raise OperatorContractError("cursor position must be a non-negative integer")
        page_size = _positive(self.page_size, "cursor page_size", 100_000)
        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "configuration_fingerprint", fingerprint)
        object.__setattr__(self, "mutation_epoch", epoch)
        object.__setattr__(self, "page_size", page_size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "configuration_fingerprint": self.configuration_fingerprint,
            "mutation_epoch": self.mutation_epoch,
            "position": self.position,
            "page_size": self.page_size,
        }

    def assert_compatible(self, snapshot_id: str, configuration_fingerprint: str, mutation_epoch: str) -> None:
        """Reject cursor reuse against changed observation evidence."""
        if self.snapshot_id != _text(snapshot_id, "snapshot_id", required=True):
            raise OperatorContractError("cursor belongs to a different snapshot")
        if self.configuration_fingerprint != _text(configuration_fingerprint, "configuration_fingerprint", required=True):
            raise OperatorContractError("cursor belongs to a different configuration")
        if self.mutation_epoch != _text(mutation_epoch, "mutation_epoch", required=True):
            raise OperatorContractError("cursor belongs to a different mutation epoch")

    def advance(self, count: int | None = None) -> "OperatorCursor":
        """Create the next deterministic page cursor without changing evidence."""
        step = self.page_size if count is None else count
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise OperatorContractError("cursor advance must be a positive integer")
        return OperatorCursor(
            self.snapshot_id,
            self.configuration_fingerprint,
            self.mutation_epoch,
            position=self.position + step,
            page_size=self.page_size,
        )

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorCursor":
        raw = _mapping(value, "cursor")
        return cls(
            snapshot_id=raw.get("snapshot_id", ""),
            configuration_fingerprint=raw.get("configuration_fingerprint", ""),
            mutation_epoch=raw.get("mutation_epoch", ""),
            position=raw.get("position", 0),
            page_size=raw.get("page_size", 100),
        )


@dataclass(frozen=True, slots=True)
class OperatorPage:
    """Bounded, deterministic page of operator data."""

    items: tuple[Mapping[str, Any], ...] = ()
    cursor: OperatorCursor | None = None
    complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.complete, bool):
            raise OperatorContractError("page complete must be boolean")
        normalized: list[Mapping[str, Any]] = []
        for item in self.items:
            normalized.append(dict(_mapping(item, "page item")))
        if self.cursor is not None and not isinstance(self.cursor, OperatorCursor):
            raise OperatorContractError("page cursor must be an OperatorCursor")
        if self.cursor is not None and len(normalized) > self.cursor.page_size:
            raise OperatorContractError("page contains more items than its cursor page_size")
        if self.complete and self.cursor is not None:
            raise OperatorContractError("complete page cannot contain a continuation cursor")
        object.__setattr__(self, "items", tuple(_freeze_json_value(item) for item in normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": _json_value(self.items),
            "cursor": None if self.cursor is None else self.cursor.to_dict(),
            "complete": self.complete,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorPage":
        raw = _mapping(value, "page")
        items = raw.get("items", ())
        if isinstance(items, (str, bytes)) or not isinstance(items, (list, tuple)):
            raise OperatorContractError("page items must be a list")
        cursor = raw.get("cursor")
        return cls(
            items=tuple(_mapping(item, "page item") for item in items),
            cursor=None if cursor is None else OperatorCursor.from_mapping(cursor),
            complete=raw.get("complete", True),
        )


OPERATOR_LIMIT_ENFORCEMENT_OWNERS: Mapping[str, str] = {
    "tasks": "snapshot_reader",
    "chains": "snapshot_reader",
    "occurrences": "occurrence_service",
    "history_links": "snapshot_reader",
    "findings": "inspector",
    "outbox_rows": "outbox_reader",
    "file_records": "file_provider",
    "scheduler_iterations": "scheduler_service",
    "wall_time_seconds": "invocation_context",
}


@dataclass(frozen=True, slots=True)
class OperatorLimits:
    """Independent safety limits for one operator invocation."""

    tasks: int = 100
    chains: int = 100
    occurrences: int = 1000
    history_links: int = 1000
    findings: int = 1000
    outbox_rows: int = 100
    file_records: int = 1000
    scheduler_iterations: int = 512
    wall_time_seconds: int = 120

    @classmethod
    def enforcement_owner(cls, field_name: str) -> str:
        """Return the declared owner for one resource limit."""
        try:
            return OPERATOR_LIMIT_ENFORCEMENT_OWNERS[str(field_name)]
        except KeyError as exc:
            raise OperatorContractError(f"no enforcement owner for limit {field_name!r}") from exc

    def __post_init__(self) -> None:
        for name in (
            "tasks", "chains", "occurrences", "history_links", "findings",
            "outbox_rows", "file_records", "scheduler_iterations", "wall_time_seconds",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name, 100_000))

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorLimits":
        raw = _mapping(value, "limits")
        fields = cls.__dataclass_fields__
        return cls(**{name: raw[name] for name in fields if name in raw})


@dataclass(frozen=True, slots=True)
class OperatorRequest:
    """Validated request shared by every operator composition root."""

    operation: OperatorOperation
    scope: OperatorScope
    start: str | None = None
    end: str | None = None
    detail: str = "summary"
    include_history: bool = False
    apply: bool = False
    limits: OperatorLimits = field(default_factory=OperatorLimits)
    coverage: CoverageRequirement = field(default_factory=CoverageRequirement)
    version: int = OPERATOR_API_VERSION

    def __post_init__(self) -> None:
        try:
            operation = OperatorOperation(self.operation)
        except (TypeError, ValueError) as exc:
            raise OperatorContractError("invalid operator operation") from exc
        if not isinstance(self.scope, OperatorScope):
            raise OperatorContractError("operator request requires an explicit scope")
        if not isinstance(self.limits, OperatorLimits):
            raise OperatorContractError("operator request requires typed limits")
        if not isinstance(self.coverage, CoverageRequirement):
            raise OperatorContractError("operator request requires a coverage requirement")
        detail = _text(self.detail, "detail", required=True)
        if detail not in {"summary", "standard", "verbose"}:
            raise OperatorContractError("detail must be summary, standard, or verbose")
        if not isinstance(self.include_history, bool) or not isinstance(self.apply, bool):
            raise OperatorContractError("include_history and apply must be boolean")
        if self.apply and self.coverage.minimum is not CoverageKind.COMPLETE:
            raise OperatorContractError("effectful operator requests require complete coverage")
        if isinstance(self.version, bool) or self.version != OPERATOR_API_VERSION:
            raise OperatorContractError(f"unsupported operator API version: {self.version!r}")
        start = None if self.start is None else _text(self.start, "start", required=True)
        end = None if self.end is None else _text(self.end, "end", required=True)
        if start and end and end < start:
            raise OperatorContractError("operator end must not precede start")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "detail", detail)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "operation": self.operation.value,
            "scope": self.scope.to_dict(),
            "start": self.start,
            "end": self.end,
            "detail": self.detail,
            "include_history": self.include_history,
            "apply": self.apply,
            "limits": self.limits.to_dict(),
            "coverage": self.coverage.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorRequest":
        raw = _mapping(value, "operator request")
        return cls(
            operation=raw.get("operation", ""),
            scope=OperatorScope.from_mapping(raw.get("scope")),
            start=raw.get("start"),
            end=raw.get("end"),
            detail=raw.get("detail", "summary"),
            include_history=raw.get("include_history", False),
            apply=raw.get("apply", False),
            limits=OperatorLimits.from_mapping(raw.get("limits", {})),
            coverage=CoverageRequirement.from_mapping(raw.get("coverage", {})),
            version=raw.get("version", 0),
        )


@dataclass(frozen=True, slots=True)
class OperatorFailure:
    """Stable failure evidence; raw exception text is optional detail only."""

    code: str
    message: str
    retryable: bool = False
    scope: OperatorScope | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = _text(self.code, "failure code", required=True)
        message = _text(self.message, "failure message", required=True)
        if not isinstance(self.retryable, bool):
            raise OperatorContractError("failure retryable must be boolean")
        if self.scope is not None and not isinstance(self.scope, OperatorScope):
            raise OperatorContractError("failure scope must be an OperatorScope")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "scope": None if self.scope is None else self.scope.to_dict(),
            "details": dict(self.details),
        }

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorFailure":
        raw = _mapping(value, "failure")
        scope_value = raw.get("scope")
        return cls(
            code=raw.get("code", ""),
            message=raw.get("message", ""),
            retryable=raw.get("retryable", False),
            scope=None if scope_value is None else OperatorScope.from_mapping(scope_value),
            details=_mapping(raw.get("details", {}), "failure details"),
        )


@dataclass(frozen=True, slots=True)
class OperatorResult:
    """Common envelope for JSON, text, Rich, and Navigator consumers."""

    operation: OperatorOperation
    status: OperatorStatus
    data: Mapping[str, Any] = field(default_factory=dict)
    failure: OperatorFailure | None = None
    page: OperatorPage | None = None
    extensions: Mapping[str, Any] = field(default_factory=dict)
    version: int = OPERATOR_API_VERSION

    def __post_init__(self) -> None:
        try:
            operation = OperatorOperation(self.operation)
            status = OperatorStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise OperatorContractError("invalid operator result operation or status") from exc
        if self.failure is not None and not isinstance(self.failure, OperatorFailure):
            raise OperatorContractError("result failure must be an OperatorFailure")
        if self.page is not None and not isinstance(self.page, OperatorPage):
            raise OperatorContractError("result page must be an OperatorPage")
        if not isinstance(self.extensions, Mapping):
            raise OperatorContractError("result extensions must be an object")
        if status is OperatorStatus.OK and self.failure is not None:
            raise OperatorContractError("ok result cannot contain failure evidence")
        if status in {OperatorStatus.UNAVAILABLE, OperatorStatus.ERROR} and self.failure is None:
            raise OperatorContractError(f"{status.value} result requires failure evidence")
        if isinstance(self.version, bool) or self.version != OPERATOR_API_VERSION:
            raise OperatorContractError(f"unsupported operator API version: {self.version!r}")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "data", dict(self.data))
        object.__setattr__(self, "extensions", dict(self.extensions))
        _json_value(self.data)
        _json_value(self.extensions)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema": f"nautical.operator.{self.operation.value}",
            "version": self.version,
            "operation": self.operation.value,
            "status": self.status.value,
            "data": _json_value(self.data),
            "failure": None if self.failure is None else self.failure.to_dict(),
            "page": None if self.page is None else self.page.to_dict(),
        }
        result.update({key: _json_value(value) for key, value in self.extensions.items() if key not in result})
        return result

    @property
    def exit_code(self) -> OperatorExitCode:
        """Stable process outcome for this result envelope."""
        return exit_code_for_status(self.status)

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorResult":
        raw = _mapping(value, "operator result")
        failure_value = raw.get("failure")
        page_value = raw.get("page")
        known = {"schema", "version", "operation", "status", "data", "failure", "page"}
        return cls(
            operation=raw.get("operation", ""),
            status=raw.get("status", ""),
            data=_mapping(raw.get("data", {}), "result data"),
            failure=None if failure_value is None else OperatorFailure.from_mapping(failure_value),
            page=None if page_value is None else OperatorPage.from_mapping(page_value),
            extensions={key: value for key, value in raw.items() if key not in known},
            version=raw.get("version", 0),
        )


@dataclass(frozen=True, slots=True)
class OperatorV2Result:
    """Public v2 result with a stable top-level document contract.

    ``payload`` contains the operation-specific fields and is emitted at the
    document top level. Reserved envelope fields cannot be shadowed.
    """

    schema: str
    operation: str
    status: OperatorV2Status
    payload: Mapping[str, Any] = field(default_factory=dict)
    failure: OperatorFailure | None = None
    page: OperatorPage | None = None
    extensions: Mapping[str, Any] = field(default_factory=dict)
    version: int = OPERATOR_RESULT_VERSION

    def __post_init__(self) -> None:
        schema = _text(self.schema, "result schema", required=True)
        operation = _text(self.operation, "result operation", required=True)
        if not schema.startswith("nautical."):
            raise OperatorContractError("result schema must start with 'nautical.'")
        try:
            status = OperatorV2Status(self.status)
        except (TypeError, ValueError) as exc:
            raise OperatorContractError("invalid operator v2 result status") from exc
        if not isinstance(self.payload, Mapping):
            raise OperatorContractError("result payload must be an object")
        if self.failure is not None and not isinstance(self.failure, OperatorFailure):
            raise OperatorContractError("result failure must be an OperatorFailure")
        if self.page is not None and not isinstance(self.page, OperatorPage):
            raise OperatorContractError("result page must be an OperatorPage")
        if status in {OperatorV2Status.OK, OperatorV2Status.FOUND, OperatorV2Status.EMPTY} and self.failure is not None:
            raise OperatorContractError("successful result cannot contain failure evidence")
        if status in {OperatorV2Status.INVALID, OperatorV2Status.UNAVAILABLE, OperatorV2Status.ERROR} and self.failure is None:
            raise OperatorContractError(f"{status.value} result requires failure evidence")
        if isinstance(self.version, bool) or self.version != OPERATOR_RESULT_VERSION:
            raise OperatorContractError(f"unsupported operator v2 result version: {self.version!r}")
        reserved = {"schema", "version", "operation", "status", "failure", "page"}
        if reserved.intersection(self.payload):
            raise OperatorContractError("result payload contains reserved envelope fields")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "payload", _freeze_json_value(self.payload))
        object.__setattr__(self, "extensions", _freeze_json_value(self.extensions))
        _json_value(self.payload)
        _json_value(self.extensions)

    def to_dict(self) -> dict[str, Any]:
        payload = cast(dict[str, Any], _json_value(self.payload))
        result: dict[str, Any] = {
            "schema": self.schema,
            "version": self.version,
            "operation": self.operation,
            "status": self.status.value,
            **payload,
            "failure": None if self.failure is None else self.failure.to_dict(),
            "page": None if self.page is None else self.page.to_dict(),
        }
        result.update({key: _json_value(value) for key, value in self.extensions.items() if key not in result})
        return result

    @property
    def exit_code(self) -> OperatorExitCode:
        return exit_code_for_v2_status(self.status)

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorV2Result":
        raw = _mapping(value, "operator v2 result")
        failure_value = raw.get("failure")
        page_value = raw.get("page")
        known = {"schema", "version", "operation", "status", "failure", "page"}
        return cls(
            schema=raw.get("schema", ""),
            version=raw.get("version", 0),
            operation=raw.get("operation", ""),
            status=raw.get("status", ""),
            payload={key: item for key, item in raw.items() if key not in known},
            failure=None if failure_value is None else OperatorFailure.from_mapping(failure_value),
            page=None if page_value is None else OperatorPage.from_mapping(page_value),
        )


@dataclass(frozen=True, slots=True)
class OperatorCapabilities:
    """Machine-readable discovery document for external operator clients."""

    operations: tuple[OperatorOperation, ...] = tuple(OperatorOperation)
    scopes: tuple[OperatorScopeKind, ...] = tuple(OperatorScopeKind)
    schemas: tuple[str, ...] = ()
    limits: OperatorLimits = field(default_factory=OperatorLimits)
    taskwarrior_version: str = ""
    optional_dependencies: Mapping[str, bool] = field(default_factory=dict)
    mutation_supported: bool = False
    version: int = OPERATOR_API_VERSION

    def __post_init__(self) -> None:
        operations = tuple(dict.fromkeys(OperatorOperation(item) for item in self.operations))
        scopes = tuple(dict.fromkeys(OperatorScopeKind(item) for item in self.scopes))
        schemas = tuple(dict.fromkeys(_text(item, "capability schema", required=True) for item in self.schemas))
        if not schemas:
            schemas = tuple(f"nautical.operator.{item.value}" for item in operations)
        if not operations or not scopes:
            raise OperatorContractError("capabilities must list operations and scopes")
        if not isinstance(self.limits, OperatorLimits):
            raise OperatorContractError("capabilities require typed limits")
        if not isinstance(self.mutation_supported, bool):
            raise OperatorContractError("mutation_supported must be boolean")
        if isinstance(self.version, bool) or self.version != OPERATOR_API_VERSION:
            raise OperatorContractError(f"unsupported operator API version: {self.version!r}")
        dependencies = {str(name): bool(value) for name, value in self.optional_dependencies.items()}
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "scopes", scopes)
        object.__setattr__(self, "schemas", schemas)
        object.__setattr__(self, "taskwarrior_version", _text(self.taskwarrior_version, "taskwarrior_version"))
        object.__setattr__(self, "optional_dependencies", dependencies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "nautical.operator.capabilities",
            "version": self.version,
            "operations": [item.value for item in self.operations],
            "scopes": [item.value for item in self.scopes],
            "schemas": list(self.schemas),
            "limits": self.limits.to_dict(),
            "taskwarrior_version": self.taskwarrior_version or None,
            "optional_dependencies": dict(self.optional_dependencies),
            "mutation_supported": self.mutation_supported,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorCapabilities":
        raw = _mapping(value, "capabilities")
        operations = raw.get("operations", ())
        scopes = raw.get("scopes", ())
        schemas = raw.get("schemas", ())
        if isinstance(operations, str) or not isinstance(operations, (list, tuple)):
            raise OperatorContractError("capabilities operations must be a list")
        if isinstance(scopes, str) or not isinstance(scopes, (list, tuple)):
            raise OperatorContractError("capabilities scopes must be a list")
        if isinstance(schemas, str) or not isinstance(schemas, (list, tuple)):
            raise OperatorContractError("capabilities schemas must be a list")
        return cls(
            operations=tuple(operations),
            scopes=tuple(scopes),
            schemas=tuple(schemas),
            limits=OperatorLimits.from_mapping(raw.get("limits", {})),
            taskwarrior_version=raw.get("taskwarrior_version", "") or "",
            optional_dependencies=_mapping(raw.get("optional_dependencies", {}), "optional_dependencies"),
            mutation_supported=raw.get("mutation_supported", False),
            version=raw.get("version", 0),
        )


@dataclass(frozen=True, slots=True)
class OperatorDependency:
    """Availability evidence captured once for an operator invocation."""

    name: str
    available: bool
    version: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        name = _text(self.name, "dependency name", required=True)
        if not isinstance(self.available, bool):
            raise OperatorContractError("dependency available must be boolean")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", _text(self.version, "dependency version"))
        object.__setattr__(self, "reason", _text(self.reason, "dependency reason"))
        if self.available and self.reason:
            raise OperatorContractError("available dependency cannot have an unavailable reason")
        if not self.available and not self.reason:
            raise OperatorContractError("unavailable dependency requires a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "version": self.version or None,
            "reason": self.reason or None,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorDependency":
        raw = _mapping(value, "dependency")
        return cls(
            name=raw.get("name", ""),
            available=raw.get("available", False),
            version=raw.get("version", "") or "",
            reason=raw.get("reason", "") or "",
        )


__all__ = [
    "OPERATOR_API_VERSION", "OPERATOR_RESULT_VERSION", "OperatorContractError", "OperatorOperation",
    "OperatorStatus", "OperatorV2Status", "OperatorExitCode", "OperatorPhase", "OperatorPhaseResult", "exit_code_for_status",
    "exit_code_for_v2_status", "OperatorScopeKind",
    "CoverageKind", "OperatorScope", "OperatorCoverage", "OperatorCursor", "OperatorPage", "CoverageRequirement", "OperatorLimits", "OPERATOR_LIMIT_ENFORCEMENT_OWNERS",
    "OperatorRequest", "OperatorFailure", "OperatorResult", "OperatorV2Result", "OperatorCapabilities", "OperatorDependency",
]
