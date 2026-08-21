"""Immutable contracts for chain integrity auditing and repair planning.

This module is deliberately free of Taskwarrior, SQLite, scheduler, hook, and
presentation imports.  It describes observations and proposed operations; the
snapshot, invariant, planner, and application services own how those values
are obtained or applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, TypeAlias


class IntegrityContractError(ValueError):
    """Raised when an integrity model cannot safely cross a service boundary."""


class SnapshotCoverage(str, Enum):
    """How much authoritative Taskwarrior evidence a snapshot contains."""

    COMPLETE = "complete"
    CANDIDATES = "candidates"
    CHAIN = "chain"
    NARROW = "narrow"
    TRUNCATED = "truncated"
    UNAVAILABLE = "unavailable"


class ReferenceState(str, Enum):
    """Resolution state for a prevLink or nextLink reference."""

    RESOLVED = "resolved"
    ABSENT = "absent"
    AMBIGUOUS = "ambiguous"
    OUTSIDE_COVERAGE = "outside_coverage"
    UNAVAILABLE = "unavailable"


class FindingStatus(str, Enum):
    """Operational state of one integrity finding."""

    HEALTHY = "healthy"
    REPAIRABLE = "repairable"
    BLOCKED = "blocked"
    MANUAL_REVIEW = "manual_review"
    UNAVAILABLE = "unavailable"


class FindingSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class RepairSafety(str, Enum):
    """Whether a plan may be applied automatically."""

    SAFE = "safe"
    DEFERRED = "deferred"
    MANUAL = "manual"


class RepairOperationKind(str, Enum):
    """Named mutation families owned by the integrity application service."""

    LIFECYCLE_TRANSITION = "lifecycle_transition"
    LINK_REPAIR = "link_repair"
    NATIVE_UNTIL_REPAIR = "native_until_repair"
    METADATA_REPAIR = "metadata_repair"
    CHAIN_DISABLE = "chain_disable"
    OUTBOX_REPAIR = "outbox_repair"


class IntegrityReportStatus(str, Enum):
    HEALTHY = "healthy"
    REPAIRABLE = "repairable"
    MANUAL_REVIEW = "manual_review"
    UNAVAILABLE = "unavailable"


FrozenValue: TypeAlias = Any
FrozenPairs: TypeAlias = tuple[tuple[str, FrozenValue], ...]


def _required_text(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise IntegrityContractError(f"{field} is required")
    return text


def _freeze(value: object) -> FrozenValue:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return value


def _freeze_pairs(value: Mapping[str, object] | None) -> FrozenPairs:
    if not value:
        return ()
    return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))


def _thaw(value: FrozenValue) -> object:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw(item) for key, item in value}
        return [_thaw(item) for item in value]
    return value


def _positive_link(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise IntegrityContractError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ChainNode:
    """One observed Taskwarrior row in a chain graph.

    ``chain_id`` and ``link`` may be empty or unknown because the integrity
    engine must report those malformed observations.  Repair operations must
    use a complete identity and therefore cannot be built from such a node.
    """

    task_uuid: str
    chain_id: str
    link: int | None
    status: str
    fields: FrozenPairs = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_uuid", _required_text(self.task_uuid, "task UUID"))
        object.__setattr__(self, "chain_id", str(self.chain_id or "").strip())
        if self.link is not None and (isinstance(self.link, bool) or not isinstance(self.link, int) or self.link <= 0):
            raise IntegrityContractError("node link must be a positive integer or None")
        object.__setattr__(self, "status", _required_text(self.status, "task status").lower())
        fields = tuple(self.fields)
        if any(not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str) for item in fields):
            raise IntegrityContractError("node fields must be frozen key/value pairs")
        object.__setattr__(self, "fields", fields)

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "ChainNode":
        if not isinstance(row, Mapping):
            raise IntegrityContractError("chain row must be an object")
        task_uuid = _required_text(row.get("uuid"), "task UUID")
        raw_link = row.get("link")
        link: int | None
        if raw_link in (None, ""):
            link = None
        else:
            try:
                link = int(float(str(raw_link).strip()))
            except (TypeError, ValueError, OverflowError):
                link = None
        return cls(
            task_uuid,
            str(row.get("chainID", row.get("chain_id", "")) or ""),
            link,
            str(row.get("status", "") or ""),
            _freeze_pairs(row),
        )

    @property
    def has_complete_identity(self) -> bool:
        return bool(self.chain_id and self.link is not None)

    def field(self, name: str, default: object = None) -> object:
        for key, value in self.fields:
            if key == name:
                return _thaw(value)
        return default

    def to_dict(self) -> dict[str, object]:
        value = {key: _thaw(item) for key, item in self.fields}
        value.setdefault("uuid", self.task_uuid)
        if self.chain_id:
            value.setdefault("chainID", self.chain_id)
        if self.link is not None:
            value.setdefault("link", self.link)
        value.setdefault("status", self.status)
        return value


@dataclass(frozen=True, slots=True)
class ChainReference:
    """A typed graph edge target, including why it could not be resolved."""

    field: str
    token: str
    state: ReferenceState
    target_uuid: str = ""
    target_link: int | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        field = _required_text(self.field, "reference field")
        if field not in {"prevLink", "nextLink"}:
            raise IntegrityContractError("reference field must be prevLink or nextLink")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "token", str(self.token or "").strip())
        try:
            state = ReferenceState(self.state)
        except (TypeError, ValueError) as exc:
            raise IntegrityContractError("invalid reference state") from exc
        object.__setattr__(self, "state", state)
        target_uuid = str(self.target_uuid or "").strip()
        if state is ReferenceState.RESOLVED and not target_uuid:
            raise IntegrityContractError("resolved reference requires a target UUID")
        object.__setattr__(self, "target_uuid", target_uuid)
        if self.target_link is not None:
            object.__setattr__(self, "target_link", _positive_link(self.target_link, "target link"))
        object.__setattr__(self, "reason", str(self.reason or "").strip())


@dataclass(frozen=True, slots=True)
class ChainSnapshot:
    """One invocation-scoped, provenance-bearing chain observation."""

    snapshot_id: str
    coverage: SnapshotCoverage
    source: str
    rows: tuple[ChainNode, ...] = ()
    configuration_fingerprint: str = ""
    complete_chain_history: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _required_text(self.snapshot_id, "snapshot ID"))
        try:
            coverage = SnapshotCoverage(self.coverage)
        except (TypeError, ValueError) as exc:
            raise IntegrityContractError("invalid snapshot coverage") from exc
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "source", _required_text(self.source, "snapshot source"))
        rows = tuple(self.rows)
        if any(not isinstance(row, ChainNode) for row in rows):
            raise IntegrityContractError("snapshot rows must be ChainNode values")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "configuration_fingerprint", str(self.configuration_fingerprint or "").strip())
        object.__setattr__(self, "complete_chain_history", bool(self.complete_chain_history))
        reason = str(self.reason or "").strip()
        if coverage is SnapshotCoverage.UNAVAILABLE and not reason:
            raise IntegrityContractError("unavailable snapshot requires a reason")
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class IntegrityFinding:
    """Stable evidence produced by one pure invariant evaluation."""

    invariant_id: str
    status: FindingStatus
    severity: FindingSeverity
    snapshot_id: str
    chain_id: str = ""
    subject_uuids: tuple[str, ...] = ()
    reason_code: str = ""
    message: str = ""
    observed: FrozenPairs = ()
    expected: FrozenPairs = ()
    evidence: FrozenPairs = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "invariant_id", _required_text(self.invariant_id, "invariant ID"))
        try:
            status = FindingStatus(self.status)
            severity = FindingSeverity(self.severity)
        except (TypeError, ValueError) as exc:
            raise IntegrityContractError("invalid finding status or severity") from exc
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "snapshot_id", _required_text(self.snapshot_id, "finding snapshot ID"))
        object.__setattr__(self, "chain_id", str(self.chain_id or "").strip())
        subjects = tuple(_required_text(item, "finding subject UUID") for item in self.subject_uuids)
        object.__setattr__(self, "subject_uuids", subjects)
        object.__setattr__(self, "reason_code", _required_text(self.reason_code, "finding reason code"))
        object.__setattr__(self, "message", _required_text(self.message, "finding message"))
        for field_name in ("observed", "expected", "evidence"):
            fields = tuple(getattr(self, field_name))
            if any(not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str) for item in fields):
                raise IntegrityContractError(f"finding {field_name} must be frozen key/value pairs")
            object.__setattr__(self, field_name, fields)


@dataclass(frozen=True, slots=True)
class IntegrityOperation:
    """One named, guarded operation inside a repair plan."""

    operation_id: str
    kind: RepairOperationKind
    chain_id: str
    target_uuid: str
    guard: FrozenPairs
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    payload: FrozenPairs = ()
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _required_text(self.operation_id, "operation ID"))
        try:
            kind = RepairOperationKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise IntegrityContractError("invalid repair operation kind") from exc
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "chain_id", _required_text(self.chain_id, "operation chainID"))
        object.__setattr__(self, "target_uuid", _required_text(self.target_uuid, "operation target UUID"))
        guard = tuple(self.guard)
        if not guard:
            raise IntegrityContractError("repair operation requires a guard")
        object.__setattr__(self, "guard", guard)
        preconditions = tuple(_required_text(item, "operation precondition") for item in self.preconditions)
        postconditions = tuple(_required_text(item, "operation postcondition") for item in self.postconditions)
        if not postconditions:
            raise IntegrityContractError("repair operation requires a postcondition")
        object.__setattr__(self, "preconditions", preconditions)
        object.__setattr__(self, "postconditions", postconditions)
        object.__setattr__(self, "payload", tuple(self.payload))
        dependencies = tuple(_required_text(item, "operation dependency") for item in self.depends_on)
        if self.operation_id in dependencies:
            raise IntegrityContractError("repair operation cannot depend on itself")
        object.__setattr__(self, "depends_on", dependencies)


@dataclass(frozen=True, slots=True)
class IntegrityRepairPlan:
    """Complete deterministic repair plan produced without side effects."""

    plan_id: str
    snapshot_id: str
    chain_id: str
    safety: RepairSafety
    reason_code: str
    summary: str
    operations: tuple[IntegrityOperation, ...]
    configuration_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _required_text(self.plan_id, "repair plan ID"))
        object.__setattr__(self, "snapshot_id", _required_text(self.snapshot_id, "repair plan snapshot ID"))
        object.__setattr__(self, "chain_id", _required_text(self.chain_id, "repair plan chainID"))
        try:
            safety = RepairSafety(self.safety)
        except (TypeError, ValueError) as exc:
            raise IntegrityContractError("invalid repair safety") from exc
        object.__setattr__(self, "safety", safety)
        object.__setattr__(self, "reason_code", _required_text(self.reason_code, "repair plan reason code"))
        object.__setattr__(self, "summary", _required_text(self.summary, "repair plan summary"))
        operations = tuple(self.operations)
        if not operations:
            raise IntegrityContractError("repair plan requires at least one operation")
        if any(not isinstance(item, IntegrityOperation) for item in operations):
            raise IntegrityContractError("repair plan operations must be typed")
        operation_ids = tuple(item.operation_id for item in operations)
        if len(operation_ids) != len(set(operation_ids)):
            raise IntegrityContractError("repair plan operation IDs must be unique")
        if any(item.chain_id != self.chain_id for item in operations):
            raise IntegrityContractError("repair operation chainID differs from plan chainID")
        known_ids = set(operation_ids)
        if any(dependency not in known_ids for item in operations for dependency in item.depends_on):
            raise IntegrityContractError("repair plan contains an unknown operation dependency")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "configuration_fingerprint", str(self.configuration_fingerprint or "").strip())


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Typed audit result shared by reconcile, Doctor, and query consumers."""

    snapshot: ChainSnapshot
    status: IntegrityReportStatus
    findings: tuple[IntegrityFinding, ...] = ()
    plans: tuple[IntegrityRepairPlan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, ChainSnapshot):
            raise IntegrityContractError("integrity report requires a chain snapshot")
        try:
            status = IntegrityReportStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise IntegrityContractError("invalid integrity report status") from exc
        object.__setattr__(self, "status", status)
        findings = tuple(self.findings)
        plans = tuple(self.plans)
        if any(not isinstance(item, IntegrityFinding) for item in findings):
            raise IntegrityContractError("integrity findings must be typed")
        if any(not isinstance(item, IntegrityRepairPlan) for item in plans):
            raise IntegrityContractError("integrity plans must be typed")
        if any(item.snapshot_id != self.snapshot.snapshot_id for item in findings):
            raise IntegrityContractError("finding snapshot differs from report snapshot")
        if any(item.snapshot_id != self.snapshot.snapshot_id for item in plans):
            raise IntegrityContractError("repair plan snapshot differs from report snapshot")
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "plans", plans)


__all__ = [
    "ChainNode",
    "ChainReference",
    "ChainSnapshot",
    "FindingSeverity",
    "FindingStatus",
    "FrozenPairs",
    "IntegrityContractError",
    "IntegrityFinding",
    "IntegrityOperation",
    "IntegrityReport",
    "IntegrityReportStatus",
    "IntegrityRepairPlan",
    "ReferenceState",
    "RepairOperationKind",
    "RepairSafety",
    "SnapshotCoverage",
]
