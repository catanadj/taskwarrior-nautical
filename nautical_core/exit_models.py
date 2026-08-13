from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from nautical_core.integration_models import TaskRead
    from nautical_core.lifecycle_models import LifecyclePlan


class ExitDrainStateProtocol(Protocol):
    requeue: list[dict[str, Any]]
    errors: int
    sqlite_acked_claims: dict[int, str]
    lifecycle_defer_verification: bool
    lifecycle_batch_discovery: bool
    lifecycle_batch_plan: LifecycleBatchPlan | None

    def dead_letter(self, entry: dict[str, Any], reason: str) -> None: ...
    def record_lock_event(self, idx: int) -> bool: ...
    def to_stats_model(
        self,
        drain_t0: float,
        requeue_ok: bool,
        requeue_failed: int,
    ) -> ExitDrainStats: ...


class ExitExportCallback(Protocol):
    def __call__(self, uuid_str: str, *, prefer_cache: bool = True) -> TaskRead[dict[str, Any]]: ...


class ExitParentNextlinkStateCallback(Protocol):
    def __call__(
        self,
        parent_uuid: str,
        child_short: str,
        expected_prev: str | None = None,
        *,
        prefer_cache: bool = True,
        parent_guard: dict[str, Any] | None = None,
        guard_mismatch_fn: Any = None,
    ) -> ExitParentNextlinkStateResult: ...


class ExitImportCallback(Protocol):
    def __call__(self, child: dict[str, Any]) -> ExitImportResult: ...


class ExitParentUpdateCallback(Protocol):
    def __call__(
        self,
        parent_uuid: str,
        child_short: str,
        expected_prev: str | None = None,
        *,
        parent_guard: dict[str, Any] | None = None,
    ) -> ExitParentUpdateResult: ...


class ExitClearParentCallback(Protocol):
    def __call__(self, parent_uuid: str, child_short: str) -> ExitParentUpdateResult: ...


class ExitCleanupCallback(Protocol):
    def __call__(self, child_uuid: str, spawn_intent_id: str = "") -> None: ...


class ExitParentGuardCallback(Protocol):
    def __call__(self, ctx: "ExitEntryContext") -> str: ...


class ExitDiagnosticCallback(Protocol):
    def __call__(self, message: str) -> None: ...


class ExitRecurrenceFingerprintCallback(Protocol):
    def __call__(self, task: dict[str, Any]) -> str: ...


class ExitRequeueCallback(Protocol):
    def __call__(self, entry: dict[str, Any], idx: int, state: ExitDrainStateProtocol) -> bool: ...


@dataclass(slots=True)
class ExitEntryContext:
    entry: dict[str, Any]
    idx: int
    state: ExitDrainStateProtocol
    parent_uuid: str
    child_short: str
    expected_parent_nextlink: str | None
    parent_guard: dict[str, str] | None
    child: dict[str, Any]
    child_uuid: str
    spawn_intent_id: str


@dataclass(slots=True)
class ExitPrecheckServices:
    parent_nextlink_state: ExitParentNextlinkStateCallback
    export_uuid: ExitExportCallback
    clear_parent_nextlink_if_matches: ExitClearParentCallback
    diag: ExitDiagnosticCallback
    requeue_or_dead_letter_for_lock: ExitRequeueCallback
    recurrence_fingerprint: ExitRecurrenceFingerprintCallback | None = None


@dataclass(slots=True)
class ExitEnsureChildServices:
    export_uuid: ExitExportCallback
    import_child: ExitImportCallback
    clear_parent_nextlink_if_matches: ExitClearParentCallback
    diag: ExitDiagnosticCallback
    requeue_or_dead_letter_for_lock: ExitRequeueCallback


@dataclass(slots=True)
class ExitApplyParentUpdateServices:
    update_parent_nextlink: ExitParentUpdateCallback
    cleanup_orphan_child: ExitCleanupCallback
    diag: ExitDiagnosticCallback
    requeue_or_dead_letter_for_lock: ExitRequeueCallback
    recheck_parent_guard: ExitParentGuardCallback | None = None


@dataclass(slots=True)
class ExitImportResult:
    ok: bool
    err: str
    retryable: bool = False


class LifecycleBatchDecisionKind(str, Enum):
    """Read-only classification used before a lifecycle batch mutates Taskwarrior."""

    ALREADY_SATISFIED = "already_satisfied"
    MISSING_CHILD = "missing_child"
    STALE_CONFLICTING = "stale_conflicting"
    UNAVAILABLE = "unavailable"
    READY_TO_APPLY = "ready_to_apply"


@dataclass(frozen=True, slots=True)
class LifecycleBatchDecision:
    """One claimed lifecycle intent and its authoritative preflight outcome."""

    spawn_intent_id: str
    entry: dict[str, Any]
    plan: "LifecyclePlan"
    kind: LifecycleBatchDecisionKind
    parent: dict[str, Any] | None = None
    child: dict[str, Any] | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not str(self.spawn_intent_id or "").strip():
            raise ValueError("lifecycle batch decision requires spawn_intent_id")
        if not isinstance(self.kind, LifecycleBatchDecisionKind):
            try:
                object.__setattr__(self, "kind", LifecycleBatchDecisionKind(self.kind))
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid lifecycle batch decision kind") from exc
        object.__setattr__(self, "reason", str(self.reason or "").strip())


@dataclass(frozen=True, slots=True)
class LifecycleBatchPlan:
    """Validated, mutation-free classification of one claimed queue batch."""

    decisions: tuple[LifecycleBatchDecision, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(self.decisions)
        seen: set[str] = set()
        for decision in normalized:
            if not isinstance(decision, LifecycleBatchDecision):
                raise ValueError("lifecycle batch plan contains an invalid decision")
            if decision.spawn_intent_id in seen:
                raise ValueError(f"duplicate lifecycle batch intent: {decision.spawn_intent_id}")
            seen.add(decision.spawn_intent_id)
        object.__setattr__(self, "decisions", normalized)

    def for_kind(self, kind: LifecycleBatchDecisionKind) -> tuple[LifecycleBatchDecision, ...]:
        return tuple(decision for decision in self.decisions if decision.kind is kind)

    def by_intent(self) -> dict[str, LifecycleBatchDecision]:
        return {decision.spawn_intent_id: decision for decision in self.decisions}


@dataclass(slots=True)
class ExitParentNextlinkStateResult:
    state: str
    err: str


@dataclass(slots=True)
class ExitParentUpdateResult:
    ok: bool
    err: str
    state: str = ""
    retryable: bool = False


@dataclass(slots=True)
class ExitQueueBatch:
    entries: list[dict[str, Any]]

    @property
    def entries_total(self) -> int:
        return len(self.entries)


@dataclass(slots=True)
class ExitRequeueResult:
    ok: bool
    failed: int


@dataclass(slots=True)
class ExitDrainStats:
    processed: int
    errors: int
    requeued: int
    requeue_failed: int
    dead_lettered: int
    queue_lock_failures: int
    entries_total: int
    entries_skipped_idempotent: int
    lock_events: int
    lock_streak_max: int
    circuit_breaks: int
    intent_log_ready: int
    intent_log_size: int
    intent_log_load_ms: float
    intent_mark_ok: int
    intent_mark_fail: int
    queue_db_opens: int
    queue_db_reuses: int
    preload_export_uuids: int
    preload_export_hits: int
    preload_export_misses: int
    preload_export_chunks: int
    drain_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "errors": self.errors,
            "requeued": self.requeued,
            "requeue_failed": self.requeue_failed,
            "dead_lettered": self.dead_lettered,
            "queue_lock_failures": self.queue_lock_failures,
            "entries_total": self.entries_total,
            "entries_skipped_idempotent": self.entries_skipped_idempotent,
            "lock_events": self.lock_events,
            "lock_streak_max": self.lock_streak_max,
            "circuit_breaks": self.circuit_breaks,
            "intent_log_ready": self.intent_log_ready,
            "intent_log_size": self.intent_log_size,
            "intent_log_load_ms": self.intent_log_load_ms,
            "intent_mark_ok": self.intent_mark_ok,
            "intent_mark_fail": self.intent_mark_fail,
            "queue_db_opens": self.queue_db_opens,
            "queue_db_reuses": self.queue_db_reuses,
            "preload_export_uuids": self.preload_export_uuids,
            "preload_export_hits": self.preload_export_hits,
            "preload_export_misses": self.preload_export_misses,
            "preload_export_chunks": self.preload_export_chunks,
            "drain_ms": self.drain_ms,
        }


@dataclass(slots=True)
class ExitQueueWriteResult:
    ok: bool
    count: int
