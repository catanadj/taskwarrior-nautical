from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ExitDrainStateProtocol(Protocol):
    requeue: list[dict[str, Any]]
    errors: int
    sqlite_acked_claims: dict[int, str]

    def dead_letter(self, entry: dict[str, Any], reason: str) -> None: ...
    def record_lock_event(self, idx: int) -> bool: ...
    def to_stats_model(
        self,
        drain_t0: float,
        requeue_ok: bool,
        requeue_failed: int,
    ) -> ExitDrainStats: ...


class ExitExportCallback(Protocol):
    def __call__(self, uuid_str: str, *, prefer_cache: bool = True) -> ExitExportResult: ...


class ExitParentNextlinkStateCallback(Protocol):
    def __call__(
        self,
        parent_uuid: str,
        child_short: str,
        expected_prev: str | None = None,
        *,
        prefer_cache: bool = True,
    ) -> ExitParentNextlinkStateResult: ...


class ExitImportCallback(Protocol):
    def __call__(self, child: dict[str, Any]) -> ExitImportResult: ...


class ExitParentUpdateCallback(Protocol):
    def __call__(
        self,
        parent_uuid: str,
        child_short: str,
        expected_prev: str | None = None,
    ) -> ExitParentUpdateResult: ...


class ExitClearParentCallback(Protocol):
    def __call__(self, parent_uuid: str, child_short: str) -> ExitParentUpdateResult: ...


class ExitCleanupCallback(Protocol):
    def __call__(self, child_uuid: str, spawn_intent_id: str = "") -> None: ...


class ExitLockErrorCallback(Protocol):
    def __call__(self, error: str) -> bool: ...


class ExitDiagnosticCallback(Protocol):
    def __call__(self, message: str) -> None: ...


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
    is_lock_error: ExitLockErrorCallback
    diag: ExitDiagnosticCallback
    requeue_or_dead_letter_for_lock: ExitRequeueCallback


@dataclass(slots=True)
class ExitEnsureChildServices:
    export_uuid: ExitExportCallback
    import_child: ExitImportCallback
    clear_parent_nextlink_if_matches: ExitClearParentCallback
    is_lock_error: ExitLockErrorCallback
    diag: ExitDiagnosticCallback
    requeue_or_dead_letter_for_lock: ExitRequeueCallback


@dataclass(slots=True)
class ExitApplyParentUpdateServices:
    update_parent_nextlink: ExitParentUpdateCallback
    is_lock_error: ExitLockErrorCallback
    cleanup_orphan_child: ExitCleanupCallback
    diag: ExitDiagnosticCallback
    requeue_or_dead_letter_for_lock: ExitRequeueCallback


@dataclass(slots=True)
class ExitExportResult:
    exists: bool
    retryable: bool
    err: str
    obj: dict[str, Any] | None

    @property
    def state(self) -> str:
        return "found" if self.exists else ("unavailable" if self.retryable else "absent")


@dataclass(slots=True)
class ExitEquivalentChildResult:
    exists: bool
    retryable: bool
    err: str
    obj: dict[str, Any] | None

    @property
    def state(self) -> str:
        return "found" if self.exists else ("unavailable" if self.retryable else "absent")


@dataclass(slots=True)
class ExitImportResult:
    ok: bool
    err: str


@dataclass(slots=True)
class ExitParentNextlinkStateResult:
    state: str
    err: str


@dataclass(slots=True)
class ExitParentUpdateResult:
    ok: bool
    err: str


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
