from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ExitEntryContext:
    entry: dict[str, Any]
    idx: int
    state: Any
    parent_uuid: str
    child_short: str
    expected_parent_nextlink: str | None
    child: dict[str, Any]
    child_uuid: str
    spawn_intent_id: str


@dataclass(slots=True)
class ExitPrecheckServices:
    parent_nextlink_state: Any
    requeue_or_dead_letter_for_lock: Any


@dataclass(slots=True)
class ExitEnsureChildServices:
    export_uuid: Any
    import_child: Any
    is_lock_error: Any
    diag: Any
    requeue_or_dead_letter_for_lock: Any


@dataclass(slots=True)
class ExitApplyParentUpdateServices:
    update_parent_nextlink: Any
    is_lock_error: Any
    cleanup_orphan_child: Any
    diag: Any
    requeue_or_dead_letter_for_lock: Any


@dataclass(slots=True)
class ExitExportResult:
    exists: bool
    retryable: bool
    err: str
    obj: dict[str, Any] | None


@dataclass(slots=True)
class ExitEquivalentChildResult:
    exists: bool
    retryable: bool
    err: str
    obj: dict[str, Any] | None


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
            "drain_ms": self.drain_ms,
        }
