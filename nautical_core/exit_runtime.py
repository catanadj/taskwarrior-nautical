from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3
from typing import Any

from nautical_core.exit_models import (
    ExitApplyParentUpdateServices,
    ExitEnsureChildServices,
    ExitPrecheckServices,
)


@dataclass(slots=True)
class ExitRuntimeState:
    run_queue_db_conn: sqlite3.Connection | None = None
    run_queue_db_active: bool = False
    queue_db_open_count: int = 0
    queue_db_reuse_count: int = 0
    queue_lock_failures_this_run: int = 0
    last_queue_lock_diag_ts: float = 0.0
    diag_stats: dict[str, Any] = field(default_factory=dict)
    startup_stats: dict[str, float | int] = field(default_factory=dict)
    export_cache: dict[str, Any] = field(default_factory=dict)
    equiv_child_cache: dict[tuple[str, str, str], Any] = field(default_factory=dict)


def new_runtime_state() -> ExitRuntimeState:
    return ExitRuntimeState()


@dataclass(slots=True)
class ExitRuntimeServices:
    state: ExitRuntimeState
    parent_nextlink_state: Any
    requeue_or_dead_letter_for_lock: Any
    export_uuid: Any
    import_child: Any
    is_lock_error: Any
    diag: Any
    update_parent_nextlink: Any
    cleanup_orphan_child: Any


def build_precheck_services(runtime: ExitRuntimeServices) -> ExitPrecheckServices:
    return ExitPrecheckServices(
        parent_nextlink_state=lambda parent_uuid, child_short, expected_prev: runtime.parent_nextlink_state(
            parent_uuid, child_short, expected_prev, prefer_cache=True
        ),
        requeue_or_dead_letter_for_lock=runtime.requeue_or_dead_letter_for_lock,
    )


def build_ensure_child_services(runtime: ExitRuntimeServices) -> ExitEnsureChildServices:
    return ExitEnsureChildServices(
        export_uuid=lambda uuid_str, prefer_cache=True: runtime.export_uuid(uuid_str, prefer_cache=prefer_cache),
        import_child=runtime.import_child,
        is_lock_error=runtime.is_lock_error,
        diag=runtime.diag,
        requeue_or_dead_letter_for_lock=runtime.requeue_or_dead_letter_for_lock,
    )


def build_apply_parent_update_services(runtime: ExitRuntimeServices) -> ExitApplyParentUpdateServices:
    return ExitApplyParentUpdateServices(
        update_parent_nextlink=runtime.update_parent_nextlink,
        is_lock_error=runtime.is_lock_error,
        cleanup_orphan_child=runtime.cleanup_orphan_child,
        diag=runtime.diag,
        requeue_or_dead_letter_for_lock=runtime.requeue_or_dead_letter_for_lock,
    )
