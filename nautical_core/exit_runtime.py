"""Runtime-owned state and service builders for hook orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nautical_core.exit_models import (
    ExitApplyParentUpdateServices,
    ExitClearParentCallback,
    ExitCleanupCallback,
    ExitDiagnosticCallback,
    ExitEnsureChildServices,
    ExitExportCallback,
    ExitImportCallback,
    ExitParentNextlinkStateCallback,
    ExitParentNextlinkStateResult,
    ExitParentUpdateCallback,
    ExitPrecheckServices,
    ExitRetryOrManualReviewCallback,
)


@dataclass(slots=True)
class ExitRuntimeState:
    unit_of_work: Any | None = None
    repository: Any | None = None
    outbox_lock_failures_this_run: int = 0
    last_outbox_lock_diag_ts: float = 0.0
    diag_stats: dict[str, Any] = field(default_factory=dict)
    task_phase: str = ""
    startup_stats: dict[str, float | int] = field(default_factory=dict)
    lifecycle_parent_preflight: dict[str, dict[str, Any]] = field(default_factory=dict)
    lifecycle_batch_imported: set[str] = field(default_factory=set)
    lifecycle_batch_import_failed: set[str] = field(default_factory=set)


def new_runtime_state() -> ExitRuntimeState:
    return ExitRuntimeState()


@dataclass(slots=True)
class ExitRuntimeServices:
    state: ExitRuntimeState
    parent_nextlink_state: ExitParentNextlinkStateCallback
    retry_or_manual_review_for_lock: ExitRetryOrManualReviewCallback
    export_uuid: ExitExportCallback
    import_child: ExitImportCallback
    diag: ExitDiagnosticCallback
    update_parent_nextlink: ExitParentUpdateCallback
    clear_parent_nextlink_if_matches: ExitClearParentCallback
    cleanup_orphan_child: ExitCleanupCallback


def build_precheck_services(runtime: ExitRuntimeServices) -> ExitPrecheckServices:
    def parent_nextlink_state(
        parent_uuid: str,
        child_short: str,
        expected_prev: str | None = None,
        *,
        prefer_cache: bool = True,
        parent_guard: dict[str, Any] | None = None,
        guard_mismatch_fn: Any = None,
    ) -> ExitParentNextlinkStateResult:
        try:
            return runtime.parent_nextlink_state(
                parent_uuid,
                child_short,
                expected_prev,
                prefer_cache=prefer_cache,
                parent_guard=parent_guard,
                guard_mismatch_fn=guard_mismatch_fn,
            )
        except TypeError:
            return runtime.parent_nextlink_state(
                parent_uuid,
                child_short,
                expected_prev,
                prefer_cache=prefer_cache,
            )

    return ExitPrecheckServices(
        parent_nextlink_state=parent_nextlink_state,
        export_uuid=lambda uuid_str, *, prefer_cache=True: runtime.export_uuid(uuid_str, prefer_cache=prefer_cache),
        clear_parent_nextlink_if_matches=runtime.clear_parent_nextlink_if_matches,
        diag=runtime.diag,
        retry_or_manual_review_for_lock=runtime.retry_or_manual_review_for_lock,
    )


def build_ensure_child_services(runtime: ExitRuntimeServices) -> ExitEnsureChildServices:
    return ExitEnsureChildServices(
        export_uuid=lambda uuid_str, prefer_cache=True: runtime.export_uuid(uuid_str, prefer_cache=prefer_cache),
        import_child=runtime.import_child,
        clear_parent_nextlink_if_matches=runtime.clear_parent_nextlink_if_matches,
        diag=runtime.diag,
        retry_or_manual_review_for_lock=runtime.retry_or_manual_review_for_lock,
    )


def build_apply_parent_update_services(runtime: ExitRuntimeServices) -> ExitApplyParentUpdateServices:
    return ExitApplyParentUpdateServices(
        update_parent_nextlink=runtime.update_parent_nextlink,
        cleanup_orphan_child=runtime.cleanup_orphan_child,
        diag=runtime.diag,
        retry_or_manual_review_for_lock=runtime.retry_or_manual_review_for_lock,
    )


__all__ = (
    'ExitRuntimeState',
    'ExitRuntimeServices',
    'new_runtime_state',
    'build_precheck_services',
    'build_ensure_child_services',
    'build_apply_parent_update_services',
)
