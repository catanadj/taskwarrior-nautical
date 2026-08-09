"""Runtime-owned state and service builders for hook orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any
import time as _time

from nautical_core.modify_models import (
    AnchorFeedbackServices,
    CompletionComputeServices,
    CompletionPreflightServices,
    CompletionSpawnServices,
    CpFeedbackServices,
    BuildChildCallback,
    CompletionCapGuardCallback,
    CompletionCapsCallback,
    CompletionChainIdCallback,
    CompletionChildDueCallback,
    CompletionChildRequiredCallback,
    CompletionDurationWarningCallback,
    CompletionExistingNextCallback,
    CompletionKindCallback,
    CompletionLinkNumbersCallback,
    CompletionSnapshotCallback,
    CompletionUntilCallback,
    CompletionUntilGuardCallback,
    SpawnChildCallback,
    ServiceCallback,
)


@dataclass(slots=True)
class ModifyRuntimeState:
    evaluator_sessions: dict[Any, tuple[Any, Any]] = field(default_factory=dict)
    chain_generation_service: Any = None
    query_ctx: dict[str, dict[object, object]] = field(
        default_factory=lambda: {
            "task_text": {},
            "tw_get": {},
            "read_query": {},
            "chain_root_age": {},
            "format_root_age": {},
        }
    )
    diag_stats: dict[str, Any] = field(
        default_factory=lambda: {
            "run_task_calls": 0,
            "run_task_failures": 0,
            "run_task_calls_get": 0,
            "run_task_calls_export_chain": 0,
            "run_task_calls_export_uuid_short": 0,
            "run_task_calls_export_uuid_full": 0,
            "run_task_calls_import": 0,
            "run_task_calls_count": 0,
            "run_task_calls_other": 0,
            "run_task_failures_get": 0,
            "run_task_failures_export_chain": 0,
            "run_task_failures_export_uuid_short": 0,
            "run_task_failures_export_uuid_full": 0,
            "run_task_failures_import": 0,
            "run_task_failures_count": 0,
            "run_task_failures_other": 0,
            "run_task_seconds_get": 0.0,
            "run_task_seconds_export_chain": 0.0,
            "run_task_seconds_export_uuid_short": 0.0,
            "run_task_seconds_export_uuid_full": 0.0,
            "run_task_seconds_import": 0.0,
            "run_task_seconds_count": 0.0,
            "run_task_seconds_other": 0.0,
            "export_uuid_cache_hits": 0,
            "export_uuid_cache_misses": 0,
            "export_full_cache_hits": 0,
            "export_full_cache_misses": 0,
            "tw_get_cache_hits": 0,
            "tw_get_cache_misses": 0,
            "task_text_cache_hits": 0,
            "task_text_cache_misses": 0,
            "chain_root_age_cache_hits": 0,
            "chain_root_age_cache_misses": 0,
            "format_root_age_cache_hits": 0,
            "format_root_age_cache_misses": 0,
            "read_query_cache_hits": 0,
            "read_query_cache_misses": 0,
            "read_query_cache_invalidations": 0,
            "read_query_cache_entries": 0,
            "chain_snapshot_hits": 0,
            "chain_snapshot_misses": 0,
            "chain_snapshot_filter_hits": 0,
            "chain_snapshot_truncations": 0,
            "unexpected_cache_misses": 0,
            "evaluator_session_hits": 0,
            "evaluator_session_misses": 0,
            "chain_cache_seeded": 0,
            "run_task_seconds": 0.0,
        }
    )
    diag_start_ts: float = field(default_factory=_time.perf_counter)
    panel_chain_by_link: dict[int, list[dict[str, Any]]] | None = None
    panel_chain_by_short: dict[str, dict[str, Any]] | None = None
    panel_chain_snapshot_loaded: bool = False
    chain_cache_chain_id: str = ""
    chain_cache: list[dict[str, Any]] = field(default_factory=list)
    chain_by_short: dict[str, dict[str, Any]] = field(default_factory=dict)
    chain_by_uuid: dict[str, dict[str, Any]] = field(default_factory=dict)
    chain_cache_lock: threading.RLock = field(default_factory=threading.RLock)
    anchor_file_providers: dict[tuple[str, str, tuple[int, int], str], Any] = field(
        default_factory=dict
    )


def new_runtime_state() -> ModifyRuntimeState:
    return ModifyRuntimeState()


@dataclass(slots=True)
class ModifyRuntimeServices:
    state: ModifyRuntimeState
    core: Any
    debug_wait_sched: bool
    last_wait_sched_debug: dict[str, Any]
    diag_enabled: bool
    format_root_and_age: ServiceCallback
    append_next_wait_sched_rows: ServiceCallback
    timeline_lines: ServiceCallback
    show_timeline_gaps: bool
    root_uuid_from: ServiceCallback
    short: ServiceCallback
    format_next_anchor_rows: ServiceCallback
    format_next_cp_rows: ServiceCallback
    format_line_preview: ServiceCallback
    panel_line: ServiceCallback
    text_line: ServiceCallback
    panel: ServiceCallback
    print_task: ServiceCallback
    diag: ServiceCallback
    chain_color_per_chain: bool
    chain_colour_for_task: ServiceCallback
    strip_quotes: ServiceCallback
    human_delta: ServiceCallback


def build_anchor_feedback_services(runtime: ModifyRuntimeServices) -> AnchorFeedbackServices:
    return AnchorFeedbackServices(
        core=runtime.core,
        debug_wait_sched=runtime.debug_wait_sched,
        last_wait_sched_debug=runtime.last_wait_sched_debug,
        diag_enabled=runtime.diag_enabled,
        format_root_and_age=runtime.format_root_and_age,
        append_next_wait_sched_rows=runtime.append_next_wait_sched_rows,
        timeline_lines=lambda kind, task, child_due, child_short, dnf, *, next_count, cap_no, cur_no, show_gaps: runtime.timeline_lines(
            kind,
            task,
            child_due,
            child_short,
            dnf,
            next_count=next_count,
            cap_no=cap_no,
            cur_no=cur_no,
            show_gaps=show_gaps,
            round_anchor_gaps=True,
        ),
        show_timeline_gaps=runtime.show_timeline_gaps,
        root_uuid_from=runtime.root_uuid_from,
        short=runtime.short,
        format_next_anchor_rows=runtime.format_next_anchor_rows,
        format_line_preview=runtime.format_line_preview,
        panel_line=runtime.panel_line,
        text_line=runtime.text_line,
        panel=runtime.panel,
        chain_color_per_chain=runtime.chain_color_per_chain,
        chain_colour_for_task=runtime.chain_colour_for_task,
        strip_quotes=runtime.strip_quotes,
        human_delta=runtime.human_delta,
    )


def build_cp_feedback_services(runtime: ModifyRuntimeServices) -> CpFeedbackServices:
    return CpFeedbackServices(
        core=runtime.core,
        diag_enabled=runtime.diag_enabled,
        format_root_and_age=runtime.format_root_and_age,
        append_next_wait_sched_rows=runtime.append_next_wait_sched_rows,
        timeline_lines=lambda kind, task, child_due, child_short, dnf, *, next_count, cap_no, cur_no, show_gaps: runtime.timeline_lines(
            kind,
            task,
            child_due,
            child_short,
            dnf,
            next_count=next_count,
            cap_no=cap_no,
            cur_no=cur_no,
            show_gaps=show_gaps,
            round_anchor_gaps=False,
        ),
        show_timeline_gaps=runtime.show_timeline_gaps,
        format_next_cp_rows=runtime.format_next_cp_rows,
        format_line_preview=runtime.format_line_preview,
        panel_line=runtime.panel_line,
        text_line=runtime.text_line,
        panel=runtime.panel,
        chain_color_per_chain=runtime.chain_color_per_chain,
        chain_colour_for_task=runtime.chain_colour_for_task,
        human_delta=runtime.human_delta,
    )


def build_preflight_services(
    *,
    short: ServiceCallback,
    completion_link_numbers_or_fail: CompletionLinkNumbersCallback,
    completion_kind_or_stop: CompletionKindCallback,
    completion_chain_id_or_fail: CompletionChainIdCallback,
    completion_chain_snapshot: CompletionSnapshotCallback,
    completion_existing_next_or_fail: CompletionExistingNextCallback,
) -> CompletionPreflightServices:
    return CompletionPreflightServices(
        short=short,
        completion_link_numbers_or_fail=completion_link_numbers_or_fail,
        completion_kind_or_stop=completion_kind_or_stop,
        completion_chain_id_or_fail=completion_chain_id_or_fail,
        completion_chain_snapshot=completion_chain_snapshot,
        completion_existing_next_or_fail=completion_existing_next_or_fail,
    )


def build_compute_services(
    *,
    completion_compute_child_due: CompletionChildDueCallback,
    completion_until_or_fail: CompletionUntilCallback,
    completion_until_guard_or_stop: CompletionUntilGuardCallback,
    completion_require_child_due_or_fail: CompletionChildRequiredCallback,
    completion_warn_unreasonable_duration: CompletionDurationWarningCallback,
    completion_caps: CompletionCapsCallback,
    completion_cap_guard_or_stop: CompletionCapGuardCallback,
) -> CompletionComputeServices:
    return CompletionComputeServices(
        completion_compute_child_due=completion_compute_child_due,
        completion_until_or_fail=completion_until_or_fail,
        completion_until_guard_or_stop=completion_until_guard_or_stop,
        completion_require_child_due_or_fail=completion_require_child_due_or_fail,
        completion_warn_unreasonable_duration=completion_warn_unreasonable_duration,
        completion_caps=completion_caps,
        completion_cap_guard_or_stop=completion_cap_guard_or_stop,
    )


def build_spawn_services(
    *,
    build_child_from_parent: BuildChildCallback,
    spawn_child_atomic: SpawnChildCallback,
    panel: ServiceCallback,
    print_task: ServiceCallback,
    diag: ServiceCallback,
) -> CompletionSpawnServices:
    return CompletionSpawnServices(
        build_child_from_parent=build_child_from_parent,
        spawn_child_atomic=spawn_child_atomic,
        panel=panel,
        print_task=print_task,
        diag=diag,
    )


__all__ = (
    'ModifyRuntimeState',
    'ModifyRuntimeServices',
    'new_runtime_state',
    'build_anchor_feedback_services',
    'build_cp_feedback_services',
    'build_preflight_services',
    'build_compute_services',
    'build_spawn_services',
)
