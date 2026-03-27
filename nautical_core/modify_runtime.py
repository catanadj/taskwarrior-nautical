from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time as _time

from nautical_core.modify_models import (
    AnchorFeedbackServices,
    CompletionComputeServices,
    CompletionPreflightServices,
    CompletionSpawnServices,
    CpFeedbackServices,
)


@dataclass(slots=True)
class ModifyRuntimeState:
    query_ctx: dict[str, dict[object, object]] = field(
        default_factory=lambda: {
            "task_text": {},
            "tw_get": {},
            "chain_root_age": {},
            "format_root_age": {},
        }
    )
    diag_stats: dict[str, Any] = field(
        default_factory=lambda: {
            "run_task_calls": 0,
            "run_task_failures": 0,
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
            "unexpected_cache_misses": 0,
            "chain_cache_seeded": 0,
            "run_task_seconds": 0.0,
        }
    )
    diag_start_ts: float = field(default_factory=_time.perf_counter)


def new_runtime_state() -> ModifyRuntimeState:
    return ModifyRuntimeState()


@dataclass(slots=True)
class ModifyRuntimeServices:
    state: ModifyRuntimeState
    core: Any
    debug_wait_sched: bool
    last_wait_sched_debug: Any
    diag_enabled: bool
    format_root_and_age: Any
    append_next_wait_sched_rows: Any
    timeline_lines: Any
    show_timeline_gaps: bool
    root_uuid_from: Any
    short: Any
    format_next_anchor_rows: Any
    format_next_cp_rows: Any
    format_line_preview: Any
    panel_line: Any
    panel: Any
    print_task: Any
    diag: Any
    chain_color_per_chain: bool
    chain_colour_for_task: Any
    strip_quotes: Any
    human_delta: Any
    export_uuid_short_cached: Any
    completion_link_numbers_or_fail: Any
    completion_kind_or_stop: Any
    completion_chain_id_or_fail: Any
    completion_existing_next_or_fail: Any
    completion_compute_child_due: Any
    completion_until_or_fail: Any
    completion_until_guard_or_stop: Any
    completion_require_child_due_or_fail: Any
    completion_warn_unreasonable_duration: Any
    completion_caps: Any
    completion_cap_guard_or_stop: Any
    build_child_from_parent: Any
    spawn_child_atomic: Any


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
        panel=runtime.panel,
        chain_color_per_chain=runtime.chain_color_per_chain,
        chain_colour_for_task=runtime.chain_colour_for_task,
        human_delta=runtime.human_delta,
        export_uuid_short_cached=runtime.export_uuid_short_cached,
    )


def build_preflight_services(runtime: ModifyRuntimeServices) -> CompletionPreflightServices:
    return CompletionPreflightServices(
        short=runtime.short,
        completion_link_numbers_or_fail=runtime.completion_link_numbers_or_fail,
        completion_kind_or_stop=runtime.completion_kind_or_stop,
        completion_chain_id_or_fail=runtime.completion_chain_id_or_fail,
        completion_existing_next_or_fail=runtime.completion_existing_next_or_fail,
    )


def build_compute_services(runtime: ModifyRuntimeServices) -> CompletionComputeServices:
    return CompletionComputeServices(
        completion_compute_child_due=runtime.completion_compute_child_due,
        completion_until_or_fail=runtime.completion_until_or_fail,
        completion_until_guard_or_stop=runtime.completion_until_guard_or_stop,
        completion_require_child_due_or_fail=runtime.completion_require_child_due_or_fail,
        completion_warn_unreasonable_duration=runtime.completion_warn_unreasonable_duration,
        completion_caps=runtime.completion_caps,
        completion_cap_guard_or_stop=runtime.completion_cap_guard_or_stop,
    )


def build_spawn_services(runtime: ModifyRuntimeServices) -> CompletionSpawnServices:
    return CompletionSpawnServices(
        build_child_from_parent=runtime.build_child_from_parent,
        spawn_child_atomic=runtime.spawn_child_atomic,
        panel=runtime.panel,
        print_task=runtime.print_task,
        diag=runtime.diag,
    )
