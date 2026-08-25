"""Runtime-owned state and service builders for hook orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any
import time as _time

from nautical_core.modify_models import (
    AnchorFeedbackServices,
    CompletionComputeServices,
    CompletionPreflightServices,
    CompletionSpawnServices,
    CpFeedbackServices,
    BuildChildDraftCallback,
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
    ChainColourCallback,
    DiagnosticCallback,
    FeedbackPanelCallback,
    FeedbackRowsFormatter,
    HumanDeltaCallback,
    PanelLineCallback,
    PanelCallback,
    PreviewLineFormatter,
    PrintTaskCallback,
    RootAgeFormatter,
    ShortUuidCallback,
    SpawnChildCallback,
    StripQuotesCallback,
    TextLineCallback,
    TimelineLinesCallback,
    WaitScheduleRowsCallback,
)


@dataclass(slots=True)
class ModifyRuntimeState:
    workflow_context: Any = None
    task_repository: Any = None
    scheduler_services: dict[Any, Any] = field(default_factory=dict)
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
            "run_task_calls_import": 0,
            "run_task_calls_count": 0,
            "run_task_calls_other": 0,
            "run_task_failures_get": 0,
            "run_task_failures_export_chain": 0,
            "run_task_failures_import": 0,
            "run_task_failures_count": 0,
            "run_task_failures_other": 0,
            "run_task_seconds_get": 0.0,
            "run_task_seconds_export_chain": 0.0,
            "run_task_seconds_import": 0.0,
            "run_task_seconds_count": 0.0,
            "run_task_seconds_other": 0.0,
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
    lifecycle_read_service: Any = None
    chain_cache_store: Any = None
    anchor_file_providers: dict[tuple[str, str, tuple[int, int], str], Any] = field(
        default_factory=dict
    )


def new_runtime_state() -> ModifyRuntimeState:
    return ModifyRuntimeState()


def scheduler_service_for_task(
    task: dict[str, Any],
    *,
    state: ModifyRuntimeState,
    core: Any,
    recurrence_seed_base: Callable[[dict[str, Any]], str],
) -> Any:
    """Return one cached scheduler service for the task's scheduling state."""
    identity = str(task.get("uuid") or task.get("chainID") or "").strip()
    cache_key: tuple[Any, ...]
    if identity:
        cache_key = (
            "task",
            identity,
            str(task.get("modified") or ""),
            str(task.get("anchor") or ""),
            str(task.get("anchor_file") or ""),
            str(task.get("omit") or ""),
            str(task.get("omit_file") or ""),
            str(task.get("cp") or ""),
            str(task.get("anchor_mode") or ""),
            str(task.get("chainMax") or ""),
            str(task.get("chainUntil") or ""),
            str(task.get("bc") or ""),
        )
    else:
        cache_key = ("object", id(task))
    cached_service = state.scheduler_services.get(cache_key)
    if cached_service is not None:
        state.diag_stats["evaluator_session_hits"] = state.diag_stats.get("evaluator_session_hits", 0) + 1
        return cached_service

    from nautical_core.recurrence_context import RecurrenceContext
    from nautical_core.scheduler_service import SchedulerService
    from nautical_core.task_codec import DEFAULT_TASK_CODEC
    from nautical_core.task_models import NauticalTask

    observation = DEFAULT_TASK_CODEC.decode_row(task, source_query="modify scheduler")
    domain_task = NauticalTask.from_observation(observation)
    workflow = state.workflow_context
    business_calendar = getattr(workflow, "business_calendar", None)
    if business_calendar is None:
        business_calendar = core.business_calendar_for_task(task)
    context = RecurrenceContext.from_observation(
        observation,
        timezone=core._LOCAL_TZ,
        business_calendar=business_calendar,
        astronomy_config=getattr(core, "ASTRONOMY_CONFIG", None),
        anchor_file_dir=getattr(core, "ANCHOR_FILE_DIR", ""),
    )
    service = SchedulerService.from_task(domain_task, context=context)
    state.scheduler_services[cache_key] = service
    state.diag_stats["evaluator_session_misses"] = state.diag_stats.get("evaluator_session_misses", 0) + 1
    return service


def anchor_file_provider_for(
    anchor_file: str,
    *,
    fallback_hhmm: tuple[int, int],
    seed_base: str,
    state: ModifyRuntimeState,
    core: Any,
) -> Any:
    """Return one cached anchor-file provider for a projection session."""
    if not anchor_file:
        return None
    anchor_file_dir = getattr(core, "ANCHOR_FILE_DIR", "")
    key = (anchor_file, anchor_file_dir, fallback_hhmm, seed_base)
    provider = state.anchor_file_providers.get(key)
    if provider is None:
        provider = core._import_sibling("anchor_inclusion")._build_anchor_file_provider(
            anchor_file,
            anchor_file_dir=anchor_file_dir,
            fallback_hhmm=fallback_hhmm,
            seed_base=seed_base,
            core=core,
        )
        state.anchor_file_providers[key] = provider
    return provider


@dataclass(slots=True)
class ModifyRuntimeServices:
    state: ModifyRuntimeState
    core: Any
    debug_wait_sched: bool
    last_wait_sched_debug: dict[str, Any]
    diag_enabled: bool
    format_root_and_age: RootAgeFormatter
    append_next_wait_sched_rows: WaitScheduleRowsCallback
    timeline_lines: TimelineLinesCallback
    show_timeline_gaps: bool
    root_uuid_from: Callable[[dict[str, Any]], str]
    short: ShortUuidCallback
    format_next_anchor_rows: FeedbackRowsFormatter
    format_next_cp_rows: FeedbackRowsFormatter
    format_line_preview: PreviewLineFormatter
    panel_line: PanelLineCallback
    text_line: TextLineCallback
    panel: FeedbackPanelCallback
    print_task: PrintTaskCallback
    diag: DiagnosticCallback
    chain_color_per_chain: bool
    chain_colour_for_task: ChainColourCallback
    strip_quotes: StripQuotesCallback
    human_delta: HumanDeltaCallback


def _timeline_lines_adapter(
    runtime: ModifyRuntimeServices,
    *,
    round_anchor_gaps: bool,
) -> TimelineLinesCallback:
    def timeline_lines(
        kind: str,
        task: dict[str, Any],
        child_due: Any,
        child_short: str,
        dnf: Any,
        *,
        next_count: int = 3,
        cap_no: int | None = None,
        cur_no: int | None = None,
        show_gaps: bool = True,
        round_anchor_gaps: bool = round_anchor_gaps,
    ) -> list[str]:
        return runtime.timeline_lines(
            kind,
            task,
            child_due,
            child_short,
            dnf,
            next_count=next_count,
            cap_no=cap_no,
            cur_no=cur_no,
            show_gaps=show_gaps,
            round_anchor_gaps=round_anchor_gaps,
        )

    return timeline_lines


def build_anchor_feedback_services(runtime: ModifyRuntimeServices) -> AnchorFeedbackServices:
    return AnchorFeedbackServices(
        core=runtime.core,
        debug_wait_sched=runtime.debug_wait_sched,
        last_wait_sched_debug=runtime.last_wait_sched_debug,
        diag_enabled=runtime.diag_enabled,
        format_root_and_age=runtime.format_root_and_age,
        append_next_wait_sched_rows=runtime.append_next_wait_sched_rows,
        timeline_lines=_timeline_lines_adapter(runtime, round_anchor_gaps=True),
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
        timeline_lines=_timeline_lines_adapter(runtime, round_anchor_gaps=False),
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
    short: ShortUuidCallback,
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
    build_child_draft: BuildChildDraftCallback,
    spawn_child_atomic: SpawnChildCallback,
    panel: PanelCallback,
    print_task: PrintTaskCallback,
    diag: DiagnosticCallback,
) -> CompletionSpawnServices:
    return CompletionSpawnServices(
        build_child_draft=build_child_draft,
        spawn_child_atomic=spawn_child_atomic,
        panel=panel,
        print_task=print_task,
        diag=diag,
    )


__all__ = (
    'ModifyRuntimeState',
    'ModifyRuntimeServices',
    'new_runtime_state',
    'scheduler_service_for_task',
    'anchor_file_provider_for',
    'build_anchor_feedback_services',
    'build_cp_feedback_services',
    'build_preflight_services',
    'build_compute_services',
    'build_spawn_services',
)
