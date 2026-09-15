"""Hook-specific service assembly kept at the modify composition boundary."""

from __future__ import annotations

from typing import Any

from .chain_generation import CarryFieldError
from .task_datetime import datetime_value, parser_for_host
from .task_models import TaskPayload


def _capabilities(host: Any):
    return host._module("modify_composition").capabilities_for(host)


def _runtime(host: Any, runtime: Any = None):
    composition = host._module("modify_composition")
    return runtime or composition.ModifyRuntimeServices.from_host(host)


def expiration_services_for(host: Any):
    capabilities = _capabilities(host)
    modify_expiration = capabilities.modify_expiration
    generation_module = capabilities.modify_generation_effects
    generation = generation_module.chain_generation_service(generation_module.generation_ports_for(host))
    task_codec = capabilities.task_codec
    task_models = capabilities.task_models
    ui = capabilities.modify_ui_effects
    ui_ports = ui.ui_ports_for(host)

    def typed_task(task):
        return task_models.NauticalTask.from_observation(
            task_codec.DEFAULT_TASK_CODEC.decode_row(task, source_query="on-modify expiration")
        )

    return modify_expiration.ExpirationServices(
        core=host.core,
        reconcile=capabilities.chain_integrity_lifecycle,
        safe_parse_datetime=host._TASK_DATETIME_PARSER.parse,
        compute_anchor_child_due=lambda task: generation.compute_anchor_child_due(typed_task(task)),
        compute_cp_child_due=lambda task: generation.compute_cp_child_due(typed_task(task)),
        build_child_draft=lambda task, *args, **kwargs: generation.build_child_draft(typed_task(task), *args, **kwargs),
        stage_recovery_plan=lambda plan: capabilities.modify_spawn_effects.enqueue_spawn_intent(
            capabilities.modify_spawn_effects.spawn_intent_ports_for(host), plan
        ),
        panel=lambda title, rows, **kwargs: ui.panel(ui_ports, title, rows, **kwargs),
        short=host.core.short_uuid,
        diag=host._diag,
    )


def expiration_recovery_warning_for(host: Any, new: TaskPayload, reason: str) -> None:
    capabilities = _capabilities(host)
    modify_expiration = capabilities.modify_expiration
    if modify_expiration is not None:
        try:
            modify_expiration.render_recovery_warning(new, reason, services=expiration_services_for(host))
            return
        except Exception as exc:
            host._diag(f"expiration recovery warning render failed: {exc}")
    ui = capabilities.modify_ui_effects
    ui.panel(
        ui.ui_ports_for(host),
        "⚠ Nautical expiration recovery deferred",
        [("Task", host.core.short_uuid(new.get("uuid")) or "–"), ("Reason", reason or "The next occurrence could not be prepared."), ("Action", "Run nautical reconcile --apply.")],
        kind="warning",
    )


def handle_expired_deleted(host: Any, new: TaskPayload) -> bool:
    modify_expiration = _capabilities(host).modify_expiration
    return modify_expiration.handle_expired_deleted_modify(new, services=expiration_services_for(host))


def handle_non_completion(
    host: Any,
    old: TaskPayload,
    new: TaskPayload,
    unit_of_work,
    *,
    transition=None,
    runtime: Any = None,
) -> None:
    runtime = _runtime(host, runtime)
    runtime.runtime_state().task_repository = unit_of_work.repository
    capabilities = runtime.non_completion
    modify_ordinary = capabilities.modify_ordinary
    modify_lifecycle = capabilities.modify_lifecycle
    diagnostics = capabilities.modify_diagnostics_effects
    validation = capabilities.modify_validation_effects
    ui = capabilities.modify_ui_effects
    ui_ports = ui.ui_ports_for(host)
    field_changed = (
        (lambda _old, _new, field: transition.changed(field))
        if transition is not None
        else capabilities.modify_task_fields.field_changed
    )
    services = modify_ordinary.OrdinaryModifyServices(
        field_changed=field_changed,
        strip_quotes=capabilities.modify_task_fields.strip_quotes,
        validate_anchor=lambda old_task, new_task, expr: validation.validate_anchor(
            validation.anchor_validation_ports_for(host), old_task, new_task, expr
        ),
        validate_omit=lambda anchor, anchor_file, omit, omit_file: validation.validate_omit(
            validation.omit_validation_ports_for(host), anchor, anchor_file, omit, omit_file
        ),
        reject_conflicting_types=host.core._import_sibling("hook_validation_pipeline").reject_recurrence_kind_conflict,
        validate_chain_limits=lambda task: validation.validate_chain_limits(validation.chain_limit_ports_for(host), task),
        preserve_cp_offsets=lambda old_task, new_task, cp: runtime.preserve_cp_relative_offsets(
            old_task, new_task, cp, transition=transition
        ),
        task_has_recurrence=modify_lifecycle.task_has_nautical_recurrence_fields,
        preserve_native_until=lambda old_task, new_task, kind: runtime.preserve_native_until(
            old_task, new_task, kind, transition=transition
        ),
        validate_native_until=lambda task: validation.validate_native_until(validation.native_until_ports_for(host), task),
        validate_native_until_slots=lambda task: validation.validate_native_until_slots(
            validation.native_until_slot_ports_for(host), task
        ),
        render_cp_adjustment=lambda adjustment: render_cp_schedule_adjusted_panel_for(host, adjustment),
        render_timing_warning=lambda task, fields: render_explicit_timing_order_warning_for(host, task, fields),
        apply_transition=lambda old_task, new_task: modify_lifecycle.apply_nautical_transition(
            old_task, new_task, short_uuid=host.core.short_uuid,
        ),
        short_uuid=host.core.short_uuid,
        recurrence_enabled_rows=lambda task, source: recurrence_enabled_rows_for(host, task, source),
        panel=lambda title, rows, **kwargs: ui.panel(ui_ports, title, rows, **kwargs),
        render_disabled_summary=lambda old_task, new_task, decision: render_disabled_chain_summary_for(host, old_task, new_task, decision),
        semantic_diff_value=validation.semantic_diff_value,
        first_recurrence_target=lambda task, source: first_recurrence_target_for(host, task, source),
        fmtlocal=host._fmtlocal,
        render_recurrence_updated=lambda changes, task: render_recurrence_updated_panel_for(host, changes, task),
        print_task=lambda task: ui.print_task(ui_ports, task),
    )
    try:
        modify_ordinary.handle_non_completion_modify(
            old, new, services=services, lifecycle=modify_lifecycle, transition=transition
        )
    except CarryFieldError as exc:
        host._fail_and_exit("Nautical carry failed", str(exc))
    except modify_ordinary.RecurrenceActivationError as exc:
        host._fail_and_exit("Nautical recurrence activation failed", str(exc))


def handle_completion(
    host: Any,
    old: TaskPayload,
    new: TaskPayload,
    unit_of_work,
    *,
    transition=None,
    runtime: Any = None,
):
    runtime = _runtime(host, runtime)
    runtime.runtime_state().task_repository = unit_of_work.repository
    capabilities = runtime.completion
    completion = capabilities.modify_completion_effects
    modify_completion_flow = runtime.import_module("nautical_core.modify_completion_flow")
    preflight_ports = completion.completion_preflight_context_ports_for(host)
    spawn_ports = completion.completion_spawn_ports_for(host)
    finalize_services = modify_completion_flow.CompletionFinalizeServices(
        build_and_spawn_child=lambda task, **kwargs: completion.build_and_spawn_child(
            spawn_ports, task, **kwargs
        ),
        seed_runtime_lookup_tasks=runtime.seed_runtime_lookup_tasks,
        modify_chain_state=runtime.runtime_state,
        lifecycle_read_service=runtime.lifecycle_read_service(),
        chain_health_advice=runtime.chain_health_advice,
        chain_integrity_warnings=runtime.chain_integrity_warnings,
        render_anchor_completion_feedback=runtime.render_anchor_completion_feedback,
        render_cp_completion_feedback=runtime.render_cp_completion_feedback,
        render_lifecycle_result=runtime.render_lifecycle_result,
        print_task=runtime.print_task,
        diag_summary=runtime.diag_summary,
        show_analytics=runtime.show_analytics,
        check_integrity=runtime.check_integrity,
        analytics_style=runtime.analytics_style,
        diagnostic=runtime.diagnostic,
    )
    flow_services = modify_completion_flow.CompletionFlowServices(
        runtime_state=host._modify_runtime_state,
        prepare_recurrence=lambda old_task, new_task: runtime.prepare_recurrence(old_task, new_task, transition=transition),
        preserve_cp_relative_offsets=lambda old_task, new_task, cp: runtime.preserve_cp_relative_offsets(old_task, new_task, cp, transition=transition),
        preserve_native_until=lambda old_task, new_task, kind: runtime.preserve_native_until(old_task, new_task, kind, transition=transition),
        validate_native_until=runtime.validate_native_until,
        validate_native_until_slots=runtime.validate_native_until_slots,
        now_utc=runtime.now_utc,
        preflight_context=lambda task, now, repository: completion.preflight_context(
            preflight_ports, task, now, repository
        ),
        compute_next_and_limits=runtime.compute_next_and_limits,
        lifecycle_read_service=runtime.lifecycle_read_service(),
        diag_count=host._diag_count,
        diag_lifecycle_result=host._diag_lifecycle_result,
        finalize_completion=modify_completion_flow.finalize_completion_modify,
        finalize_services=finalize_services,
        transition=transition,
    )
    return modify_completion_flow.handle_completion_modify(old, new, unit_of_work, services=flow_services)


def handle_deleted(
    host: Any,
    old: TaskPayload,
    new: TaskPayload,
    unit_of_work,
    *,
    transition=None,
    terminal_decision=None,
    runtime: Any = None,
) -> None:
    runtime = _runtime(host, runtime)
    runtime.runtime_state().task_repository = unit_of_work.repository
    capabilities = runtime.deletion
    diagnostics = capabilities.modify_diagnostics_effects
    ui = capabilities.modify_ui_effects
    ui_ports = ui.ui_ports_for(host)
    summary_ports = diagnostics.end_chain_summary_ports_for(host)
    modify_expiration = capabilities.modify_expiration
    if modify_expiration is None:
        expiration_recovery_warning_for(host, new, "Expiration recovery module is unavailable; deletion was not classified.")
        return
    services = modify_expiration.DeletedModifyServices(
        expiration=expiration_services_for(host),
        terminal_chain_off=lambda task, event=None: ensure_terminal_chain_off_for(host, task, event),
        now_utc=host.core.now_utc,
        end_chain_summary=lambda task, reason, now, current_task=None: diagnostics.end_chain_summary(
            summary_ports, task, reason, now, current_task
        ),
        format_root_and_age=lambda task, now: capabilities.modify_queries.cached_format_root_and_age(
            capabilities.modify_queries.query_ports_for(host), task, now
        ),
        short=host.core.short_uuid,
        panel=lambda title, rows, **kwargs: ui.panel(ui_ports, title, rows, **kwargs),
        diag=host._diag,
        recovery_warning=lambda task, reason: expiration_recovery_warning_for(host, task, reason),
    )
    modify_expiration.handle_deleted_modify(
        old, new, services=services, transition=transition, terminal_decision=terminal_decision
    )


def render_recurrence_updated_panel_for(host: Any, changes: list[tuple[str, str, str]], new: TaskPayload) -> None:
    feedback = host._module("modify_feedback")
    models = host._module("modify_models")
    add_validation = host.core._import_sibling("add_validation")
    ui = host._module("modify_ui_effects")
    ui_ports = ui.ui_ports_for(host)
    feedback.render_recurrence_updated_panel(
        changes,
        models.TaskView.from_mapping(new),
        parse_datetime=lambda value: datetime_value(parser_for_host(host), value),
        format_local=host._fmtlocal,
        describe_native_until_carry=add_validation.describe_native_until_carry,
        to_local=host.core.to_local,
        coerce_int=host.core.coerce_int,
        describe_anchor=host.core.describe_anchor_expr,
        resolve_omit_presets=host.core._parser_api.resolve_omit_presets,
        first_recurrence_target=lambda task, source: first_recurrence_target_for(host, task, source),
        panel_mode=getattr(host.core, "PANEL_MODE", "rich"),
        strip_markup=host.core.strip_rich_markup,
        panel=lambda title, rows, **kwargs: ui.panel(ui_ports, title, rows, **kwargs),
    )


def first_recurrence_target_for(host: Any, new: TaskPayload, source: str) -> Any:
    task_view = host._module("modify_models").TaskView.from_mapping(new)
    generation = host._module("modify_generation_effects")
    return host._module("modify_completion_compute").first_recurrence_target(
        task_view,
        source,
        parse_datetime=lambda value: datetime_value(parser_for_host(host), value),
        format_datetime=host.core.fmt_isoz,
        generation_service=lambda: generation.chain_generation_service(generation.generation_ports_for(host)),
    )


def recurrence_enabled_rows_for(host: Any, new: TaskPayload, source: str) -> list[tuple[str, str]]:
    task_view = host._module("modify_models").TaskView.from_mapping(new)
    return host._module("modify_feedback").recurrence_enabled_rows(
        task_view,
        source,
        describe_anchor=host.core.describe_anchor_expr,
        parse_cp_sequence_tokens=host.core.parse_cp_sequence_tokens,
        first_recurrence_target=lambda task, value: first_recurrence_target_for(host, task, value),
        format_local=host._fmtlocal,
    )


def render_cp_schedule_adjusted_panel_for(host: Any, adjustment: Any) -> None:
    ui = host._module("modify_ui_effects")
    ui_ports = ui.ui_ports_for(host)
    host._module("modify_feedback").render_cp_schedule_adjusted_panel(
        adjustment,
        format_local=host._fmtlocal,
        semantic_diff_value=host._module("modify_validation_effects").semantic_diff_value,
        format_offset=host._module("modify_value_effects").format_delta,
        panel=lambda title, rows, **kwargs: ui.panel(ui_ports, title, rows, **kwargs),
    )


def render_explicit_timing_order_warning_for(host: Any, new: TaskPayload, changed_fields: tuple[str, ...]) -> None:
    task = host._module("modify_models").TaskView.from_mapping(new)
    ui = host._module("modify_ui_effects")
    ui_ports = ui.ui_ports_for(host)
    host._module("modify_feedback").render_explicit_timing_order_warning(
        task,
        changed_fields,
        format_offset=host._module("modify_value_effects").format_delta,
        panel=lambda title, rows, **kwargs: ui.panel(ui_ports, title, rows, **kwargs),
    )


def render_disabled_chain_summary_for(host: Any, old: TaskPayload, new: TaskPayload, decision: Any) -> None:
    reason = str(getattr(decision, "reason", decision))
    if not (old.get("chainID") or new.get("chainID")):
        return
    models = host._module("modify_models")
    old_view = models.TaskView.from_mapping(old)
    new_view = models.TaskView.from_mapping(new)
    now_utc = host._workflow_now_utc()
    try:
        diagnostics = host._module("modify_diagnostics_effects")
        diagnostics.end_chain_summary(
            diagnostics.end_chain_summary_ports_for(host), old_view, reason, now_utc, current_task=new_view
        )
    except Exception as exc:
        host._diag(f"removed recurrence chain summary failed: {exc}")
        ui = host._module("modify_ui_effects")
        ui_ports = ui.ui_ports_for(host)
        queries = host._module("modify_queries")
        ui.panel(
            ui_ports,
            "⛔ Nautical chain stopped",
            [
                ("Reason", reason),
                ("Root", queries.cached_format_root_and_age(queries.query_ports_for(host), old_view, now_utc)),
                ("Task", host.core.short_uuid(old_view.get("uuid")) or "–"),
            ],
            kind="summary",
        )


def ensure_terminal_chain_off_for(host: Any, task: TaskPayload, event: str | None = None) -> bool:
    if event:
        lifecycle_models = host._module("lifecycle_models")
        lifecycle_planner = host._module("lifecycle_planner")
        task_codec = host._module("task_codec")
        lifecycle_planner.terminal_plan_for_snapshot(
            lifecycle_models.TaskSnapshot.from_observation(
                task_codec.DEFAULT_TASK_CODEC.decode_row(task, source_query="on-modify terminal")
            ),
            lifecycle_models.LifecycleEvent(event),
        )
    return host._module("modify_lifecycle").ensure_terminal_chain_off(task)


def render_anchor_completion_feedback_for(host: Any, *, request: Any) -> None:
    feedback = host._module("modify_feedback")
    models = host._module("modify_models")
    ui = host._module("modify_ui_effects")
    ui_ports = ui.ui_ports_for(host)
    feedback.orchestrate_anchor_completion_feedback(
        request=request,
        core=host.core,
        panel=lambda title, rows, **options: ui.panel(ui_ports, title, rows, **options),
        calendar_feedback=host.importlib.import_module("nautical_core.calendar_feedback"),
        panel_diagnostics=host._module("panel_diagnostics"),
        modify_models=models,
        modify_runtime=host._module("modify_runtime"),
        build_runtime_services=lambda: runtime_services_for(host),
    )


def render_cp_completion_feedback_for(host: Any, *, request: Any) -> None:
    feedback = host._module("modify_feedback")
    feedback.orchestrate_cp_completion_feedback(
        request=request,
        core=host.core,
        panel_diagnostics=host._module("panel_diagnostics"),
        modify_models=host._module("modify_models"),
        modify_runtime=host._module("modify_runtime"),
        build_runtime_services=lambda: runtime_services_for(host),
    )


def timeline_lines_for(host: Any, kind: str, task: Any, child_due_utc: Any, child_short: str, dnf: Any, **kwargs: Any) -> list[str]:
    if not host._require_core():
        return []
    schedule = host._module("modify_schedule_effects")
    evaluator_callback, service_callback = schedule.scheduler_callbacks(schedule.scheduler_ports_for(host))
    collector_override = kwargs.pop("_collect_prev_two_override", None)
    collect_prev_two = collector_override if callable(collector_override) else (
        lambda value, chain_by_link=None: host._module("modify_read_effects").collect_prev_two(
            host._module("modify_read_effects").PreviousChainPorts(
                service=host._module("modify_composition").lifecycle_read_service_for(host),
                panel_chain_by_link=host._modify_runtime_state().panel_chain_by_link,
                panel_chain_snapshot_loaded=host._modify_runtime_state().panel_chain_snapshot_loaded,
            ), value, chain_by_link
        )
    )
    timeline = host._module("modify_timeline")
    presentation = host._module("modify_presentation_effects")
    chain_style_ports = presentation.chain_style_ports_for(host)
    anchor_omit = host._module("anchor_omit")
    formatting = host._module("modify_format_effects")
    projection_services = timeline.TimelineProjectionServices(
        max_iterations=host._MAX_ITERATIONS,
        collect_prev_two=collect_prev_two,
        dtparse=lambda value: datetime_value(parser_for_host(host), value),
        to_local_cached=host._to_local_cached,
        safe_parse_datetime=host._TASK_DATETIME_PARSER.parse,
        omit_dnf_from_parent=lambda value: host._module("modify_anchor_effects").omit_dnf_from_parent(
            host._module("modify_anchor_effects").omit_ports_for(host), value
        ),
        omit_description_for_date=anchor_omit.omit_description_for_date,
        recurrence_evaluator_for_task=evaluator_callback,
        scheduler_service_for_task=service_callback,
    )
    formatting_services = timeline.TimelineFormattingServices(
        future_style_for_chain=lambda value, value_kind: presentation.future_style_for_chain(
            chain_style_ports, value, value_kind
        ),
        coerce_int=host.core.coerce_int,
        fmt_on_time_delta=lambda due, end, tol=60: formatting.on_time_delta(
            formatting.HumanDeltaPort(host.core.humanize_delta), due, end, tol
        ),
        fmtlocal=host._fmtlocal,
        fmt_dt_local=host.core.fmt_dt_local,
        short=host.core.short_uuid,
        format_gap=timeline.format_gap,
    )
    return timeline.timeline_lines_for_task(
        kind, task, child_due_utc, child_short, dnf, **kwargs,
        projection=projection_services, formatting=formatting_services,
    )


def runtime_services_for(host: Any) -> Any:
    runtime = host._module("modify_runtime")
    ui = host._module("modify_ui_effects")
    ui_ports = ui.ui_ports_for(host)
    presentation = host._module("modify_presentation_effects")
    chain_style_ports = presentation.chain_style_ports_for(host)
    formatting = host._module("modify_format_effects")
    line_preview_ports = formatting.line_preview_ports_for(host)
    queries = host._module("modify_queries")
    task_fields = host._module("modify_task_fields")
    feedback = host._module("modify_feedback")
    return runtime.ModifyRuntimeServices(
        state=host._modify_runtime_state(), core=host.core,
        debug_wait_sched=host._DEBUG_WAIT_SCHED,
        last_wait_sched_debug=host._LAST_WAIT_SCHED_DEBUG,
        diag_enabled=host.os.environ.get("NAUTICAL_DIAG") == "1",
        format_root_and_age=lambda task, now: queries.cached_format_root_and_age(
            queries.query_ports_for(host), task, now
        ),
        append_next_wait_sched_rows=host._append_next_wait_sched_rows,
        timeline_lines=getattr(host, "_timeline_lines", lambda *args, **kwargs: timeline_lines_for(host, *args, **kwargs)),
        show_timeline_gaps=host._SHOW_TIMELINE_GAPS,
        root_uuid_from=task_fields.root_uuid, short=host.core.short_uuid,
        format_next_anchor_rows=feedback.format_next_anchor_rows,
        format_next_cp_rows=feedback.format_next_cp_rows,
        format_line_preview=lambda *args, **kwargs: formatting.line_preview(line_preview_ports, *args, **kwargs),
        panel_line=lambda title, line, **kwargs: ui.panel_line(ui_ports, title, line, **kwargs),
        text_line=lambda line, **kwargs: ui.text_line(ui_ports, line, **kwargs),
        panel=lambda title, rows, **kwargs: ui.panel(ui_ports, title, rows, **kwargs),
        print_task=lambda task: ui.print_task(ui_ports, task), diag=host._diag,
        chain_color_per_chain=host._CHAIN_COLOR_PER_CHAIN,
        chain_colour_for_task=lambda task, kind: presentation.chain_colour_for_task(chain_style_ports, task, kind),
        strip_quotes=task_fields.strip_quotes,
        human_delta=lambda start, end, prefer=True: formatting.human_delta(
            formatting.HumanDeltaPort(host.core.humanize_delta), start, end, prefer
        ),
    )


__all__ = (
    "ensure_terminal_chain_off_for", "expiration_recovery_warning_for", "expiration_services_for",
    "first_recurrence_target_for", "handle_completion", "handle_deleted", "handle_expired_deleted",
    "handle_non_completion", "recurrence_enabled_rows_for", "render_anchor_completion_feedback_for",
    "render_cp_completion_feedback_for", "render_cp_schedule_adjusted_panel_for",
    "render_disabled_chain_summary_for", "render_explicit_timing_order_warning_for",
    "render_recurrence_updated_panel_for", "runtime_services_for", "timeline_lines_for",
)
