"""Typed on-modify route effects.

These functions own route-level effect assembly; the legacy hook module is
passed only as a host for still-shared, lower-level helpers during extraction.
"""

from __future__ import annotations

from .chain_generation import CarryFieldError

from typing import Any

from .task_models import TaskPayload
from .modify_composition import ModifyRuntimeServices, capabilities_for


def _cap(host: Any):
    return capabilities_for(host)


def handle_non_completion(host: Any, old: TaskPayload, new: TaskPayload, unit_of_work, *, transition=None, runtime: ModifyRuntimeServices | None = None) -> None:
    runtime = runtime or ModifyRuntimeServices.from_host(host)
    runtime.runtime_state().task_repository = unit_of_work.repository
    capabilities = runtime.capabilities
    modify_ordinary = capabilities.modify_ordinary
    modify_lifecycle = capabilities.modify_lifecycle
    transition_effects = capabilities.modify_transition_effects
    presentation = capabilities.modify_presentation_effects
    diagnostics = capabilities.modify_diagnostics_effects
    validation = capabilities.modify_validation_effects
    ui = capabilities.modify_ui_effects
    field_changed = (
        (lambda _old, _new, field: transition.changed(field))
        if transition is not None
        else capabilities.modify_task_fields.field_changed
    )
    services = modify_ordinary.OrdinaryModifyServices(
        field_changed=field_changed,
        strip_quotes=capabilities.modify_task_fields.strip_quotes,
        validate_anchor=lambda old_task, new_task, expr: validation.validate_anchor(host, old_task, new_task, expr),
        validate_omit=lambda anchor, anchor_file, omit, omit_file: validation.validate_omit(host, anchor, anchor_file, omit, omit_file),
        reject_conflicting_types=host.core._import_sibling("hook_validation_pipeline").reject_recurrence_kind_conflict,
        validate_chain_limits=lambda task: validation.validate_chain_limits(validation.chain_limit_ports_for(host), task),
        preserve_cp_offsets=lambda old_task, new_task, cp: transition_effects.preserve_cp_relative_offsets_on_due_change(
            transition_effects.CPCarryPorts(
                carry=host._module("modify_carry").preserve_cp_relative_offsets_on_due_change,
                field_changed=capabilities.modify_task_fields.field_changed,
                parse_datetime=lambda value: host._TASK_DATETIME_PARSER.parse(value),
                utc_to_local_naive=host.core.utc_to_local_naive,
                local_naive_to_utc=host.core.local_naive_to_utc,
                format_datetime=host.core.fmt_isoz,
                carry_error=host._module("chain_generation").CarryFieldError,
                workflow=host._module("modify_carry_workflow"),
            ),
            old_task, new_task, cp, transition=transition,
        ),
        task_has_recurrence=modify_lifecycle.task_has_nautical_recurrence_fields,
        preserve_native_until=lambda old_task, new_task, kind: transition_effects.preserve_native_until_on_target_change(
            host,
            old_task, new_task, kind, transition=transition,
        ),
        validate_native_until=lambda task: validation.validate_native_until(validation.native_until_ports_for(host), task),
        validate_native_until_slots=lambda task: validation.validate_native_until_slots(
            validation.native_until_slot_ports_for(host), task
        ),
        render_cp_adjustment=lambda adjustment: presentation.render_cp_schedule_adjusted_panel(host, adjustment),
        render_timing_warning=lambda task, fields: presentation.render_explicit_timing_order_warning(host, task, fields),
        apply_transition=lambda old_task, new_task: modify_lifecycle.apply_nautical_transition(
            old_task, new_task, short_uuid=host.core.short_uuid,
        ),
        short_uuid=host.core.short_uuid,
        recurrence_enabled_rows=lambda task, source: presentation.recurrence_enabled_rows(host, task, source),
        panel=lambda title, rows, **kwargs: ui.panel(host, title, rows, **kwargs),
        render_disabled_summary=lambda old_task, new_task, decision: presentation.render_disabled_chain_summary(host, old_task, new_task, decision),
        semantic_diff_value=validation.semantic_diff_value,
        first_recurrence_target=lambda task, source: presentation.first_recurrence_target(host, task, source),
        fmtlocal=host._fmtlocal,
        render_recurrence_updated=lambda changes, task: presentation.render_recurrence_updated_panel(host, changes, task),
        print_task=lambda task: ui.print_task(host, task),
    )
    try:
        modify_ordinary.handle_non_completion_modify(
            old, new, services=services, lifecycle=modify_lifecycle, transition=transition
        )
    except CarryFieldError as exc:
        host._fail_and_exit("Nautical carry failed", str(exc))
    except modify_ordinary.RecurrenceActivationError as exc:
        host._fail_and_exit("Nautical recurrence activation failed", str(exc))


def handle_completion(host: Any, old: TaskPayload, new: TaskPayload, unit_of_work, *, transition=None, runtime: ModifyRuntimeServices | None = None):
    runtime = runtime or ModifyRuntimeServices.from_host(host)
    runtime.runtime_state().task_repository = unit_of_work.repository
    capabilities = runtime.capabilities
    completion = capabilities.modify_completion_effects
    transition_effects = capabilities.modify_transition_effects
    presentation = capabilities.modify_presentation_effects
    diagnostics = capabilities.modify_diagnostics_effects
    validation = capabilities.modify_validation_effects
    modify_completion_flow = runtime.import_module("nautical_core.modify_completion_flow")
    finalize_services = modify_completion_flow.CompletionFinalizeServices(
        build_and_spawn_child=lambda task, **kwargs: completion.build_and_spawn_child(host, task, **kwargs),
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
        preflight_context=lambda task, now, repository: completion.preflight_context(host, task, now, repository),
        compute_next_and_limits=runtime.compute_next_and_limits,
        lifecycle_read_service=runtime.lifecycle_read_service(),
        diag_count=host._diag_count,
        diag_lifecycle_result=host._diag_lifecycle_result,
        finalize_completion=modify_completion_flow.finalize_completion_modify,
        finalize_services=finalize_services,
        transition=transition,
    )
    return modify_completion_flow.handle_completion_modify(old, new, unit_of_work, services=flow_services)


def handle_deleted(host: Any, old: TaskPayload, new: TaskPayload, unit_of_work, *, transition=None, terminal_decision=None, runtime: ModifyRuntimeServices | None = None) -> None:
    runtime = runtime or ModifyRuntimeServices.from_host(host)
    runtime.runtime_state().task_repository = unit_of_work.repository
    capabilities = runtime.capabilities
    presentation = capabilities.modify_presentation_effects
    diagnostics = capabilities.modify_diagnostics_effects
    ui = capabilities.modify_ui_effects
    modify_expiration = capabilities.modify_expiration
    if modify_expiration is None:
        expiration_recovery_warning(host, new, "Expiration recovery module is unavailable; deletion was not classified.")
        return
    services = modify_expiration.DeletedModifyServices(
        expiration=expiration_services(host),
        terminal_chain_off=lambda task, event=None: presentation.ensure_terminal_chain_off(host, task, event),
        now_utc=host.core.now_utc,
        end_chain_summary=lambda task, reason, now, current_task=None: diagnostics.end_chain_summary(host, task, reason, now, current_task),
        format_root_and_age=lambda task, now: capabilities.modify_queries.cached_format_root_and_age(
            capabilities.modify_queries.query_ports_for(host), task, now
        ),
        short=host.core.short_uuid,
        panel=lambda title, rows, **kwargs: ui.panel(host, title, rows, **kwargs),
        diag=host._diag,
        recovery_warning=lambda task, reason: expiration_recovery_warning(host, task, reason),
    )
    modify_expiration.handle_deleted_modify(
        old, new, services=services, transition=transition, terminal_decision=terminal_decision
    )


def expiration_services(host: Any):
    capabilities = _cap(host)
    modify_expiration = capabilities.modify_expiration
    generation = capabilities.modify_generation_effects.chain_generation_service(
        capabilities.modify_generation_effects.generation_ports_for(host)
    )
    task_codec = capabilities.task_codec
    task_models = capabilities.task_models

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
        stage_recovery_plan=lambda plan: capabilities.modify_spawn_effects.enqueue_spawn_intent(host, plan),
        panel=lambda title, rows, **kwargs: capabilities.modify_ui_effects.panel(host, title, rows, **kwargs),
        short=host.core.short_uuid,
        diag=host._diag,
    )


def expiration_recovery_warning(host: Any, new: TaskPayload, reason: str) -> None:
    modify_expiration = _cap(host).modify_expiration
    if modify_expiration is not None:
        try:
            modify_expiration.render_recovery_warning(new, reason, services=expiration_services(host))
            return
        except Exception as exc:
            host._diag(f"expiration recovery warning render failed: {exc}")
    _cap(host).modify_ui_effects.panel(
        "⚠ Nautical expiration recovery deferred",
        [("Task", host.core.short_uuid(new.get("uuid")) or "–"), ("Reason", reason or "The next occurrence could not be prepared."), ("Action", "Run nautical reconcile --apply.")],
        kind="warning",
    )


def handle_expired_deleted(host: Any, new: TaskPayload) -> bool:
    modify_expiration = _cap(host).modify_expiration
    return modify_expiration.handle_expired_deleted_modify(new, services=expiration_services(host))


__all__ = (
    "handle_non_completion",
    "handle_completion",
    "handle_deleted",
    "expiration_services",
    "handle_expired_deleted",
)
