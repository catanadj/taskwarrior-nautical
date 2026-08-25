"""Typed on-modify route effects.

These functions own route-level effect assembly; the legacy hook module is
passed only as a host for still-shared, lower-level helpers during extraction.
"""

from __future__ import annotations

from typing import Any


def handle_non_completion(host: Any, old: dict, new: dict, unit_of_work, *, transition=None) -> None:
    host._modify_runtime_state().task_repository = unit_of_work.repository
    modify_ordinary = host._module("modify_ordinary")
    modify_lifecycle = host._module("modify_lifecycle")
    field_changed = (
        (lambda _old, _new, field: transition.changed(field))
        if transition is not None
        else host._field_changed
    )
    services = modify_ordinary.OrdinaryModifyServices(
        field_changed=field_changed,
        strip_quotes=host._strip_quotes,
        validate_anchor=host._non_completion_validate_anchor,
        validate_omit=host._validate_omit_for_anchor_or_fail,
        reject_conflicting_types=host.core._import_sibling("hook_validation_pipeline").reject_recurrence_kind_conflict,
        validate_chain_limits=host._validate_chain_limits_on_modify,
        preserve_cp_offsets=lambda old_task, new_task, cp: host._preserve_cp_relative_offsets_on_due_change(
            old_task, new_task, cp, transition=transition,
        ),
        task_has_recurrence=modify_lifecycle.task_has_nautical_recurrence_fields,
        preserve_native_until=lambda old_task, new_task, kind: host._preserve_native_until_on_target_change(
            old_task, new_task, kind, transition=transition,
        ),
        validate_native_until=host._validate_native_until_after_target_or_fail,
        validate_native_until_slots=host._validate_native_until_anchor_slots_or_fail,
        render_cp_adjustment=host._render_cp_schedule_adjusted_panel,
        render_timing_warning=host._render_explicit_timing_order_warning,
        apply_transition=lambda old_task, new_task: modify_lifecycle.apply_nautical_transition(
            old_task, new_task, short_uuid=host.core.short_uuid,
        ),
        short_uuid=host.core.short_uuid,
        recurrence_enabled_rows=host._recurrence_enabled_rows,
        panel=host._panel,
        render_disabled_summary=host._render_disabled_chain_summary,
        semantic_diff_value=host._semantic_diff_value,
        first_recurrence_target=host._first_recurrence_target,
        fmtlocal=host._fmtlocal,
        render_recurrence_updated=host._render_recurrence_updated_panel,
        print_task=host._print_task,
    )
    try:
        modify_ordinary.handle_non_completion_modify(
            old, new, services=services, lifecycle=modify_lifecycle, transition=transition
        )
    except host._module("chain_generation").CarryFieldError as exc:
        host._fail_and_exit("Nautical carry failed", str(exc))
    except modify_ordinary.RecurrenceActivationError as exc:
        host._fail_and_exit("Nautical recurrence activation failed", str(exc))


def handle_completion(host: Any, old: dict, new: dict, unit_of_work, *, transition=None):
    host._modify_runtime_state().task_repository = unit_of_work.repository
    modify_completion_flow = host.importlib.import_module("nautical_core.modify_completion_flow")
    finalize_services = modify_completion_flow.CompletionFinalizeServices(
        build_and_spawn_child=host._completion_build_and_spawn_child,
        seed_runtime_lookup_tasks=host._seed_runtime_lookup_tasks,
        modify_chain_state=host._modify_chain_state,
        lifecycle_read_service=host._lifecycle_read_service(),
        chain_health_advice=host._chain_health_advice,
        chain_integrity_warnings=host._chain_integrity_warnings,
        render_anchor_completion_feedback=host._render_anchor_completion_feedback,
        render_cp_completion_feedback=host._render_cp_completion_feedback,
        render_lifecycle_result=host._render_lifecycle_result,
        print_task=host._print_task,
        diag_summary=host._diag_summary,
        show_analytics=host._SHOW_ANALYTICS,
        check_integrity=host._CHECK_CHAIN_INTEGRITY,
        analytics_style=host._ANALYTICS_STYLE,
    )
    flow_services = modify_completion_flow.CompletionFlowServices(
        runtime_state=host._modify_runtime_state,
        prepare_recurrence=lambda old_task, new_task: host._completion_validate_cp_and_anchor(
            old_task, new_task, transition=transition,
        ),
        preserve_cp_relative_offsets=host._preserve_cp_relative_offsets_on_due_change,
        preserve_native_until=host._preserve_native_until_on_target_change,
        validate_native_until=host._validate_native_until_after_target_or_fail,
        validate_native_until_slots=host._validate_native_until_anchor_slots_or_fail,
        now_utc=host.core.now_utc,
        preflight_context=host._completion_preflight_context,
        compute_next_and_limits=host._completion_compute_next_and_limits,
        lifecycle_read_service=host._lifecycle_read_service(),
        diag_count=host._diag_count,
        diag_lifecycle_result=host._diag_lifecycle_result,
        finalize_completion=modify_completion_flow.finalize_completion_modify,
        finalize_services=finalize_services,
        transition=transition,
    )
    return modify_completion_flow.handle_completion_modify(old, new, unit_of_work, services=flow_services)


def handle_deleted(host: Any, old: dict, new: dict, unit_of_work, *, transition=None, terminal_decision=None) -> None:
    host._modify_runtime_state().task_repository = unit_of_work.repository
    modify_expiration = host._module("modify_expiration", required=False)
    if modify_expiration is None:
        host._expiration_recovery_warning(new, "Expiration recovery module is unavailable; deletion was not classified.")
        return
    services = modify_expiration.DeletedModifyServices(
        expiration=host._expiration_services(),
        terminal_chain_off=host._ensure_terminal_chain_off,
        now_utc=host.core.now_utc,
        end_chain_summary=host._end_chain_summary,
        format_root_and_age=host._format_root_and_age,
        short=host._short,
        panel=host._panel,
        diag=host._diag,
        recovery_warning=host._expiration_recovery_warning,
    )
    modify_expiration.handle_deleted_modify(
        old, new, services=services, transition=transition, terminal_decision=terminal_decision
    )


__all__ = ("handle_non_completion", "handle_completion", "handle_deleted")
