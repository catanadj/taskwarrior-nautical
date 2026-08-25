"""Typed carry and recurrence-transition effects for on-modify."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .task_models import TaskPayload


def preserve_cp_relative_offsets_on_due_change(
    host: Any,
    old: TaskPayload,
    new: TaskPayload,
    new_cp: str,
    *,
    transition=None,
):
    result = host._module("modify_carry").preserve_cp_relative_offsets_on_due_change(
        old,
        new,
        new_cp,
        field_changed=(
            (lambda _old, _new, field: transition.changed(field))
            if transition is not None
            else host._field_changed
        ),
        parse_datetime=host.core.parse_dt_any,
        utc_to_local_naive=host._utc_to_local_naive,
        local_naive_to_utc=host._local_naive_to_utc,
        format_datetime=host.core.fmt_isoz,
        carry_error=host._module("chain_generation").CarryFieldError,
    )
    workflow = host._module("modify_carry_workflow")
    decision = workflow.decision_from_cp_adjustments(result)
    workflow.apply_temporal_carry_patch(new, decision)
    workflow.verify_temporal_carry_task(new, decision)
    return decision


def reject_native_until_carry(
    host: Any,
    old: TaskPayload,
    new: TaskPayload,
    new_target: datetime | None,
    old_target_field: str,
    exc: Exception,
) -> None:
    """Reject a target edit when its native expiration cannot be carried."""
    carry = None
    try:
        add_validation = host.core._import_sibling("add_validation")
        carry = add_validation.describe_native_until_carry(
            host.core.parse_dt_any(old.get("until")),
            host.core.parse_dt_any(old.get(old_target_field)),
            to_local=host.core.to_local,
        )
    except Exception:
        pass
    target_label = (
        host.core.fmt_dt_local(new_target)
        if isinstance(new_target, datetime)
        else str(host._recurrence_anchor_field(new) or "–")
    )
    rows = [("Target", target_label), ("Required", str(exc))]
    if carry:
        rows.insert(1, ("Carry", carry))
    host._panel("❌ Invalid expiration window", rows, kind="error")
    host.sys.exit(1)


def preserve_native_until_on_target_change(
    host: Any,
    old: TaskPayload,
    new: TaskPayload,
    kind: str,
    *,
    transition=None,
):
    carried = host._module("modify_carry").preserve_native_until_on_target_change(
        old,
        new,
        kind,
        field_changed=(
            (lambda _old, _new, field: transition.changed(field))
            if transition is not None
            else host._field_changed
        ),
        recurrence_anchor_field=host._recurrence_anchor_field,
        parse_datetime=host.core.parse_dt_any,
        native_until=host.core._import_sibling("native_until"),
        generation_service=host._chain_generation_service,
        reject_carry=lambda old_task, new_task, target, field, exc: reject_native_until_carry(
            host, old_task, new_task, target, field, exc
        ),
        diagnostic=host._diag,
    )
    if not carried:
        return host._module("modify_carry_workflow").NativeUntilDecision("unchanged")
    value = host.core.parse_dt_any(new.get("until"))
    if value is None:
        return host._module("modify_carry_workflow").NativeUntilDecision(
            "rejected", reason="native-until carry produced no parseable value"
        )
    decision = host._module("modify_carry_workflow").NativeUntilDecision(
        "carried", value=host._module("task_models").TaskTimestamp(value)
    )
    host._module("modify_carry_workflow").apply_native_until_patch(new, decision)
    host._module("modify_carry_workflow").verify_native_until_task(new, decision)
    return decision


def validate_completion_cp_and_anchor(host: Any, old: TaskPayload, new: TaskPayload, *, transition=None) -> tuple[str, str, str]:
    modify_validation = host._module("modify_validation")
    modify_lifecycle = host._module("modify_lifecycle")
    validation_effects = host._module("modify_validation_effects")
    return modify_validation.validate_completion_cp_and_anchor(
        old,
        new,
        services=modify_validation.CompletionValidationServices(
            strip_quotes=host._strip_quotes,
            reject_conflicting_types=host.core._import_sibling("hook_validation_pipeline").reject_recurrence_kind_conflict,
            validate_omit=lambda anchor, anchor_file, omit, omit_file: validation_effects.validate_omit(host, anchor, anchor_file, omit, omit_file),
            validate_chain_limits=lambda task: validation_effects.validate_chain_limits(host, task),
            parse_cp_sequence=host.core.parse_cp_sequence,
            cp_sequence_parse_error=host.core.cp_sequence_parse_error,
            field_changed=(
                (lambda _old, _new, field: transition.changed(field))
                if transition is not None
                else host._field_changed
            ),
            validate_anchor=lambda expr: validation_effects.validate_shared_anchor(host, expr),
            validate_cp=lambda cp, chain_max, chain_until: validation_effects.validate_cp(
                host, cp, chain_max, chain_until
            ),
            apply_transition=lambda old_task, new_task: modify_lifecycle.apply_nautical_transition(
                old_task,
                new_task,
                short_uuid=host.core.short_uuid,
            ),
            fail=host._fail_and_exit,
            diagnostic=host._diag,
        ),
    )


__all__ = (
    "preserve_cp_relative_offsets_on_due_change",
    "reject_native_until_carry",
    "preserve_native_until_on_target_change",
    "validate_completion_cp_and_anchor",
)
