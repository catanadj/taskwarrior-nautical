"""Typed carry and recurrence-transition effects for on-modify."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .task_models import TaskPayload
from .task_changes import TaskTransition
from .task_datetime import datetime_value, parser_for_host
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NativeCarryPorts:
    describe_carry: Any
    parse_datetime: Any
    to_local: Any
    format_local: Any
    anchor_field: Any
    panel: Any
    abort: Any


@dataclass(frozen=True, slots=True)
class CPCarryPorts:
    carry: Any
    field_changed: Any
    parse_datetime: Any
    utc_to_local_naive: Any
    local_naive_to_utc: Any
    format_datetime: Any
    carry_error: Any
    workflow: Any


@dataclass(frozen=True, slots=True)
class NativePreservePorts:
    carry: Any
    field_changed: Any
    anchor_field: Any
    parse_datetime: Any
    native_until: Any
    generation_service: Any
    reject_carry: Any
    diagnostic: Any
    workflow: Any
    timestamp: Any


def preserve_cp_relative_offsets_on_due_change(
    ports: CPCarryPorts,
    old: TaskPayload,
    new: TaskPayload,
    new_cp: str,
    *,
    transition: TaskTransition | None = None,
) -> Any:
    result = ports.carry(
        old,
        new,
        new_cp,
        field_changed=(
            (lambda _old, _new, field: transition.changed(field))
            if transition is not None
            else ports.field_changed
        ),
        parse_datetime=ports.parse_datetime,
        utc_to_local_naive=ports.utc_to_local_naive,
        local_naive_to_utc=ports.local_naive_to_utc,
        format_datetime=ports.format_datetime,
        carry_error=ports.carry_error,
    )
    decision = ports.workflow.decision_from_cp_adjustments(result)
    ports.workflow.apply_temporal_carry_patch(new, decision)
    ports.workflow.verify_temporal_carry_task(new, decision)
    return decision


def reject_native_until_carry(
    ports: NativeCarryPorts,
    old: TaskPayload,
    new: TaskPayload,
    new_target: datetime | None,
    old_target_field: str,
    exc: Exception,
) -> None:
    """Reject a target edit when its native expiration cannot be carried."""
    carry = None
    try:
        carry = ports.describe_carry(
            ports.parse_datetime(old.get("until")),
            ports.parse_datetime(old.get(old_target_field)),
            to_local=ports.to_local,
        )
    except Exception:
        pass
    target_label = (
        ports.format_local(new_target)
        if isinstance(new_target, datetime)
        else str(ports.anchor_field(new) or "–")
    )
    rows = [("Target", target_label), ("Required", str(exc))]
    if carry:
        rows.insert(1, ("Carry", carry))
    ports.panel("❌ Invalid expiration window", rows, kind="error")
    ports.abort(1)


def preserve_native_until_on_target_change(
    ports: NativePreservePorts,
    old: TaskPayload,
    new: TaskPayload,
    kind: str,
    *,
    transition: TaskTransition | None = None,
) -> Any:
    carried = ports.carry(
        old,
        new,
        kind,
        field_changed=(
            (lambda _old, _new, field: transition.changed(field))
            if transition is not None
            else ports.field_changed
        ),
        recurrence_anchor_field=ports.anchor_field,
        parse_datetime=ports.parse_datetime,
        native_until=ports.native_until,
        generation_service=ports.generation_service,
        reject_carry=ports.reject_carry,
        diagnostic=ports.diagnostic,
    )
    if not carried:
        return ports.workflow.NativeUntilDecision("unchanged")
    value = ports.parse_datetime(new.get("until"))
    if value is None:
        return ports.workflow.NativeUntilDecision(
            "rejected", reason="native-until carry produced no parseable value"
        )
    decision = ports.workflow.NativeUntilDecision(
        "carried", value=ports.timestamp(value)
    )
    ports.workflow.apply_native_until_patch(new, decision)
    ports.workflow.verify_native_until_task(new, decision)
    return decision


def native_preserve_ports_for(host: Any) -> NativePreservePorts:
    generation = host._module("modify_generation_effects")
    return NativePreservePorts(
        carry=host._module("modify_carry").preserve_native_until_on_target_change,
        field_changed=host._module("modify_task_fields").field_changed,
        anchor_field=host._module("modify_task_fields").recurrence_anchor_field,
        parse_datetime=host._TASK_DATETIME_PARSER.parse,
        native_until=host.core._import_sibling("native_until"),
        generation_service=lambda: generation.chain_generation_service(generation.generation_ports_for(host)),
        reject_carry=lambda *args: reject_native_until_carry(
            NativeCarryPorts(
                describe_carry=host.core._import_sibling("add_validation").describe_native_until_carry,
                parse_datetime=host._TASK_DATETIME_PARSER.parse,
                to_local=host.core.to_local, format_local=host.core.fmt_dt_local,
                anchor_field=host._module("modify_task_fields").recurrence_anchor_field,
                panel=lambda title, rows, **kwargs: host._module("modify_ui_effects").panel(host, title, rows, **kwargs),
                abort=host.sys.exit,
            ), *args
        ),
        diagnostic=host._diag,
        workflow=host._module("modify_carry_workflow"),
        timestamp=host._module("task_models").TaskTimestamp,
    )


def validate_completion_cp_and_anchor(
    host: Any,
    old: TaskPayload,
    new: TaskPayload,
    *,
    transition: TaskTransition | None = None,
) -> tuple[str, str, str]:
    modify_validation = host._module("modify_validation")
    modify_lifecycle = host._module("modify_lifecycle")
    validation_effects = host._module("modify_validation_effects")
    return modify_validation.validate_completion_cp_and_anchor(
        old,
        new,
        services=modify_validation.CompletionValidationServices(
            strip_quotes=host._module("modify_task_fields").strip_quotes,
            reject_conflicting_types=host.core._import_sibling("hook_validation_pipeline").reject_recurrence_kind_conflict,
            validate_omit=lambda anchor, anchor_file, omit, omit_file: validation_effects.validate_omit(host, anchor, anchor_file, omit, omit_file),
            validate_chain_limits=lambda task: validation_effects.validate_chain_limits(validation_effects.chain_limit_ports_for(host), task),
            parse_cp_sequence=host.core.parse_cp_sequence,
            cp_sequence_parse_error=host.core.cp_sequence_parse_error,
            field_changed=(
                (lambda _old, _new, field: transition.changed(field))
                if transition is not None
                else host._module("modify_task_fields").field_changed
            ),
            validate_anchor=lambda expr: validation_effects.validate_shared_anchor(
                validation_effects.SharedValidationPorts(
                    host.core._import_sibling("hook_validation_pipeline"),
                    host.core.parse_anchor_expr_to_dnf,
                    host._validate_anchor_expr_cached,
                    host._validate_omit_expr_cached,
                ), expr,
            ),
            validate_cp=lambda cp, chain_max, chain_until: validation_effects.validate_cp(
                validation_effects.CPValidationPorts(
                    host._module("modify_validation").validate_cp_on_modify,
                    host.core.parse_cp_sequence,
                    host.core.cp_sequence_parse_error,
                    host.core._import_sibling("add_validation").parse_chain_max,
                    lambda value: validation_effects.datetime_value(validation_effects.parser_for_host(host), value),
                ), cp, chain_max, chain_until,
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
