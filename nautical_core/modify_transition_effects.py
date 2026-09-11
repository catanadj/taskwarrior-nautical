"""Typed carry and recurrence-transition effects for on-modify."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .task_changes import TaskTransition
from .task_models import TaskPayload


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


@dataclass(frozen=True, slots=True)
class CompletionValidationPorts:
    validate: Any
    services_type: Any
    strip_quotes: Any
    reject_conflicting_types: Any
    validate_omit: Any
    validate_chain_limits: Any
    parse_cp_sequence: Any
    cp_sequence_parse_error: Any
    field_changed: Any
    validate_anchor: Any
    validate_cp: Any
    apply_transition: Any
    fail: Any
    diagnostic: Any


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


def validate_completion_cp_and_anchor(
    ports: CompletionValidationPorts,
    old: TaskPayload,
    new: TaskPayload,
    *,
    transition: TaskTransition | None = None,
) -> tuple[str, str, str]:
    return ports.validate(
        old,
        new,
        services=ports.services_type(
            strip_quotes=ports.strip_quotes,
            reject_conflicting_types=ports.reject_conflicting_types,
            validate_omit=ports.validate_omit,
            validate_chain_limits=ports.validate_chain_limits,
            parse_cp_sequence=ports.parse_cp_sequence,
            cp_sequence_parse_error=ports.cp_sequence_parse_error,
            field_changed=(
                (lambda _old, _new, field: transition.changed(field))
                if transition is not None
                else ports.field_changed
            ),
            validate_anchor=ports.validate_anchor,
            validate_cp=ports.validate_cp,
            apply_transition=ports.apply_transition,
            fail=ports.fail,
            diagnostic=ports.diagnostic,
        ),
    )


__all__ = (
    "NativeCarryPorts",
    "CPCarryPorts",
    "NativePreservePorts",
    "CompletionValidationPorts",
    "preserve_cp_relative_offsets_on_due_change",
    "reject_native_until_carry",
    "preserve_native_until_on_target_change",
    "validate_completion_cp_and_anchor",
)
