"""Carry-forward calculations used by ordinary recurrence edits."""

from __future__ import annotations

from typing import Any


def preserve_cp_relative_offsets_on_due_change(
    old: dict[str, Any],
    new: dict[str, Any],
    new_cp: str,
    *,
    field_changed: Any,
    parse_datetime: Any,
    utc_to_local_naive: Any,
    local_naive_to_utc: Any,
    format_datetime: Any,
    carry_error: Any,
) -> tuple[Any, Any, list[tuple[str, Any, Any, Any]]] | None:
    """Keep scheduled/wait relative to due when a CP task's due moves."""
    if not new_cp or not str(old.get("cp") or "").strip():
        return None
    if not field_changed(old, new, "due") or not (old.get("due") and new.get("due")):
        return None

    try:
        old_due = parse_datetime(old.get("due"))
        new_due = parse_datetime(new.get("due"))
        if not (old_due and new_due):
            raise ValueError("due timestamp is missing or invalid")
    except Exception as exc:
        raise carry_error("due", str(exc) or "timestamp conversion failed") from exc

    adjustments: list[tuple[str, Any, Any, Any]] = []
    for field in ("scheduled", "wait"):
        if field_changed(old, new, field) or not old.get(field):
            continue
        try:
            old_value = parse_datetime(old.get(field))
            if not old_value:
                raise ValueError("timestamp is missing or invalid")
            local_offset = utc_to_local_naive(old_value) - utc_to_local_naive(old_due)
            new_value_local = utc_to_local_naive(new_due) + local_offset
            new_value = local_naive_to_utc(new_value_local)
            new[field] = format_datetime(new_value)
            adjustments.append((field, old_value, new_value, local_offset))
        except Exception as exc:
            raise carry_error(field, str(exc) or "timezone conversion failed") from exc
    return (old_due, new_due, adjustments) if adjustments else None


def preserve_native_until_on_target_change(
    old: dict[str, Any],
    new: dict[str, Any],
    kind: str,
    *,
    field_changed: Any,
    recurrence_anchor_field: Any,
    parse_datetime: Any,
    native_until: Any,
    generation_service: Any,
    reject_carry: Any,
    diagnostic: Any,
) -> bool:
    """Carry an untouched native until when an existing recurrence target moves."""
    if field_changed(old, new, "until") or not old.get("until"):
        return False
    old_target_field = recurrence_anchor_field(old)
    new_target_field = recurrence_anchor_field(new)
    target_changed = old_target_field != new_target_field or field_changed(old, new, old_target_field)
    if not target_changed:
        return False

    new_target = None
    try:
        new_target = parse_datetime(new.get(new_target_field))
        if not new_target:
            raise native_until.NativeUntilCarryError(
                native_until.CARRY_INVALID,
                f"{new_target_field} timestamp is missing or invalid",
            )
        candidate = dict(new)
        from nautical_core.task_codec import DEFAULT_TASK_CODEC
        from nautical_core.task_models import NauticalTask
        parent_task = NauticalTask.from_observation(
            DEFAULT_TASK_CODEC.decode_row(old, source_query="ordinary-edit native-until carry")
        )
        generation_service().carry_native_until(
            parent_task,
            candidate,
            new_target,
            kind,
            parent_anchor_field=old_target_field,
            child_anchor_field=new_target_field,
        )
        carried = candidate.get("until")
        if not carried:
            raise native_until.NativeUntilCarryError(
                native_until.CARRY_FAILED,
                "native until carry produced no expiration value",
            )
        new["until"] = carried
        return True
    except native_until.NativeUntilCarryError as exc:
        reject_carry(old, new, new_target, old_target_field, exc)
    except Exception as exc:
        diagnostic(f"native until target carry failed: {exc}")
        typed_error = native_until.NativeUntilCarryError(
            native_until.CARRY_FAILED,
            f"native until target carry failed: {type(exc).__name__}: {exc}",
        )
        reject_carry(old, new, new_target, old_target_field, typed_error)
    return False


__all__ = (
    "preserve_cp_relative_offsets_on_due_change",
    "preserve_native_until_on_target_change",
)
