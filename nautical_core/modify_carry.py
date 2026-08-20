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


__all__ = ("preserve_cp_relative_offsets_on_due_change",)
