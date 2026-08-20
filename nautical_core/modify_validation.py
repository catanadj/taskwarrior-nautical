"""Mutation-specific validation used by the on-modify hook."""

from __future__ import annotations

from typing import Any


def validate_native_until_after_target_or_fail(
    task: dict[str, Any],
    *,
    validate_anchor_mode: Any,
    safe_parse_datetime: Any,
    validate_after_target: Any,
    format_local: Any,
    panel: Any,
    fail: Any,
    abort: Any,
) -> None:
    """Reject native expiration windows that cannot contain the target."""
    until_raw = task.get("until")
    if not until_raw:
        return
    mode_is_valid, mode_reason = validate_anchor_mode(
        until_raw,
        task.get("anchor"),
        task.get("anchor_file"),
        task.get("anchor_mode"),
    )
    if not mode_is_valid:
        mode = str(task.get("anchor_mode") or "skip").strip().lower()
        panel(
            "❌ Invalid expiration mode",
            [
                ("Mode", mode),
                ("Conflict", mode_reason or "Native until conflicts with strict anchor backfill."),
                ("Action", "Remove until or use anchor_mode:skip."),
            ],
            kind="error",
        )
        abort(1)
        return

    target_field = "due" if task.get("due") else "scheduled" if task.get("scheduled") else ""
    if not target_field:
        return
    target_dt, target_err = safe_parse_datetime(task.get(target_field))
    if target_err or target_dt is None:
        fail(f"Invalid {target_field}", target_err or f"{target_field} must be a valid datetime")
        return
    until_dt, until_err = safe_parse_datetime(until_raw)
    if until_err or until_dt is None:
        fail("Invalid until", until_err or "until must be a valid datetime")
        return
    is_valid, reason = validate_after_target(until_dt, target_dt, target_field)
    if is_valid:
        return
    label = "Scheduled" if target_field == "scheduled" else "Due"
    panel(
        "❌ Invalid expiration window",
        [
            (label, format_local(target_dt)),
            ("Expires", format_local(until_dt)),
            ("Required", reason or f"until must be later than {target_field}"),
        ],
        kind="error",
    )
    abort(1)


def validate_native_until_anchor_slots_or_fail(
    task: dict[str, Any],
    *,
    safe_parse_datetime: Any,
    validate_anchor: Any,
    collect_time_slots: Any,
    normalize_time_slots: Any,
    anchor_file_dir: str,
    recurrence_context: Any,
    to_local: Any,
    format_local: Any,
    astronomy_is_error: Any,
    astronomy_error_message: Any,
    panel: Any,
    abort: Any,
) -> None:
    """Reject native expiration windows before every timed anchor slot."""
    until_raw = task.get("until")
    anchor_value = str(task.get("anchor") or "").strip()
    anchor_file_value = str(task.get("anchor_file") or "").strip()
    if not until_raw or not (anchor_value or anchor_file_value):
        return
    target_field = "due" if task.get("due") else "scheduled" if task.get("scheduled") else ""
    if not target_field:
        return
    target_dt, target_err = safe_parse_datetime(task.get(target_field))
    until_dt, until_err = safe_parse_datetime(until_raw)
    if target_err or until_err or target_dt is None or until_dt is None:
        return
    dnf = None
    if anchor_value:
        try:
            dnf = validate_anchor(anchor_value)
        except Exception:
            return
    target_local = to_local(target_dt)
    try:
        slots = collect_time_slots(
            dnf,
            anchor_file_value,
            (target_local.hour, target_local.minute),
            normalize_time_slots=normalize_time_slots,
            anchor_file_dir=anchor_file_dir,
            target_date=target_local.date(),
            resolve_time_slots=lambda value, target_date: normalize_time_slots(value, target_date),
            recurrence_context=recurrence_context(task),
        )
    except Exception as exc:
        if astronomy_is_error(exc):
            panel(
                "❌ Invalid astronomy time",
                [("Required", astronomy_error_message(exc))],
                kind="error",
            )
            abort(1)
        return
    is_valid, reason = validate_time_slots(
        until_dt,
        target_dt,
        slots,
        to_local=to_local,
    )
    if is_valid:
        return
    panel(
        "❌ Invalid expiration window",
        [
            ("Expires", format_local(until_dt)),
            ("Anchor slots", ", ".join(f"{hh:02d}:{mm:02d}" for hh, mm in slots) or "none"),
            ("Required", reason or "calendar expiration must be later than every anchor slot"),
        ],
        kind="error",
    )
    abort(1)


__all__ = (
    "validate_native_until_after_target_or_fail",
    "validate_native_until_anchor_slots_or_fail",
)
