"""Mutation-specific validation used by the on-modify hook."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any


@dataclass(slots=True)
class CompletionValidationServices:
    strip_quotes: Callable[[str], str]
    reject_conflicting_types: Callable[[str, str, str], None]
    validate_omit: Callable[[str, str, str, str], None]
    validate_chain_limits: Callable[[dict[str, Any]], None]
    parse_cp_sequence: Callable[[str], Any]
    cp_sequence_parse_error: Callable[[str], str | None]
    field_changed: Callable[[dict[str, Any], dict[str, Any], str], bool]
    validate_anchor: Callable[[str], None]
    validate_cp: Callable[[str, Any, Any], None]
    apply_transition: Callable[[dict[str, Any], dict[str, Any]], None]
    fail: Callable[[str, str], Any]
    diagnostic: Callable[[str], None]


def validate_completion_cp_and_anchor(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    services: CompletionValidationServices,
) -> tuple[str, str, str]:
    """Validate recurrence fields while a task is completing."""
    new_cp = services.strip_quotes(str(new.get("cp") or "").strip())
    new_anchor = services.strip_quotes(str(new.get("anchor") or "").strip())
    new_anchor_file = services.strip_quotes(str(new.get("anchor_file") or "").strip())
    if new_anchor_file:
        new["anchor_file"] = new_anchor_file
    new_omit = services.strip_quotes(str(new.get("omit") or "").strip())
    if new_omit:
        new["omit"] = new_omit
    new_omit_file = services.strip_quotes(str(new.get("omit_file") or "").strip())
    if new_omit_file:
        new["omit_file"] = new_omit_file

    services.reject_conflicting_types(new_anchor, new_anchor_file, new_cp)
    services.validate_omit(new_anchor, new_anchor_file, new_omit, new_omit_file)
    if new_cp or new_anchor or new_anchor_file:
        services.validate_chain_limits(new)

    if new_cp:
        try:
            sequence = services.parse_cp_sequence(new_cp)
            if not sequence:
                reason = services.cp_sequence_parse_error(new_cp) or f"invalid duration format '{new_cp}'"
                raise ValueError(reason)
        except ValueError as exc:
            services.fail("Invalid CP", str(exc))
        except Exception as exc:
            services.diagnostic(f"cp parse unexpected error: {exc}")
            services.fail("CP parsing error", "Unexpected error while parsing cp")

        if (
            services.field_changed(old, new, "anchor")
            or services.field_changed(old, new, "anchor_mode")
            or services.field_changed(old, new, "anchor_file")
        ) and new_anchor:
            services.validate_anchor(new_anchor)

        if (
            services.field_changed(old, new, "cp")
            or services.field_changed(old, new, "chainMax")
            or services.field_changed(old, new, "chainUntil")
        ):
            services.validate_cp(new_cp, new.get("chainMax"), new.get("chainUntil"))

        try:
            services.apply_transition(old, new)
        except Exception as exc:
            services.fail(
                "Nautical recurrence activation failed",
                f"Nautical recurrence transition failed: {type(exc).__name__}: {exc}",
            )

    return new_cp, new_anchor, new_anchor_file


def validate_anchor_on_modify(
    expr: str,
    *,
    parse_anchor_expr: Any,
    validate_anchor_expr: Any,
) -> None:
    """Mirror strict on-add anchor checks for ordinary modifications."""
    if not expr or not expr.strip():
        raise ValueError("anchor is required if chaining by anchor")
    try:
        parse_anchor_expr(expr)
    except Exception as exc:
        raise ValueError(f"anchor syntax error: {exc}") from exc
    try:
        validate_anchor_expr(expr)
    except Exception as exc:
        raise ValueError(f"anchor validation failed: {exc}") from exc


def validate_omit_on_modify(expr: str, *, validate_omit_expr: Any) -> None:
    if not expr or not expr.strip():
        return
    try:
        validate_omit_expr(expr)
    except Exception as exc:
        raise ValueError(f"omit validation failed: {exc}") from exc


def validate_cp_on_modify(
    cp_value: str,
    chain_max_value: Any,
    chain_until_value: Any,
    *,
    parse_cp_sequence: Any,
    cp_sequence_parse_error: Any,
    parse_chain_max: Any,
    parse_datetime: Any,
) -> None:
    """Validate a CP value and its optional chain limits."""
    if not cp_value or not cp_value.strip():
        return
    sequence = parse_cp_sequence(cp_value)
    if not sequence:
        reason = cp_sequence_parse_error(cp_value) or f"invalid duration format '{cp_value}'"
        raise ValueError(f"{reason} (expected: 3d, 2w, 1h, etc.)")
    _cpmax, chain_max_error = parse_chain_max(chain_max_value)
    if chain_max_error:
        raise ValueError(chain_max_error)
    chain_until = (chain_until_value or "").strip()
    if chain_until and parse_datetime(chain_until) is None:
        raise ValueError(f"Invalid chainUntil '{chain_until}'")


def validate_chain_limits_on_modify(
    task: dict[str, Any],
    *,
    parse_chain_max: Any,
    parse_datetime: Any,
    validate_until_not_past: Any,
    now_utc: Any,
    fail: Any,
) -> None:
    """Normalize and validate chainMax/chainUntil during modification."""
    cpmax, chain_max_error = parse_chain_max(task.get("chainMax"))
    if chain_max_error:
        fail("Invalid chainMax", chain_max_error)
        return
    if cpmax is not None:
        task["chainMax"] = cpmax
    chain_until = str(task.get("chainUntil") or "").strip()
    if not chain_until:
        return
    until_dt = parse_datetime(chain_until)
    if until_dt is None:
        fail("Invalid chainUntil", f"Unrecognized datetime format '{chain_until}'")
        return
    is_valid, until_error = validate_until_not_past(until_dt, now_utc())
    if not is_valid:
        fail("Invalid chainUntil", until_error or "chainUntil is in the past")


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
    "validate_anchor_on_modify",
    "validate_chain_limits_on_modify",
    "validate_cp_on_modify",
    "validate_native_until_after_target_or_fail",
    "validate_native_until_anchor_slots_or_fail",
    "validate_omit_on_modify",
)
