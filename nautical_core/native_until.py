from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable


CARRY_CONFLICT = "calendar_target_conflict"
CARRY_INVALID = "invalid_carry_input"
CARRY_FAILED = "carry_failed"


class NativeUntilCarryError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code or CARRY_FAILED)


def uses_exact_carry(until_local: Any) -> bool:
    """Return whether the stored +1s expiration marker requests exact carry."""
    try:
        return int(until_local.second) == 1
    except Exception:
        return False


def describe_carry(
    until_dt: Any,
    target_dt: Any,
    *,
    to_local: Callable[[Any], Any],
) -> str | None:
    """Describe the expiration carry policy represented by one occurrence."""
    if until_dt is None or target_dt is None:
        return None
    try:
        until_local = to_local(until_dt)
        target_local = to_local(target_dt)
        if uses_exact_carry(until_local):
            seconds = max(0, int((until_dt - target_dt).total_seconds()))
            days, seconds = divmod(seconds, 86400)
            hours, seconds = divmod(seconds, 3600)
            minutes, seconds = divmod(seconds, 60)
            parts: list[str] = []
            if days:
                parts.append(f"{days}d")
            if hours or parts:
                parts.append(f"{hours:02d}h")
            if minutes or parts:
                parts.append(f"{minutes:02d}m")
            parts.append(f"{seconds:02d}s" if parts else f"{seconds}s")
            return "Exact · " + " ".join(parts) + " after occurrence"

        day_gap = (until_local.date() - target_local.date()).days
        clock = f"{until_local.hour:02d}:{until_local.minute:02d}"
        if until_local.second:
            clock += f":{until_local.second:02d}"
        if day_gap == 0:
            return f"Same day at {clock}"
        if day_gap == 1:
            return f"1 calendar day later at {clock}"
        return f"{day_gap} calendar days later at {clock}"
    except Exception:
        return None


def validate_after_target(
    until_dt: Any,
    target_dt: Any,
    target_field: str,
) -> tuple[bool, str | None]:
    if until_dt is None or target_dt is None:
        return (True, None)
    label = "scheduled" if str(target_field or "").strip().lower() == "scheduled" else "due"
    try:
        if until_dt <= target_dt:
            return (False, f"until must be later than {label}")
    except Exception:
        return (False, f"until and {label} could not be compared")
    return (True, None)


def validate_calendar_slots(
    until_dt: Any,
    target_dt: Any,
    slots: Any,
    *,
    to_local: Callable[[Any], Any],
) -> tuple[bool, str | None]:
    """Reject same-day calendar expirations that are not later than every fixed slot."""
    if until_dt is None or target_dt is None:
        return (True, None)
    try:
        until_local = to_local(until_dt)
        if uses_exact_carry(until_local):
            return (True, None)
        target_local = to_local(target_dt)
        if until_local.date() > target_local.date():
            return (True, None)
        expiration = (until_local.hour, until_local.minute, until_local.second)
        blocked = sorted(
            (int(hh), int(mm))
            for hh, mm in slots or ()
            if (int(hh), int(mm), 0) >= expiration
        )
    except Exception:
        return (False, "could not compare calendar expiration with anchor times")
    if not blocked:
        return (True, None)
    hh, mm = blocked[0]
    return (False, f"calendar expiration must be later than anchor slot {hh:02d}:{mm:02d}")


def carry(
    parent_target: datetime,
    parent_until: datetime,
    child_target: datetime,
    kind: str,
    *,
    utc_to_local_naive: Callable[[datetime], datetime],
    local_naive_to_utc: Callable[[datetime], datetime],
) -> datetime:
    """Carry native expiration to a recurrence target using the configured policy."""
    if not all(isinstance(value, datetime) for value in (parent_target, parent_until, child_target)):
        raise NativeUntilCarryError(CARRY_INVALID, "native until carry requires valid recurrence timestamps")
    try:
        parent_until_local = utc_to_local_naive(parent_until)
        if uses_exact_carry(parent_until_local):
            child_until = child_target + (parent_until - parent_target)
        else:
            parent_target_local = utc_to_local_naive(parent_target)
            child_target_local = utc_to_local_naive(child_target)
            day_gap = (parent_until_local.date() - parent_target_local.date()).days
            child_until_local = datetime.combine(
                child_target_local.date() + timedelta(days=day_gap),
                parent_until_local.time(),
            )
            child_until = local_naive_to_utc(child_until_local)
            if str(kind or "").strip().lower() == "cp" and child_until <= child_target:
                child_until = local_naive_to_utc(child_until_local + timedelta(days=1))
    except NativeUntilCarryError:
        raise
    except Exception as exc:
        raise NativeUntilCarryError(CARRY_FAILED, "native until carry could not be calculated") from exc

    if child_until <= child_target:
        raise NativeUntilCarryError(
            CARRY_CONFLICT,
            "native until must be later than the child recurrence target",
        )
    return child_until


__all__ = (
    "CARRY_CONFLICT",
    "CARRY_FAILED",
    "CARRY_INVALID",
    "NativeUntilCarryError",
    "carry",
    "describe_carry",
    "uses_exact_carry",
    "validate_after_target",
    "validate_calendar_slots",
)
