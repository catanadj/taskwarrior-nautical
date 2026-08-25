"""Shared astronomy-aware validation for native expiration windows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def validate_native_until_slots(
    *,
    until_dt: Any,
    target_dt: Any,
    dnf: Any,
    anchor_file_value: str,
    fallback_hhmm: tuple[int, int],
    collect_time_slots: Callable[..., Any],
    normalize_time_slots: Callable[..., Any],
    resolve_time_slots: Callable[..., Any] | None,
    anchor_file_dir: str,
    recurrence_context: Any,
    to_local: Callable[[Any], Any],
    validate_time_slots: Callable[..., tuple[bool, str | None]],
) -> tuple[bool, str | None, tuple[tuple[int, int], ...]]:
    """Resolve every effective astronomical slot and validate expiration."""
    if until_dt is None or target_dt is None or not (dnf or str(anchor_file_value or "").strip()):
        return True, None, ()
    slots = collect_time_slots(
        dnf,
        anchor_file_value,
        fallback_hhmm,
        normalize_time_slots=normalize_time_slots,
        anchor_file_dir=anchor_file_dir,
        target_date=to_local(target_dt).date(),
        resolve_time_slots=resolve_time_slots,
        recurrence_context=recurrence_context,
    )
    valid, reason = validate_time_slots(until_dt, target_dt, slots, to_local=to_local)
    return bool(valid), reason, tuple((int(hh), int(mm)) for hh, mm in slots or ())


__all__ = ("validate_native_until_slots",)
