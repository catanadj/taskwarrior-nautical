"""Shared normalization and resolution for anchor ``@t=`` values."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Callable

from . import astronomy


_HHMM_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


def _apply_offset(hhmm: tuple[int, int], offset_minutes: int) -> tuple[int, int]:
    minute = (hhmm[0] * 60 + hhmm[1] + int(offset_minutes or 0)) % (24 * 60)
    return minute // 60, minute % 60


def _parse_hhmm(value: str) -> tuple[int, int] | None:
    token = str(value or "").strip()
    if not _HHMM_RE.fullmatch(token):
        return None
    hour, minute = token.split(":", 1)
    return int(hour), int(minute)


def resolve_time_slots(
    value: Any,
    target_date: date,
    *,
    config: dict[str, Any] | None = None,
    to_local: Callable[[Any], Any] | None = None,
) -> list[tuple[int, int]]:
    """Resolve numeric or astronomical time values to local ``(hour, minute)`` slots.

    ``value`` may be a parsed modifier dictionary, a single time/event, or a
    list of either. Astronomical events are resolved through the shared
    astronomy provider and then converted with the caller's local-time policy.
    """
    offset_minutes = 0
    if isinstance(value, dict):
        offset_minutes = int(value.get("time_offset_minutes", 0) or 0)
        value = value.get("t")

    if isinstance(value, list):
        slots: list[tuple[int, int]] = []
        for item in value:
            slots.extend(resolve_time_slots(item, target_date, config=config, to_local=to_local))
        return [_apply_offset(slot, offset_minutes) for slot in slots] if offset_minutes else slots

    if isinstance(value, tuple) and len(value) == 2:
        try:
            slot = (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return []
        return [_apply_offset(slot, offset_minutes)]

    if isinstance(value, str):
        token = value.strip().lower()
        if token in astronomy.EVENT_NAMES:
            # Preview and parsing paths may normalize a modifier before a
            # concrete calendar date exists; defer astronomical resolution.
            if not isinstance(target_date, date):
                return []
            event = astronomy.resolve_event(token, target_date, config=config or {})
            local = to_local(event) if to_local is not None else event
            slot = (int(local.hour), int(local.minute))
            return [_apply_offset(slot, offset_minutes)]
        slots = [_parse_hhmm(part) for part in value.split(",")]
        parsed = [slot for slot in slots if slot is not None]
        return [_apply_offset(slot, offset_minutes) for slot in parsed] if offset_minutes else parsed

    return []


__all__ = ("resolve_time_slots",)
