"""Shared normalization and resolution for anchor ``@t=`` values."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Callable

from . import astronomy
from .time_windows import parse_time_window_spec
from .time_windows import parse_random_time_window_spec


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
        parsed_slots: list[tuple[int, int] | None] = [_parse_hhmm(part) for part in value.split(",")]
        parsed: list[tuple[int, int]] = []
        for slot in parsed_slots:
            if slot is not None:
                parsed.append(slot)
        return [_apply_offset(slot, offset_minutes) for slot in parsed] if offset_minutes else parsed

    return []


def resolve_time_slots_with_offsets(
    value: Any,
    target_date: date,
    *,
    config: dict[str, Any] | None = None,
    to_local: Callable[[Any], Any] | None = None,
    seed_base: str = "",
) -> list[tuple[int, int, int]]:
    """Resolve slots as ``(day_offset, hour, minute)`` values.

    This preserves the date ownership of overnight windows while retaining
    the ordinary resolver's behavior for clocks and astronomical events.
    """
    offset_minutes = 0
    raw = value
    if isinstance(raw, dict):
        offset_minutes = int(raw.get("time_offset_minutes", 0) or 0)
        raw = raw.get("t")
        window_spec = value.get("time_window")
        if isinstance(window_spec, str):
            window = parse_time_window_spec(window_spec)
            if window is not None:
                slots = list(window.slots_with_offsets)
                if offset_minutes:
                    adjusted = []
                    for day_offset, hour, minute in slots:
                        total = day_offset * 1440 + hour * 60 + minute + offset_minutes
                        adjusted.append((total // 1440, (total % 1440) // 60, total % 60))
                    return adjusted
                return slots
        random_spec = value.get("time_random")
        if isinstance(random_spec, str):
            random_window = parse_random_time_window_spec(random_spec)
            if random_window is None:
                return []
            if not seed_base:
                raise ValueError("Random time windows require a stable chain seed.")
            return list(random_window.slots_with_offsets(f"{seed_base}/{target_date.isoformat()}"))
    ordinary = resolve_time_slots(raw, target_date, config=config, to_local=to_local)
    if offset_minutes:
        ordinary = resolve_time_slots({"t": ordinary, "time_offset_minutes": offset_minutes}, target_date, config=config, to_local=to_local)
    return [(0, hour, minute) for hour, minute in ordinary]


__all__ = ("resolve_time_slots", "resolve_time_slots_with_offsets")
