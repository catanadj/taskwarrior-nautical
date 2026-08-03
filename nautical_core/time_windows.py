"""Parsing and expansion for bounded repeated clock-time windows."""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import file_resource_limits as resource_limits


_WINDOW_RE = re.compile(
    r"^(?P<start>(?:[01]\d|2[0-3])(?::[0-5]\d)?)\.\."
    r"(?P<end>(?:[01]\d|2[0-3])(?::[0-5]\d)?)/"
    r"(?P<interval>(?:(?:\d+)h)?(?:(?:\d+)m)?)$"
)
_DURATION_RE = re.compile(r"^(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?$")


def _parse_clock(value: str) -> tuple[int, int]:
    if ":" not in value:
        return int(value), 0
    hour, minute = value.split(":", 1)
    return int(hour), int(minute)


def _clock_minutes(value: tuple[int, int]) -> int:
    return value[0] * 60 + value[1]


@dataclass(frozen=True)
class TimeWindow:
    """A same-day inclusive clock range with a fixed minute interval."""

    start: tuple[int, int]
    end: tuple[int, int]
    interval_minutes: int

    @property
    def slots(self) -> tuple[tuple[int, int], ...]:
        start = _clock_minutes(self.start)
        end = _clock_minutes(self.end)
        values = tuple(
            (minute // 60, minute % 60)
            for minute in range(start, end + 1, self.interval_minutes)
        )
        limit = int(resource_limits.MAX_TIME_WINDOW_SLOTS)
        if len(values) > limit:
            raise ValueError(
                f"Time window produces too many slots ({len(values)}); "
                f"increase the interval or keep it below {limit} slots."
            )
        return values

    @property
    def canonical(self) -> str:
        return (
            f"{self.start[0]:02d}:{self.start[1]:02d}.."
            f"{self.end[0]:02d}:{self.end[1]:02d}/{_format_duration(self.interval_minutes)}"
        )


def _format_duration(minutes: int) -> str:
    hours, remainder = divmod(minutes, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if remainder:
        parts.append(f"{remainder}m")
    return "".join(parts)


def parse_time_window_spec(value: str) -> TimeWindow | None:
    """Parse ``HH[:MM]..HH[:MM]/interval`` or return ``None`` for ordinary times."""
    text = str(value or "").strip().lower()
    if ".." not in text or "/" not in text:
        return None
    match = _WINDOW_RE.fullmatch(text)
    if not match:
        raise ValueError(
            "Invalid time window. Use HH[:MM]..HH[:MM]/interval, for example 06..18/3h."
        )

    start = _parse_clock(match.group("start"))
    end = _parse_clock(match.group("end"))
    start_minutes = _clock_minutes(start)
    end_minutes = _clock_minutes(end)
    if start_minutes >= end_minutes:
        raise ValueError("Invalid time window: the end time must be later than the start time.")

    interval_text = match.group("interval")
    duration = _DURATION_RE.fullmatch(interval_text)
    if not duration or not interval_text:
        raise ValueError("Invalid time window interval: use a positive duration such as 30m or 2h.")
    interval_minutes = int(duration.group("hours") or 0) * 60 + int(duration.group("minutes") or 0)
    if interval_minutes <= 0:
        raise ValueError("Invalid time window interval: it must be greater than zero.")
    span = end_minutes - start_minutes
    if interval_minutes > span:
        raise ValueError("Invalid time window interval: it cannot exceed the window span.")

    window = TimeWindow(start, end, interval_minutes)
    _ = window.slots
    return window


def validate_time_window_slots(spec: str, slots: object) -> None:
    """Reject serialized window metadata whose expanded slots no longer agree."""
    window = parse_time_window_spec(spec)
    if window is None:
        raise ValueError("Invalid cached time window metadata.")
    if not isinstance(slots, (list, tuple)):
        raise ValueError("Cached time window slots must be a list.")
    normalized = []
    for value in slots:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("Cached time window slots contain an invalid clock value.")
        try:
            normalized.append((int(value[0]), int(value[1])))
        except (TypeError, ValueError):
            raise ValueError("Cached time window slots contain an invalid clock value.") from None
    if tuple(normalized) != window.slots:
        raise ValueError("Cached time window metadata does not match its expanded slots.")


__all__ = ("TimeWindow", "parse_time_window_spec", "validate_time_window_slots")
