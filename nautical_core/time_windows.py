"""Parsing and expansion for bounded repeated clock-time windows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from . import file_resource_limits as resource_limits


_WINDOW_RE = re.compile(
    r"^(?P<start>(?:[01]\d|2[0-3])(?::[0-5]\d)?)\.\."
    r"(?P<end>(?:[01]\d|2[0-3])(?::[0-5]\d)?)/"
    r"(?P<interval>(?:\d+|(?:(?:\d+(?:\.\d+)?)h(?:(?:\d+)(?:m|min))?|(?:\d+)(?:m|min))))$"
)
_DURATION_RE = re.compile(
    r"^(?:(?P<hours>\d+(?:\.\d+)?)h(?:(?P<minutes>\d+)(?:m|min))?|"
    r"(?P<minutes_only>\d+)(?:m|min))$"
)
_CLOCK_TOKEN_RE = re.compile(r"^(?:[01]\d|2[0-3])(?::[0-5]\d)?$")


def _parse_clock(value: str) -> tuple[int, int]:
    if ":" not in value:
        return int(value), 0
    hour, minute = value.split(":", 1)
    return int(hour), int(minute)


def parse_clock_value(value: str) -> tuple[int, int] | None:
    """Parse a strict ``HH`` or ``HH:MM`` clock token."""
    text = str(value or "").strip()
    if not _CLOCK_TOKEN_RE.fullmatch(text):
        return None
    return _parse_clock(text)


def _clock_minutes(value: tuple[int, int]) -> int:
    return value[0] * 60 + value[1]


@dataclass(frozen=True)
class TimeWindow:
    """A same-day inclusive clock range with a duration or slot count."""

    start: tuple[int, int]
    end: tuple[int, int]
    interval_minutes: int | None = None
    partition_count: int | None = None

    @property
    def crosses_midnight(self) -> bool:
        return self.end < self.start

    @property
    def slots_with_offsets(self) -> tuple[tuple[int, int, int], ...]:
        """Expand slots as ``(day_offset, hour, minute)`` values."""
        start = _clock_minutes(self.start)
        end = _clock_minutes(self.end) + (24 * 60 if self.crosses_midnight else 0)
        span = end - start
        if self.partition_count is not None:
            divisor = self.partition_count - 1
            minutes = tuple(start + (span * index + divisor // 2) // divisor for index in range(self.partition_count))
        else:
            interval = self.interval_minutes or 0
            minutes = tuple(range(start, end + 1, interval))
        limit = int(resource_limits.MAX_TIME_WINDOW_SLOTS)
        if len(minutes) > limit:
            raise ValueError(
                f"Time window produces too many slots ({len(minutes)}); "
                f"increase the interval or keep it below {limit} slots."
            )
        return tuple((minute // 1440, (minute % 1440) // 60, minute % 60) for minute in minutes)

    @property
    def slots(self) -> tuple[tuple[int, int], ...]:
        return tuple((hour, minute) for _offset, hour, minute in self.slots_with_offsets)

    @property
    def canonical(self) -> str:
        return (
            f"{self.start[0]:02d}:{self.start[1]:02d}.."
            f"{self.end[0]:02d}:{self.end[1]:02d}/"
            f"{self.partition_count if self.partition_count is not None else _format_duration(self.interval_minutes or 0)}"
        )


@dataclass(frozen=True)
class TimeSchedule:
    """A deduplicated union of numeric windows and fixed clock slots."""

    canonical: str
    slots: tuple[tuple[int, int], ...]


def _format_duration(minutes: int) -> str:
    hours, remainder = divmod(minutes, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if remainder:
        parts.append(f"{remainder}m")
    return "".join(parts)


def _parse_duration_minutes(value: str) -> int:
    """Parse a clock interval into exact whole minutes."""
    match = _DURATION_RE.fullmatch(value)
    if not match:
        raise ValueError("Invalid time window interval: use a positive duration such as 30m, 3h30min, or 3.5h.")
    try:
        if match.group("minutes_only") is not None:
            total = Decimal(match.group("minutes_only"))
        else:
            hours = Decimal(match.group("hours") or "0")
            minutes = Decimal(match.group("minutes") or "0")
            if hours != hours.to_integral_value() and minutes:
                raise ValueError("Invalid time window interval: do not combine decimal hours with minutes.")
            total = hours * 60 + minutes
    except InvalidOperation:
        raise ValueError("Invalid time window interval: use a positive duration such as 30m, 3h30min, or 3.5h.") from None
    if total != total.to_integral_value():
        raise ValueError("Invalid time window interval: decimal hours must resolve to whole minutes.")
    return int(total)


def parse_time_window_spec(value: str) -> TimeWindow | None:
    """Parse ``HH[:MM]..HH[:MM]/interval`` or return ``None`` for ordinary times."""
    text = str(value or "").strip().lower()
    if "," in text or ".." not in text or "/" not in text:
        return None
    match = _WINDOW_RE.fullmatch(text)
    if not match:
        raise ValueError(
            "Invalid time window. Use HH[:MM]..HH[:MM]/interval, for example 06..18/3h, 06..18/4, or 04:30..19:30/3h30min."
        )

    start = _parse_clock(match.group("start"))
    end = _parse_clock(match.group("end"))
    start_minutes = _clock_minutes(start)
    end_minutes = _clock_minutes(end)
    if start_minutes == end_minutes:
        raise ValueError("Invalid time window: the end time must differ from the start time.")

    interval_text = match.group("interval")
    span = (end_minutes - start_minutes) % (24 * 60)
    if re.fullmatch(r"\d+", interval_text):
        partition_count = int(interval_text)
        if partition_count < 2:
            raise ValueError("Invalid time window partition count: use at least 2 total slots.")
        if partition_count > span + 1:
            raise ValueError("Invalid time window partition count: it cannot create duplicate minute slots.")
        window = TimeWindow(start, end, partition_count=partition_count)
    else:
        duration = _DURATION_RE.fullmatch(interval_text)
        if not duration or not interval_text:
            raise ValueError("Invalid time window interval: use a positive duration such as 30m, 3h30min, or 3.5h, or use a unitless slot count such as /4.")
        interval_minutes = _parse_duration_minutes(interval_text)
        if interval_minutes <= 0:
            raise ValueError("Invalid time window interval: it must be greater than zero.")
        if interval_minutes > span:
            raise ValueError("Invalid time window interval: it cannot exceed the window span.")
        window = TimeWindow(start, end, interval_minutes=interval_minutes)
    _ = window.slots
    return window


def parse_time_schedule_spec(value: str) -> TimeSchedule | None:
    """Parse a comma-separated union containing at least one time window."""
    text = str(value or "").strip().lower()
    if ".." not in text:
        return None
    raw_parts = text.split(",")
    if any(not part.strip() for part in raw_parts):
        raise ValueError(
            "Invalid composable time schedule. Remove empty comma-separated members."
        )
    parts = [part.strip() for part in raw_parts]
    if not parts:
        return None
    members: list[tuple[int, int, str]] = []
    seen_members: set[str] = set()
    slots: set[tuple[int, int]] = set()
    has_window = False
    for part in parts:
        window = parse_time_window_spec(part)
        if window is not None:
            has_window = True
            if window.canonical not in seen_members:
                members.append((_clock_minutes(window.start), 0, window.canonical))
                seen_members.add(window.canonical)
            slots.update(window.slots)
            continue
        clock = parse_clock_value(part)
        if clock is None:
            raise ValueError(
                "Invalid composable time schedule. Use numeric windows and clocks, "
                "for example 06..18/3h,22."
            )
        canonical = f"{clock[0]:02d}:{clock[1]:02d}"
        if canonical not in seen_members:
            members.append((_clock_minutes(clock), 1, canonical))
            seen_members.add(canonical)
        slots.add(clock)
    if not has_window:
        return None
    limit = int(resource_limits.MAX_TIME_WINDOW_SLOTS)
    if len(slots) > limit:
        raise ValueError(
            f"Time schedule produces too many slots ({len(slots)}); "
            f"increase the interval or keep it below {limit} slots."
        )
    members.sort(key=lambda item: (item[0], item[1], item[2]))
    return TimeSchedule(",".join(item[2] for item in members), tuple(sorted(slots)))


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


def validate_time_window_offsets(spec: str, offsets: object) -> None:
    """Reject serialized day-offset metadata that disagrees with a window."""
    window = parse_time_window_spec(spec)
    if window is None:
        raise ValueError("Invalid cached time window metadata.")
    if not isinstance(offsets, (list, tuple)):
        raise ValueError("Cached time window offsets must be a list.")
    normalized = []
    for value in offsets:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("Cached time window offsets contain an invalid slot.")
        try:
            normalized.append((int(value[0]), int(value[1]), int(value[2])))
        except (TypeError, ValueError):
            raise ValueError("Cached time window offsets contain an invalid slot.") from None
    if tuple(normalized) != window.slots_with_offsets:
        raise ValueError("Cached time window offset metadata does not match its expanded slots.")


def validate_time_schedule_slots(spec: str, slots: object) -> None:
    """Reject serialized schedule metadata whose expanded slots no longer agree."""
    schedule = parse_time_schedule_spec(spec)
    if schedule is None:
        raise ValueError("Invalid cached time schedule metadata.")
    if not isinstance(slots, (list, tuple)):
        raise ValueError("Cached time schedule slots must be a list.")
    normalized = []
    for value in slots:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("Cached time schedule slots contain an invalid clock value.")
        try:
            normalized.append((int(value[0]), int(value[1])))
        except (TypeError, ValueError):
            raise ValueError("Cached time schedule slots contain an invalid clock value.") from None
    if tuple(normalized) != schedule.slots:
        raise ValueError("Cached time schedule metadata does not match its expanded slots.")


__all__ = (
    "TimeSchedule",
    "TimeWindow",
    "parse_clock_value",
    "parse_time_schedule_spec",
    "parse_time_window_spec",
    "validate_time_schedule_slots",
    "validate_time_window_offsets",
    "validate_time_window_slots",
)
