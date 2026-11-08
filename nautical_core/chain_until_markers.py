"""Resolve symbolic chain-until markers to concrete UTC timestamps."""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta, timezone, tzinfo
from . import season_support
from .astronomical_seasons import seasonal_event_utc
from .timeutil import fmt_isoz, local_naive_to_utc, to_local


_HOLIDAY_NAMES = (
    "christmas", "christmas-eve", "new-year", "new-years-eve", "halloween",
    "valentines-day", "easter", "good-friday", "easter-monday",
)
_SEASON_NAMES = (
    "spring-start", "spring-end", "summer-start", "summer-end",
    "autumn-start", "autumn-end", "winter-start", "winter-end",
)
_PERIOD_NAMES = (
    "month-start", "month-end", "quarter-start", "quarter-end",
    "year-start", "year-end",
)
_EVENT_NAMES = (
    "spring-equinox", "summer-solstice", "autumn-equinox", "winter-solstice",
)


def _easter_sunday(year: int) -> date:
    """Return Gregorian Easter Sunday for ``year``."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _parse_concrete(value: str) -> bool:
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        parsed = None
    if parsed is not None:
        return True
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue
    return False


def _next_annual(reference: date, month: int, day: int) -> date:
    year = reference.year
    candidate = date(year, month, day)
    return candidate if candidate >= reference else date(year + 1, month, day)


def _holiday_date(marker: str, reference: date) -> date:
    if marker == "easter":
        candidate = _easter_sunday(reference.year)
        return candidate if candidate >= reference else _easter_sunday(reference.year + 1)
    easter = _easter_sunday(reference.year)
    if marker == "good-friday":
        candidate = easter - timedelta(days=2)
        if candidate < reference:
            easter = _easter_sunday(reference.year + 1)
            candidate = easter - timedelta(days=2)
        return candidate
    if marker == "easter-monday":
        candidate = easter + timedelta(days=1)
        if candidate < reference:
            candidate = _easter_sunday(reference.year + 1) + timedelta(days=1)
        return candidate
    annual = {
        "christmas": (12, 25),
        "christmas-eve": (12, 24),
        "new-year": (1, 1),
        "new-years-eve": (12, 31),
        "halloween": (10, 31),
        "valentines-day": (2, 14),
    }
    return _next_annual(reference, *annual[marker])


def _period_date(marker: str, reference: date) -> date:
    if marker.startswith("month-"):
        if marker == "month-start":
            candidate = reference.replace(day=1)
            if candidate < reference:
                month = 1 if reference.month == 12 else reference.month + 1
                year = reference.year + 1 if reference.month == 12 else reference.year
                return date(year, month, 1)
            return candidate
        last = calendar.monthrange(reference.year, reference.month)[1]
        candidate = date(reference.year, reference.month, last)
        if candidate < reference:
            month = 1 if reference.month == 12 else reference.month + 1
            year = reference.year + 1 if reference.month == 12 else reference.year
            return date(year, month, calendar.monthrange(year, month)[1])
        return candidate
    if marker.startswith("quarter-"):
        start_month = ((reference.month - 1) // 3) * 3 + 1
        if marker == "quarter-start":
            candidate = date(reference.year, start_month, 1)
            if candidate < reference:
                start_month += 3
                year = reference.year + (start_month > 12)
                start_month = ((start_month - 1) % 12) + 1
                return date(year, start_month, 1)
            return candidate
        end_month = start_month + 2
        candidate = date(reference.year, end_month, calendar.monthrange(reference.year, end_month)[1])
        if candidate < reference:
            start_month += 3
            year = reference.year + (start_month > 12)
            start_month = ((start_month - 1) % 12) + 1
            end_month = start_month + 2
            return date(year, end_month, calendar.monthrange(year, end_month)[1])
        return candidate
    if marker == "year-start":
        candidate = date(reference.year, 1, 1)
        return candidate if candidate >= reference else date(reference.year + 1, 1, 1)
    candidate = date(reference.year, 12, 31)
    return candidate if candidate >= reference else date(reference.year + 1, 12, 31)


def _season_date(marker: str, reference: date) -> date:
    season, boundary = marker.rsplit("-", 1)
    candidates: list[date] = []
    for year in range(max(1, reference.year - 1), min(9999, reference.year + 2) + 1):
        start, end = season_support.season_bounds(season, year)
        candidates.append(start if boundary == "start" else end)
    return min(candidate for candidate in candidates if candidate >= reference)


def _event_date(marker: str, reference: date, local_tz: tzinfo | None) -> date:
    event = marker.replace("-", "_")
    for year in range(reference.year, min(9999, reference.year + 2) + 1):
        moment = seasonal_event_utc(year, event)
        candidate = to_local(moment, local_tz or timezone.utc).date()
        if candidate >= reference:
            return candidate
    raise ValueError(f"No future date is available for chainUntil marker '{marker}'")


def resolve_chain_until(
    value: object,
    *,
    reference: datetime,
    local_tz: tzinfo | None = None,
) -> str | None:
    """Resolve a concrete timestamp or symbolic marker to canonical UTC text."""
    text = str(value or "").strip()
    if not text:
        return None
    if _parse_concrete(text):
        return text
    if not isinstance(reference, datetime) or reference.tzinfo is None:
        raise TypeError("chainUntil marker resolution requires an aware reference datetime")
    marker = text.lower().replace("_", "-")
    local = to_local(reference, local_tz or timezone.utc)
    if marker in _HOLIDAY_NAMES:
        resolved = _holiday_date(marker, local.date())
    elif marker in _SEASON_NAMES:
        resolved = _season_date(marker, local.date())
    elif marker in _PERIOD_NAMES:
        resolved = _period_date(marker, local.date())
    elif marker in _EVENT_NAMES:
        resolved = _event_date(marker, local.date(), local_tz)
    else:
        families = ", ".join(("holiday", "season", "calendar period", "astronomical event"))
        raise ValueError(
            f"Unrecognized datetime format: Unknown chainUntil marker '{text}'. "
            f"Use a {families} marker."
        )
    local_end = datetime(resolved.year, resolved.month, resolved.day, 23, 59, 59)
    return fmt_isoz(local_naive_to_utc(local_end, local_tz or timezone.utc))


__all__ = ("resolve_chain_until",)
