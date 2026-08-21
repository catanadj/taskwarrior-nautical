"""Small, dependency-free astronomical season boundary calculator.

This module is intentionally isolated from the scheduler.  It provides the
UTC instants at which the apparent solar longitude reaches the four cardinal
season points; the seasonal scheduler can adopt it only after its accuracy and
failure behavior have been validated independently.

The solar-longitude approximation follows the low-precision Meeus equations
used by the relevant ``solar_longitude_after`` path in PyCalCal.  PyCalCal is
MIT-licensed for its author-written portions (the project also contains
separately licensed Calendrica material); this module is a small, independent
stdlib-only adaptation rather than a vendored copy.  Results are suitable for
calendar boundaries and recurrence dates, not precision ephemeris work.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from math import radians, sin
from typing import Final


UTC = timezone.utc
SEASON_EVENT_NAMES: Final = (
    "spring_equinox",
    "summer_solstice",
    "autumn_equinox",
    "winter_solstice",
)
SEASON_EVENT_TARGETS: Final = {
    "spring_equinox": 0.0,
    "summer_solstice": 90.0,
    "autumn_equinox": 180.0,
    "winter_solstice": 270.0,
}
_APPROXIMATE_EVENT_DATES: Final = {
    "spring_equinox": (3, 20),
    "summer_solstice": (6, 21),
    "autumn_equinox": (9, 22),
    "winter_solstice": (12, 21),
}
_SEASON_TO_EVENT: Final = {
    "spring": "spring_equinox",
    "summer": "summer_solstice",
    "autumn": "autumn_equinox",
    "winter": "winter_solstice",
}


class AstronomicalSeasonError(ValueError):
    """Raised when a requested astronomical boundary cannot be computed."""


def _validate_year(year: object) -> int:
    if isinstance(year, bool) or not isinstance(year, int):
        raise TypeError("Astronomical season year must be an integer.")
    if not 1 <= year <= 9999:
        raise ValueError("Astronomical season year must be between 1 and 9999.")
    return year


def _normalize_event(event: object) -> str:
    name = str(event or "").strip().lower().replace("-", "_")
    if name not in SEASON_EVENT_TARGETS:
        expected = ", ".join(SEASON_EVENT_NAMES)
        raise AstronomicalSeasonError(
            f"Unknown astronomical season event '{event}'. Expected one of: {expected}."
        )
    return name


def _julian_day(moment: datetime) -> float:
    utc = moment.astimezone(UTC)
    seconds = (
        utc.hour * 3600
        + utc.minute * 60
        + utc.second
        + utc.microsecond / 1_000_000
    )
    # datetime.toordinal() is portable for the complete datetime year range,
    # unlike platform timestamps near years 1 and 9999.
    return utc.toordinal() + 1721424.5 + seconds / 86400.0


def solar_longitude(moment: datetime) -> float:
    """Return apparent solar longitude in degrees for an aware datetime."""
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise TypeError("Solar-longitude calculations require an aware datetime.")
    t = (_julian_day(moment) - 2451545.0) / 36525.0
    mean_longitude = 280.46646 + 36000.76983 * t + 0.0003032 * t * t
    mean_anomaly = 357.52911 + 35999.05029 * t - 0.0001537 * t * t
    anomaly = radians(mean_anomaly)
    center = (
        (1.914602 - 0.004817 * t - 0.000014 * t * t) * sin(anomaly)
        + (0.019993 - 0.000101 * t) * sin(2.0 * anomaly)
        + 0.000289 * sin(3.0 * anomaly)
    )
    true_longitude = mean_longitude + center
    omega = radians(125.04 - 1934.136 * t)
    apparent_longitude = true_longitude - 0.00569 - 0.00478 * sin(omega)
    return apparent_longitude % 360.0


def _signed_delta(longitude: float, target: float) -> float:
    """Return the shortest signed angular distance from target."""
    return (longitude - target + 180.0) % 360.0 - 180.0


def _shift_clamped(moment: datetime, delta: timedelta) -> datetime:
    try:
        candidate = moment + delta
    except OverflowError:
        return (
            datetime.min.replace(tzinfo=UTC)
            if delta < timedelta(0)
            else datetime.max.replace(tzinfo=UTC)
        )
    if candidate < datetime.min.replace(tzinfo=UTC):
        return datetime.min.replace(tzinfo=UTC)
    if candidate > datetime.max.replace(tzinfo=UTC):
        return datetime.max.replace(tzinfo=UTC)
    return candidate


def _find_crossing(year: int, event: str) -> datetime:
    month, day = _APPROXIMATE_EVENT_DATES[event]
    center = datetime(year, month, day, 12, tzinfo=UTC)
    # The approximation remains a deliberately broad calendar window.  A
    # scan gives us an explicit exhaustion error instead of fabricating a date.
    start = _shift_clamped(center, timedelta(days=-45))
    end = _shift_clamped(center, timedelta(days=45))
    target = SEASON_EVENT_TARGETS[event]
    step = timedelta(hours=6)
    previous = start
    previous_delta = _signed_delta(solar_longitude(previous), target)
    if previous_delta == 0.0:
        return previous
    cursor = start + step
    while cursor <= end:
        current_delta = _signed_delta(solar_longitude(cursor), target)
        if current_delta == 0.0:
            return cursor
        if previous_delta < 0.0 <= current_delta:
            lo, hi = previous, cursor
            for _ in range(64):
                middle = lo + (hi - lo) / 2
                middle_delta = _signed_delta(solar_longitude(middle), target)
                if middle_delta < 0.0:
                    lo = middle
                else:
                    hi = middle
            return hi.astimezone(UTC)
        previous, previous_delta = cursor, current_delta
        if cursor >= end:
            break
        cursor += step
    raise AstronomicalSeasonError(
        f"No {event.replace('_', ' ')} crossing found near {year}; "
        "the astronomical search window was exhausted."
    )


@lru_cache(maxsize=128)
def _seasonal_events_utc_cached(year: int) -> tuple[tuple[str, datetime], ...]:
    return tuple((event, _find_crossing(year, event)) for event in SEASON_EVENT_NAMES)


def seasonal_event_utc(year: int, event: object) -> datetime:
    """Return one named season boundary as an aware UTC datetime."""
    valid_year = _validate_year(year)
    name = _normalize_event(event)
    return dict(_seasonal_events_utc_cached(valid_year))[name]


def seasonal_events_utc(year: int) -> dict[str, datetime]:
    """Return all four cardinal season boundaries for ``year`` in UTC."""
    valid_year = _validate_year(year)
    return dict(_seasonal_events_utc_cached(valid_year))


def season_boundary_utc(year: int, season: object) -> datetime:
    """Return the astronomical boundary corresponding to a season name."""
    name = str(season or "").strip().lower()
    try:
        event = _SEASON_TO_EVENT[name]
    except KeyError as exc:
        expected = ", ".join(_SEASON_TO_EVENT)
        raise AstronomicalSeasonError(
            f"Unknown astronomical season '{season}'. Expected one of: {expected}."
        ) from exc
    return seasonal_event_utc(year, event)


__all__ = (
    "AstronomicalSeasonError",
    "SEASON_EVENT_NAMES",
    "SEASON_EVENT_TARGETS",
    "season_boundary_utc",
    "seasonal_event_utc",
    "seasonal_events_utc",
    "solar_longitude",
)
