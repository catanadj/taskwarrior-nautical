from __future__ import annotations

from datetime import date, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

from .astronomical_seasons import seasonal_event_utc


SEASON_NAMES = ("spring", "summer", "autumn", "winter")
HEMISPHERE_NAMES = ("north", "south")
SEASON_MODE_NAMES = ("fixed", "astronomical")

_FIXED_BOUNDARIES_BY_HEMISPHERE = {
    "north": {
    "spring": ((3, 1), (5, 31)),
    "summer": ((6, 1), (8, 31)),
    "autumn": ((9, 1), (11, 30)),
    "winter": ((12, 1), (2, 28)),
    },
    "south": {
        "spring": ((9, 1), (11, 30)),
        "summer": ((12, 1), (2, 28)),
        "autumn": ((3, 1), (5, 31)),
        "winter": ((6, 1), (8, 31)),
    },
}
_ACTIVE_HEMISPHERE = "north"
_ACTIVE_MODE = "fixed"
_ACTIVE_TIMEZONE: tzinfo = timezone.utc


def normalize_hemisphere(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in HEMISPHERE_NAMES:
        expected = ", ".join(HEMISPHERE_NAMES)
        raise ValueError(f"Unknown season hemisphere '{value}'. Expected one of: {expected}.")
    return normalized


def configure_hemisphere(value: object) -> str:
    """Set the process-wide fixed-season profile and return its canonical name."""
    global _ACTIVE_HEMISPHERE
    _ACTIVE_HEMISPHERE = normalize_hemisphere(value)
    import sys
    nc = sys.modules.get("nautical_core")
    if nc is not None:
        setattr(nc, "SEASON_HEMISPHERE", _ACTIVE_HEMISPHERE)
    return _ACTIVE_HEMISPHERE


def active_hemisphere() -> str:
    return _ACTIVE_HEMISPHERE


def normalize_season_mode(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in SEASON_MODE_NAMES:
        expected = ", ".join(SEASON_MODE_NAMES)
        raise ValueError(f"Unknown season mode '{value}'. Expected one of: {expected}.")
    return normalized


def configure_mode(value: object) -> str:
    """Set the season-boundary backend and return its canonical name."""
    global _ACTIVE_MODE
    _ACTIVE_MODE = normalize_season_mode(value)
    return _ACTIVE_MODE


def active_mode() -> str:
    return _ACTIVE_MODE


def configure_timezone(value: object) -> str:
    """Set the timezone used to turn UTC astronomical instants into dates."""
    global _ACTIVE_TIMEZONE
    if isinstance(value, tzinfo):
        _ACTIVE_TIMEZONE = value
        return str(getattr(value, "key", value))
    name = str(value or "").strip()
    if not name:
        raise ValueError("Astronomical season timezone cannot be empty.")
    try:
        _ACTIVE_TIMEZONE = ZoneInfo(name)
    except Exception as exc:
        raise ValueError(f"Astronomical season timezone '{name}' is invalid or unavailable.") from exc
    return name


def _fixed_boundaries() -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    return _FIXED_BOUNDARIES_BY_HEMISPHERE[_ACTIVE_HEMISPHERE]


def _fixed_boundary_description(season: str) -> str:
    descriptions = {
        "spring": "September 1 through November 30" if _ACTIVE_HEMISPHERE == "south" else "March 1 through May 31",
        "summer": "December 1 through February 28/29" if _ACTIVE_HEMISPHERE == "south" else "June 1 through August 31",
        "autumn": "March 1 through May 31" if _ACTIVE_HEMISPHERE == "south" else "September 1 through November 30",
        "winter": "June 1 through August 31" if _ACTIVE_HEMISPHERE == "south" else "December 1 through February 28/29",
    }
    return descriptions[season]


def normalize_season_name(value: object) -> str:
    """Return a canonical fixed-season name."""
    normalized = str(value or "").strip().lower()
    if normalized not in SEASON_NAMES:
        expected = ", ".join(SEASON_NAMES)
        raise ValueError(f"Unknown season '{value}'. Expected one of: {expected}.")
    return normalized


def fixed_season_boundary_description(season: object) -> str:
    """Return a concise description of one fixed seasonal window."""
    return _fixed_boundary_description(normalize_season_name(season))


_ASTRONOMICAL_EVENTS_BY_HEMISPHERE = {
    "north": {
        "spring": ("spring_equinox", "summer_solstice", 0, 0),
        "summer": ("summer_solstice", "autumn_equinox", 0, 0),
        "autumn": ("autumn_equinox", "winter_solstice", 0, 0),
        "winter": ("winter_solstice", "spring_equinox", 0, 1),
    },
    "south": {
        "spring": ("autumn_equinox", "winter_solstice", 0, 0),
        "summer": ("winter_solstice", "spring_equinox", 0, 1),
        "autumn": ("spring_equinox", "summer_solstice", 0, 0),
        "winter": ("summer_solstice", "autumn_equinox", 0, 0),
    },
}


def _astronomical_boundaries(name: str, start_year: int) -> tuple[date, date]:
    start_event, end_event, start_offset, end_offset = _ASTRONOMICAL_EVENTS_BY_HEMISPHERE[
        _ACTIVE_HEMISPHERE
    ][name]
    start_event_year = start_year + start_offset
    end_event_year = start_year + end_offset
    if end_event_year > 9999:
        raise ValueError(f"{name.capitalize()} starting in year {start_year} exceeds the supported date range.")
    start = seasonal_event_utc(start_event_year, start_event).astimezone(_ACTIVE_TIMEZONE).date()
    end_instant = seasonal_event_utc(end_event_year, end_event)
    # Season selectors operate on calendar dates.  The next transition's
    # local date starts the following season, even when its instant is later
    # in that day, so date windows never overlap.
    end = end_instant.astimezone(_ACTIVE_TIMEZONE).date() - timedelta(days=1)
    return start, end


def season_boundary_description(season: object) -> str:
    """Describe the active season backend without exposing implementation detail."""
    name = normalize_season_name(season)
    if _ACTIVE_MODE == "fixed":
        return fixed_season_boundary_description(name)
    start_event, end_event, _start_offset, _end_offset = _ASTRONOMICAL_EVENTS_BY_HEMISPHERE[
        _ACTIVE_HEMISPHERE
    ][name]
    return (
        f"{start_event.replace('_', ' ')} through "
        f"{end_event.replace('_', ' ')}"
    )


def season_bounds(season: object, start_year: int) -> tuple[date, date]:
    """Return inclusive boundaries identified by the season's start year."""
    name = normalize_season_name(season)
    if isinstance(start_year, bool) or not isinstance(start_year, int):
        raise TypeError("Season start year must be an integer.")
    if not 1 <= start_year <= 9999:
        raise ValueError("Season start year must be between 1 and 9999.")

    if _ACTIVE_MODE == "astronomical":
        return _astronomical_boundaries(name, start_year)

    (start_month, start_day), (end_month, end_day) = _fixed_boundaries()[name]
    crosses_year = end_month < start_month
    end_year = start_year + 1 if crosses_year else start_year
    if end_year > 9999:
        raise ValueError(f"{name.capitalize()} starting in year 9999 exceeds the supported date range.")
    end = (
        date(end_year, 3, 1) - timedelta(days=1)
        if crosses_year
        else date(end_year, end_month, end_day)
    )
    return (
        date(start_year, start_month, start_day),
        end,
    )


def season_window_on_or_after(season: object, value: date) -> tuple[date, date]:
    """Return the active season window, or the next one when outside that season."""
    name = normalize_season_name(season)
    if not isinstance(value, date):
        raise TypeError("Season window reference must be a date.")
    day = date(value.year, value.month, value.day)

    first_year = max(1, day.year - 1)
    last_year = min(9999, day.year + 1)
    for start_year in range(first_year, last_year + 1):
        try:
            start, end = season_bounds(name, start_year)
        except ValueError:
            continue
        if end >= day:
            return start, end
    raise OverflowError(f"No representable {name} season exists on or after {day.isoformat()}.")


def season_windows_on_or_after(value: date) -> tuple[str, date, date]:
    """Return the active or next fixed-season window chronologically."""
    if not isinstance(value, date):
        raise TypeError("Season window reference must be a date.")
    day = date(value.year, value.month, value.day)
    candidates: list[tuple[date, date, str]] = []
    first_year = max(1, day.year - 2)
    last_year = min(9999, day.year + 2)
    for start_year in range(first_year, last_year + 1):
        for name in SEASON_NAMES:
            try:
                start, end = season_bounds(name, start_year)
            except ValueError:
                continue
            candidates.append((start, end, name))
    candidates.sort(key=lambda item: item[0])
    for start, end, name in candidates:
        if start <= day <= end or start >= day:
            return name, start, end
    raise OverflowError(f"No representable fixed season exists on or after {day.isoformat()}.")


def next_season_window(start: date) -> tuple[str, date, date]:
    """Return the fixed-season window strictly after ``start``."""
    if not isinstance(start, date):
        raise TypeError("Season start must be a date.")
    probe = start + timedelta(days=1)
    while True:
        name, window_start, window_end = season_windows_on_or_after(probe)
        if window_start > start:
            return name, window_start, window_end
        probe = window_end + timedelta(days=1)


__all__ = (
    "HEMISPHERE_NAMES",
    "SEASON_NAMES",
    "SEASON_MODE_NAMES",
    "active_mode",
    "active_hemisphere",
    "configure_mode",
    "configure_hemisphere",
    "configure_timezone",
    "fixed_season_boundary_description",
    "next_season_window",
    "normalize_hemisphere",
    "normalize_season_mode",
    "normalize_season_name",
    "season_bounds",
    "season_boundary_description",
    "season_window_on_or_after",
    "season_windows_on_or_after",
)
