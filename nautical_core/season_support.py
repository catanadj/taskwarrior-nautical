from __future__ import annotations

from datetime import date, timedelta


SEASON_NAMES = ("spring", "summer", "autumn", "winter")
HEMISPHERE_NAMES = ("north", "south")

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
    return _ACTIVE_HEMISPHERE


def active_hemisphere() -> str:
    return _ACTIVE_HEMISPHERE


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


def season_bounds(season: object, start_year: int) -> tuple[date, date]:
    """Return fixed inclusive boundaries, identified by the season's start year."""
    name = normalize_season_name(season)
    if isinstance(start_year, bool) or not isinstance(start_year, int):
        raise TypeError("Season start year must be an integer.")
    if not 1 <= start_year <= 9999:
        raise ValueError("Season start year must be between 1 and 9999.")

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
