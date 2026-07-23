from __future__ import annotations

from datetime import date, timedelta


SEASON_NAMES = ("spring", "summer", "autumn", "winter")

_FIXED_BOUNDARIES = {
    "spring": ((3, 1), (5, 31)),
    "summer": ((6, 1), (8, 31)),
    "autumn": ((9, 1), (11, 30)),
    "winter": ((12, 1), (2, 28)),
}
_FIXED_BOUNDARY_DESCRIPTIONS = {
    "spring": "March 1 through May 31",
    "summer": "June 1 through August 31",
    "autumn": "September 1 through November 30",
    "winter": "December 1 through February 28/29",
}


def normalize_season_name(value: object) -> str:
    """Return a canonical fixed-season name."""
    normalized = str(value or "").strip().lower()
    if normalized not in _FIXED_BOUNDARIES:
        expected = ", ".join(SEASON_NAMES)
        raise ValueError(f"Unknown season '{value}'. Expected one of: {expected}.")
    return normalized


def fixed_season_boundary_description(season: object) -> str:
    """Return a concise description of one fixed seasonal window."""
    return _FIXED_BOUNDARY_DESCRIPTIONS[normalize_season_name(season)]


def season_bounds(season: object, start_year: int) -> tuple[date, date]:
    """Return fixed inclusive boundaries, identified by the season's start year."""
    name = normalize_season_name(season)
    if isinstance(start_year, bool) or not isinstance(start_year, int):
        raise TypeError("Season start year must be an integer.")
    if not 1 <= start_year <= 9999:
        raise ValueError("Season start year must be between 1 and 9999.")

    (start_month, start_day), (end_month, end_day) = _FIXED_BOUNDARIES[name]
    end_year = start_year + 1 if name == "winter" else start_year
    if end_year > 9999:
        raise ValueError("Winter starting in year 9999 exceeds the supported date range.")
    end = (
        date(end_year, 3, 1) - timedelta(days=1)
        if name == "winter"
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
