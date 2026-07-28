"""Optional astronomical event resolution for symbolic anchor times."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

from .moon_phase import PHASES, canonical_phase

EVENT_NAMES = frozenset({"sunrise", "sunset", "dawn", "dusk", "moonrise", "moonset"})
ASTRONOMICAL_TIMES = EVENT_NAMES
PHASE_TARGETS = {"new": 0.0, "first-quarter": 7.0, "full": 14.0, "last-quarter": 21.0}
PHASE_RANGES = {
    "new": (0.0, 6.99),
    "first-quarter": (7.0, 13.99),
    "full": (14.0, 20.99),
    "last-quarter": (21.0, 27.99),
}


class AstronomyUnavailableError(RuntimeError):
    """Raised when symbolic times are used without the optional provider."""


class AstronomyConfigurationError(ValueError):
    """Raised when an astronomy location profile is incomplete."""


def scheduling_error_message(exc: BaseException) -> str:
    """Return an actionable, stable message for astronomy scheduling failures."""
    text = str(exc).strip() or type(exc).__name__
    if isinstance(exc, AstronomyUnavailableError):
        return f"Astronomy provider unavailable: {text}. Install astral or remove the moon-based recurrence."
    if isinstance(exc, AstronomyConfigurationError):
        return f"Astronomy profile invalid: {text}. Configure [astronomy] before using moon recurrence."
    return text


def is_event_name(value: Any) -> bool:
    return str(value or "").strip().lower() in EVENT_NAMES


def _phase_distance(value: float, target: float) -> float:
    """Distance on Astral's circular 0..28 phase scale."""
    distance = abs(float(value) - float(target))
    return min(distance, 28.0 - distance)


def _phase_matches(value: float, phase: str) -> bool:
    """Return whether Astral's 0..27.99 phase age is in a named band."""
    bounds = PHASE_RANGES[phase]
    return bounds[0] <= float(value) <= bounds[1]


def _timezone_for_profile(config: dict[str, Any] | None, name: str | None = None) -> tuple[str, str]:
    selected, profile = _profile(config, name)
    timezone = str(profile.get("timezone") or "").strip()
    if not timezone:
        raise AstronomyConfigurationError(f"astronomy location '{selected}' requires an explicit timezone")
    try:
        from zoneinfo import ZoneInfo

        ZoneInfo(timezone)
    except Exception as exc:
        raise AstronomyConfigurationError(
            f"astronomy location '{selected}' has invalid timezone '{timezone}'"
        ) from exc
    return selected, timezone


def resolve_phase_date(
    phase: str,
    reference_day: date,
    *,
    config: dict[str, Any] | None = None,
    location_name: str | None = None,
    horizon_days: int = 60,
) -> date:
    """Return the first local calendar date in the requested phase band."""
    name = canonical_phase(phase)
    if name is None:
        raise ValueError(f"unknown moon phase '{phase}'")
    if not isinstance(reference_day, date):
        raise TypeError("reference_day must be a date")
    if horizon_days < 28:
        raise ValueError("moon phase lookup horizon must cover at least one lunar cycle")
    selected, timezone = _timezone_for_profile(config, location_name)
    return _resolve_phase_date_cached(name, reference_day, selected, timezone, int(horizon_days))


@lru_cache(maxsize=256)
def _resolve_phase_date_cached(
    phase: str,
    reference_day: date,
    selected: str,
    timezone: str,
    horizon_days: int,
) -> date:
    try:
        from astral import moon
    except ImportError as exc:
        raise AstronomyUnavailableError(
            "moon phase anchors require the optional 'astral' package"
        ) from exc
    start = reference_day + timedelta(days=1)
    for offset in range(horizon_days):
        day = start + timedelta(days=offset)
        try:
            value = float(moon.phase(day))
        except (TypeError, ValueError, AttributeError) as exc:
            raise LookupError(
                f"moon phase '{phase}' is unavailable on {day.isoformat()} at {selected} ({timezone})"
            ) from exc
        if _phase_matches(value, phase):
            return day
    raise LookupError(f"moon phase '{phase}' has no matching date within {horizon_days} days")


def _profile(config: dict[str, Any] | None, name: str | None = None) -> tuple[str, dict[str, Any]]:
    data = config if isinstance(config, dict) else {}
    locations = data.get("locations") if isinstance(data.get("locations"), dict) else {}
    selected = str(name or data.get("default_location") or "").strip()
    if not selected and len(locations) == 1:
        selected = str(next(iter(locations)))
    profile = locations.get(selected) if selected else None
    if not isinstance(profile, dict):
        raise AstronomyConfigurationError(
            "astronomy location is not configured; define [astronomy] default_location "
            "and [astronomy.locations.<name>]"
        )
    return selected, profile


def _observer(config: dict[str, Any] | None, location_name: str | None = None):
    selected, profile = _profile(config, location_name)
    try:
        latitude = float(profile["latitude"])
        longitude = float(profile["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AstronomyConfigurationError(
            f"astronomy location '{selected}' requires numeric latitude and longitude"
        ) from exc
    timezone = str(profile.get("timezone") or "").strip()
    if not timezone:
        raise AstronomyConfigurationError(f"astronomy location '{selected}' requires an explicit timezone")
    try:
        from zoneinfo import ZoneInfo

        tzinfo = ZoneInfo(timezone)
    except Exception as exc:
        raise AstronomyConfigurationError(
            f"astronomy location '{selected}' has invalid timezone '{timezone}'"
        ) from exc
    try:
        from astral import Observer
    except ImportError as exc:
        raise AstronomyUnavailableError(
            "astronomical anchor times require the optional 'astral' package"
        ) from exc
    return selected, Observer(latitude=latitude, longitude=longitude, elevation=float(profile.get("elevation", 0) or 0)), tzinfo


def resolve_event(event: str, day: date, *, config: dict[str, Any] | None = None, location_name: str | None = None) -> datetime:
    """Resolve one event as a timezone-aware local datetime."""
    name = str(event or "").strip().lower()
    if name not in EVENT_NAMES:
        raise ValueError(f"unknown astronomical event '{event}'")
    selected, profile = _profile(config, location_name)
    try:
        latitude = float(profile["latitude"])
        longitude = float(profile["longitude"])
        elevation = float(profile.get("elevation", 0) or 0)
    except (KeyError, TypeError, ValueError) as exc:
        raise AstronomyConfigurationError(
            f"astronomy location '{selected}' requires numeric latitude and longitude"
        ) from exc
    timezone = str(profile.get("timezone") or "").strip()
    if not timezone:
        raise AstronomyConfigurationError(f"astronomy location '{selected}' requires an explicit timezone")
    return _resolve_event_cached(name, day, selected, latitude, longitude, elevation, timezone)


@lru_cache(maxsize=512)
def _resolve_event_cached(
    name: str,
    day: date,
    selected: str,
    latitude: float,
    longitude: float,
    elevation: float,
    timezone: str,
) -> datetime:
    """Resolve repeated hook/reconcile lookups without retaining mutable config."""
    try:
        from zoneinfo import ZoneInfo
        tzinfo = ZoneInfo(timezone)
        from astral import Observer
        observer = Observer(latitude=latitude, longitude=longitude, elevation=elevation)
    except ImportError as exc:
        raise AstronomyUnavailableError(
            "astronomical anchor times require the optional 'astral' package"
        ) from exc
    except Exception as exc:
        raise AstronomyConfigurationError(
            f"astronomy location '{selected}' has invalid timezone '{timezone}'"
        ) from exc
    try:
        from astral import moon
        from astral import sun

        if name in {"sunrise", "sunset", "dawn", "dusk"}:
            value = getattr(sun, name)(observer, date=day, tzinfo=tzinfo)
        else:
            value = getattr(moon, name)(observer, date=day, tzinfo=tzinfo)
    except (ValueError, AttributeError) as exc:
        raise LookupError(f"astronomical event '{name}' is unavailable on {day.isoformat()} at {selected}") from exc
    return value


__all__ = (
    "ASTRONOMICAL_TIMES",
    "AstronomyConfigurationError",
    "AstronomyUnavailableError",
    "EVENT_NAMES",
    "PHASE_RANGES",
    "is_event_name",
    "resolve_event",
    "resolve_phase_date",
    "scheduling_error_message",
)
