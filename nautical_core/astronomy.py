"""Optional astronomical event resolution for symbolic anchor times."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from typing import Any

EVENT_NAMES = frozenset({"sunrise", "sunset", "dawn", "dusk", "moonrise", "moonset"})
ASTRONOMICAL_TIMES = EVENT_NAMES


class AstronomyUnavailableError(RuntimeError):
    """Raised when symbolic times are used without the optional provider."""


class AstronomyConfigurationError(ValueError):
    """Raised when an astronomy location profile is incomplete."""


def is_event_name(value: Any) -> bool:
    return str(value or "").strip().lower() in EVENT_NAMES


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
    "is_event_name",
    "resolve_event",
)
