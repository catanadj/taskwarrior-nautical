"""Timezone resolution boundary for the compatibility facade."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_local_timezone: Any = None
_configuration_error = ""


def resolve(
    timezone_name: str,
    zoneinfo_module: Any,
    warn_once: Callable[[str, str], Any],
) -> tuple[Any, str]:
    """Resolve and retain timezone state in its owning module."""
    global _local_timezone, _configuration_error
    if zoneinfo_module is None:
        message = "timezone support unavailable (zoneinfo import failed)"
        warn_once(
            "timezone_zoneinfo_unavailable",
            "[nautical] timezone support unavailable (zoneinfo import failed); using UTC fallback.",
        )
        _local_timezone, _configuration_error = None, message
        return None, message
    try:
        _local_timezone = zoneinfo_module.ZoneInfo(timezone_name)
        _configuration_error = ""
        return _local_timezone, ""
    except Exception:
        message = f"configured timezone '{timezone_name}' is invalid or unavailable"
        warn_once(
            "timezone_local_invalid",
            f"[nautical] timezone '{timezone_name}' is invalid/unavailable; using UTC fallback.",
        )
        _local_timezone, _configuration_error = None, message
        return None, message


def current_timezone() -> Any:
    """Return the currently resolved timezone, if available."""
    return _local_timezone


def configuration_error() -> str:
    """Return the most recent timezone-resolution diagnostic."""
    return _configuration_error


__all__ = ("configuration_error", "current_timezone", "resolve")
