"""Timezone resolution boundary for the compatibility facade."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def resolve(
    timezone_name: str,
    zoneinfo_module: Any,
    warn_once: Callable[[str, str], Any],
) -> tuple[Any, str]:
    """Resolve a configured timezone without mutating facade globals."""
    if zoneinfo_module is None:
        message = "timezone support unavailable (zoneinfo import failed)"
        warn_once(
            "timezone_zoneinfo_unavailable",
            "[nautical] timezone support unavailable (zoneinfo import failed); using UTC fallback.",
        )
        return None, message
    try:
        return zoneinfo_module.ZoneInfo(timezone_name), ""
    except Exception:
        message = f"configured timezone '{timezone_name}' is invalid or unavailable"
        warn_once(
            "timezone_local_invalid",
            f"[nautical] timezone '{timezone_name}' is invalid/unavailable; using UTC fallback.",
        )
        return None, message


__all__ = ("resolve",)
