"""Datetime parsing effects shared by the typed on-modify workflow."""

from __future__ import annotations

from typing import Any


def safe_parse_datetime(host: Any, dt_str: str) -> tuple[Any | None, str | None]:
    """Parse a Taskwarrior datetime without allowing malformed input to escape."""
    if not (dt_str or "").strip():
        return None, None
    try:
        dt = host.core.parse_dt_any(dt_str)
        if dt is None:
            return None, f"Unrecognized datetime format '{dt_str}'"
        return dt, None
    except ValueError as exc:
        host._diag(f"datetime parse value error: {exc}")
        return None, "DateTime parsing error"
    except TypeError as exc:
        host._diag(f"datetime parse type error: {exc}")
        return None, "DateTime type error"
    except Exception as exc:
        host._diag(f"datetime parse unexpected error: {exc}")
        return None, "Unexpected error parsing datetime"


def safe_dt(host: Any, value: Any):
    try:
        return host._dtparse(value) if isinstance(value, str) else value
    except Exception:
        return None


def utc_to_local_naive(host: Any, value):
    if not hasattr(value, "tzinfo"):
        raise TypeError("dt_utc must be datetime")
    return host.core.utc_to_local_naive(value)


def local_naive_to_utc(host: Any, value):
    if not hasattr(value, "tzinfo"):
        raise TypeError("dt_local_naive must be datetime")
    return host.core.local_naive_to_utc(value.replace(microsecond=0))


__all__ = ("safe_parse_datetime", "safe_dt", "utc_to_local_naive", "local_naive_to_utc")
