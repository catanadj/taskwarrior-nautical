"""Datetime parsing effects shared by the typed on-modify workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from .task_datetime import TaskDatetimeParser


def parse_datetime(
    parser: TaskDatetimeParser,
    value: object,
) -> tuple[datetime | None, str | None]:
    """Parse through the shared datetime port."""
    return parser.parse(value)


def safe_dt(host: Any, value: Any) -> datetime | None:
    try:
        return host._dtparse(value) if isinstance(value, str) else value
    except Exception:
        return None


def utc_to_local_naive(host: Any, value: datetime) -> datetime:
    if not hasattr(value, "tzinfo"):
        raise TypeError("dt_utc must be datetime")
    return host.core.utc_to_local_naive(value)


def local_naive_to_utc(host: Any, value: datetime) -> datetime:
    if not hasattr(value, "tzinfo"):
        raise TypeError("dt_local_naive must be datetime")
    return host.core.local_naive_to_utc(value.replace(microsecond=0))


__all__ = ("parse_datetime", "safe_dt", "utc_to_local_naive", "local_naive_to_utc")
