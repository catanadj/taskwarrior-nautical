"""Small value-level effects shared by the typed on-modify routes."""

from __future__ import annotations

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True, slots=True)
class DatetimePorts:
    compare: Callable[[datetime, datetime], int]


def compare_datetimes(ports: DatetimePorts, left: datetime, right: datetime) -> int:
    return ports.compare(left, right)


def format_delta(delta: timedelta) -> str:
    try:
        total = int(delta.total_seconds())
    except Exception:
        return str(delta)
    sign = "-" if total < 0 else "+"
    total = abs(total)
    total_minutes = total // 60
    days, remainder = divmod(total_minutes, 1440)
    hours, minutes = divmod(remainder, 60)
    return f"{sign}{days}d {hours:02}h:{minutes:02}m"


__all__ = ("compare_datetimes", "format_delta")
