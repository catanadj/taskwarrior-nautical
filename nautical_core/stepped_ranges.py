"""Shared parsing and expansion helpers for stepped calendar-date ranges."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import re
from typing import Iterator


_STEP_SUFFIX_RE = re.compile(r"^(?P<range>.+?)/(?P<days>[1-9]\d*)d$")
MAX_STEP_DAYS = 366


@dataclass(frozen=True, slots=True)
class SteppedRange:
    """A range token with a positive calendar-day stride."""

    range_text: str
    step_days: int


def parse_step_suffix(token: str) -> SteppedRange | None:
    """Split a ``/Nd`` suffix, rejecting malformed step suffixes."""
    text = str(token or "").strip().lower()
    if "/" not in text:
        return None
    match = _STEP_SUFFIX_RE.fullmatch(text)
    if not match or ".." not in match.group("range"):
        raise ValueError("stepped date ranges require a positive '/Nd' suffix")
    step_days = int(match.group("days"))
    if step_days > MAX_STEP_DAYS:
        raise ValueError(f"stepped date range step cannot exceed {MAX_STEP_DAYS} days")
    return SteppedRange(match.group("range"), step_days)


def iter_stepped_dates(start: date, end: date, step_days: int) -> Iterator[date]:
    """Yield inclusive dates from ``start`` through ``end`` at ``step_days``."""
    if step_days < 1 or end < start:
        return
    current = start
    stride = timedelta(days=step_days)
    while current <= end:
        yield current
        current += stride


__all__ = ("MAX_STEP_DAYS", "SteppedRange", "iter_stepped_dates", "parse_step_suffix")
