"""Business-calendar benchmark workloads."""

from __future__ import annotations

from datetime import date, timedelta
import importlib
import time
from typing import Any


def large_omissions(core: Any, rounds: int) -> float:
    """Measure recurrence selection with a large explicit omission set."""
    business_calendar = importlib.import_module("nautical_core.business_calendar")
    dnf = core.validate_anchor_expr_strict("m:1..31@bd")
    first_day = date(2026, 1, 1)
    calendar_days = frozenset(first_day + timedelta(days=index) for index in range(366))
    omitted = frozenset(
        item
        for item in calendar_days
        if item.day not in {1, 15} or item.weekday() >= 5
    )
    calendar = business_calendar.ConfiguredBusinessCalendar(
        name="perf-omissions",
        fingerprint="perf-omissions-v1",
        anchor_dates=calendar_days,
        omit_dates=omitted,
        _anchor_matches=lambda _value: False,
        _omit_matches=lambda _value: False,
    )
    started = time.perf_counter()
    for index in range(max(1, int(rounds))):
        result, _meta = core.next_after_expr(
            dnf,
            first_day + timedelta(days=index % 300),
            seed_base="perf-omissions",
            business_calendar=calendar,
        )
        if result is None or result.day not in {1, 15} or result.weekday() >= 5:
            raise RuntimeError("large omission-set benchmark selected an omitted date")
    return time.perf_counter() - started
