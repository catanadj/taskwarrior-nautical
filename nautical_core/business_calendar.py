from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Callable, Protocol


class BusinessCalendar(Protocol):
    name: str

    def is_business_day(self, value: date) -> bool: ...


class BusinessCalendarSearchError(ValueError):
    pass


@dataclass(frozen=True)
class WeekdayBusinessCalendar:
    name: str = "weekday"

    def is_business_day(self, value: date) -> bool:
        return value.weekday() < 5


@dataclass(frozen=True, eq=False)
class ConfiguredBusinessCalendar:
    name: str
    anchor_dates: frozenset[date]
    omit_dates: frozenset[date]
    _anchor_matches: Callable[[date], bool]
    _omit_matches: Callable[[date], bool]

    @lru_cache(maxsize=4096)
    def is_business_day(self, value: date) -> bool:
        included = value in self.anchor_dates or self._anchor_matches(value)
        excluded = value in self.omit_dates or self._omit_matches(value)
        return included and not excluded


DEFAULT_BUSINESS_CALENDAR: BusinessCalendar = WeekdayBusinessCalendar()
WEEKDAY_BUSINESS_DAYS = frozenset({0, 1, 2, 3, 4})


def is_business_day(
    value: date,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> bool:
    return business_calendar.is_business_day(value)


def find_business_day(
    value: date,
    direction: int,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
    *,
    max_scan_days: int = 8,
) -> date:
    if direction not in {-1, 1}:
        raise ValueError("business-day search direction must be -1 or 1")
    current = value
    for _ in range(max_scan_days):
        if business_calendar.is_business_day(current):
            return current
        current += timedelta(days=direction)
    raise BusinessCalendarSearchError("no business day found within the search limit")


def nearest_business_day(
    value: date,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
    *,
    max_scan_days: int = 8,
) -> date:
    previous = find_business_day(
        value,
        -1,
        business_calendar,
        max_scan_days=max_scan_days,
    )
    following = find_business_day(
        value,
        1,
        business_calendar,
        max_scan_days=max_scan_days,
    )
    return previous if (value - previous) <= (following - value) else following


def shift_business_days(
    value: date,
    offset: int,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
    *,
    max_scan_days_per_step: int = 3660,
) -> date:
    if not offset:
        return value
    direction = 1 if offset > 0 else -1
    current = value
    for _ in range(abs(offset)):
        current += timedelta(days=direction)
        for _ in range(max_scan_days_per_step):
            if business_calendar.is_business_day(current):
                break
            current += timedelta(days=direction)
        else:
            raise BusinessCalendarSearchError("no business day found within the offset search limit")
    return current


def business_days_in_month(
    year: int,
    month: int,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> list[date]:
    current = date(year, month, 1)
    out: list[date] = []
    while current.month == month:
        if business_calendar.is_business_day(current):
            out.append(current)
        current += timedelta(days=1)
    return out


def nth_business_day_of_month(
    year: int,
    month: int,
    ordinal: int,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> date | None:
    if ordinal == 0:
        return None
    days = business_days_in_month(year, month, business_calendar)
    index = ordinal - 1 if ordinal > 0 else ordinal
    try:
        return days[index]
    except IndexError:
        return None


def business_day_offsets_for_iso_week(
    iso_year: int,
    iso_week: int,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> list[int]:
    monday = date.fromisocalendar(iso_year, iso_week, 1)
    return [
        offset
        for offset in range(7)
        if business_calendar.is_business_day(monday + timedelta(days=offset))
    ]
