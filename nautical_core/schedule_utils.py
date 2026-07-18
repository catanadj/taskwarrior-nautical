from __future__ import annotations

from datetime import date, timedelta

from .business_calendar import (
    DEFAULT_BUSINESS_CALENDAR,
    BusinessCalendar,
    BusinessCalendarSearchError,
    find_business_day,
    record_business_calendar_displacement,
    shift_business_days,
)


def roll_apply(
    dt: date,
    mods: dict,
    *,
    parse_error_cls,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> date:
    roll = mods.get("roll")
    original = dt

    if roll in ("pbd", "nbd", "nw"):
        if not business_calendar.is_business_day(dt):
            if roll == "pbd":
                try:
                    dt = find_business_day(dt, -1, business_calendar)
                except BusinessCalendarSearchError:
                    raise parse_error_cls("roll_apply: failed to reach business day (pbd)")
            elif roll == "nbd":
                try:
                    dt = find_business_day(dt, 1, business_calendar)
                except BusinessCalendarSearchError:
                    raise parse_error_cls("roll_apply: failed to reach business day (nbd)")
            else:
                try:
                    prev_dt = find_business_day(dt, -1, business_calendar)
                except BusinessCalendarSearchError:
                    raise parse_error_cls("roll_apply: failed to reach business day (nw prev)")
                try:
                    next_dt = find_business_day(dt, 1, business_calendar)
                except BusinessCalendarSearchError:
                    raise parse_error_cls("roll_apply: failed to reach business day (nw next)")
                dt = prev_dt if (dt - prev_dt) <= (next_dt - dt) else next_dt

        record_business_calendar_displacement(
            original,
            dt,
            business_calendar,
            operation=str(roll),
        )

    elif roll in ("next-wd", "prev-wd"):
        tgt = mods.get("wd")
        if tgt is not None:
            if roll == "next-wd":
                dt += timedelta(days=1)
                for _ in range(8):
                    if dt.weekday() == tgt:
                        break
                    dt += timedelta(days=1)
                else:
                    raise parse_error_cls("roll_apply: failed to reach target weekday (next-wd)")
            else:
                dt -= timedelta(days=1)
                for _ in range(8):
                    if dt.weekday() == tgt:
                        break
                    dt -= timedelta(days=1)
                else:
                    raise parse_error_cls("roll_apply: failed to reach target weekday (prev-wd)")

    return dt


def weeks_between(d1: date, d2: date) -> int:
    iso1 = d1.isocalendar()
    iso2 = d2.isocalendar()
    mon1 = date.fromisocalendar(iso1.year, iso1.week, 1)
    mon2 = date.fromisocalendar(iso2.year, iso2.week, 1)
    return (mon2 - mon1).days // 7


def apply_day_offset(
    d: date,
    mods: dict,
    *,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> date:
    """Apply calendar-day offsets, then exclusive business-day offsets."""
    off = int(mods.get("day_offset", 0) or 0)
    if off:
        d += timedelta(days=off)
    business_off = int(mods.get("business_day_offset", 0) or 0)
    adjusted = shift_business_days(d, business_off, business_calendar)
    if business_off:
        record_business_calendar_displacement(
            d,
            adjusted,
            business_calendar,
            operation=f"{business_off:+d}bd",
        )
    return adjusted


def expr_has_m_or_y(dnf) -> bool:
    for term in dnf or []:
        for atom in term:
            if atom.get("kind") == "select":
                if atom.get("scope") in ("month", "quarter", "year"):
                    return True
                if expr_has_m_or_y(atom.get("expr") or []):
                    return True
                continue
            if atom["typ"] in ("m", "y"):
                return True
    return False


def pick_hhmm_from_dnf_for_date(
    dnf,
    target: date,
    default_seed: date,
    seed_base=None,
    *,
    atom_matches_on,
):
    for term in dnf:
        if all(atom_matches_on(atom, target, default_seed, seed_base=seed_base) for atom in term):
            for atom in term:
                tval = atom["mods"].get("t")
                if not tval:
                    continue
                if isinstance(tval, list):
                    return tval[0] if tval else None
                return tval
    return None
