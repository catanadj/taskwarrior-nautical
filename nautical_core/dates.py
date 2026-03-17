from __future__ import annotations

from datetime import date, datetime


def month_len(y, m):
    """Get number of days in month."""
    import calendar

    return calendar.monthrange(y, m)[1]


def add_months(d: date, months: int) -> date:
    """Add months to date, handling month-end correctly."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last = month_len(y, m)
    return date(y, m, min(d.day, last))


def months_days_between(d1: date, d2: date):
    """Calculate months and days between two dates."""
    sign = 1
    if d2 < d1:
        d1, d2 = d2, d1
        sign = -1
    months = (d2.year - d1.year) * 12 + (d2.month - d1.month)
    if add_months(d1, months) > d2:
        months -= 1
    anchor = add_months(d1, months)
    days = (d2 - anchor).days
    return sign * months, sign * days


def humanize_delta(from_dt: datetime, to_dt: datetime, use_months_days: bool):
    """Human-readable time difference between datetimes."""
    td = to_dt - from_dt
    if use_months_days:
        m, d = months_days_between(from_dt.date(), to_dt.date())
        future = m > 0 or (m == 0 and d > 0)
        label = "in" if future else "overdue by"
        m, d = abs(m), abs(d)
        parts = []
        if m:
            parts.append(f"{m}mo")
        if d or not parts:
            parts.append(f"{d}d")
        return f"{label} " + " ".join(parts)
    secs = int(abs(td.total_seconds()))
    label = "in" if td.total_seconds() > 0 else "overdue by"
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{label} {days}d {hours}h"
    if hours:
        return f"{label} {hours}h {minutes}m"
    return f"{label} {minutes}m"
