from __future__ import annotations

from datetime import date, timedelta
import re


def parse_nth_wd_tokens(
    spec: str,
    *,
    split_csv_lower,
    nth_weekday_re: re.Pattern,
    weekdays: dict[str, int],
):
    """Return list of (k, wd) for pure nth-weekday spec, else None."""
    toks = split_csv_lower(spec)
    out = []
    for tok in toks:
        m = nth_weekday_re.match(tok)
        if not m:
            return None
        n_raw, wd_s = m.group(1), m.group(2)
        if n_raw == "last":
            k = -1
        else:
            n_txt = re.sub(r"(st|nd|rd|th)$", "", n_raw)
            k = int(n_txt)
            if k == 0 or abs(k) > 5:
                return None
        out.append((k, weekdays[wd_s]))
    return out


def month_has_any_nth(
    y: int,
    m: int,
    pairs: list[tuple[int, int]],
    *,
    month_len,
) -> bool:
    """Does month (y,m) have ANY of the requested nth-weekdays?"""
    last = month_len(y, m)

    def kth(n, wd):
        if n == 0:
            return None
        if n > 0:
            d = date(y, m, 1)
            off = (wd - d.weekday()) % 7
            d = d + timedelta(days=off + (n - 1) * 7)
            return d.day if d.month == m else None
        d = date(y, m, last)
        off = (d.weekday() - wd) % 7
        d = d - timedelta(days=off + (abs(n) - 1) * 7)
        return d.day if d.month == m else None

    for n, wd in pairs:
        if kth(n, wd):
            return True
    return False


def advance_to_next_allowed_month(y: int, m: int, pairs, *, month_has_any_nth) -> tuple[int, int]:
    """Next (including current) month that has an nth-weekday match."""
    yy, mm = y, m
    for _ in range(24):
        if month_has_any_nth(yy, mm, pairs):
            return (yy, mm)
        mm = 1 if mm == 12 else mm + 1
        if mm == 1:
            yy += 1
    return (y, m)
