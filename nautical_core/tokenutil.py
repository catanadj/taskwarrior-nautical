from __future__ import annotations

from . import common as _common

WD_ABBR = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

WEEKLY_ALIAS = {
    "wk": "mon..fri",
    "we": "sat..sun",
    "wd": "mon..fri",
}

MONTHLY_ALIAS = {
    "ld": "-1",
    "lbd": "-1bd",
}

MONTH_ALIAS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def static_month_last_day(mm: int) -> int:
    # Use 29 for February so leap years are fully covered; clamp at expansion time.
    return {
        1: 31,
        2: 29,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }[mm]


def month_from_alias(tok: str) -> int | None:
    s = (tok or "").strip().lower()
    if s.isdigit() and len(s) == 2:
        mm = int(s)
        return mm if 1 <= mm <= 12 else None
    return MONTH_ALIAS.get(s)


def mon_to_int(tok: str) -> int | None:
    s = (tok or "").strip().lower()
    if not s:
        return None
    if s.isdigit() and len(s) == 2:
        mm = int(s)
        return mm if 1 <= mm <= 12 else None
    return MONTH_ALIAS.get(s)


def unwrap_quotes(s: str) -> str:
    """Trim one pair of wrapping quotes ('...' or "...") if present."""
    if not s:
        return s
    s = str(s).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        return s[1:-1].strip()
    return s


def expand_weekly_aliases(spec: str) -> str:
    spec = (spec or "").strip().lower()
    if not spec:
        return spec
    toks = _common.split_csv_tokens(spec)
    out = []
    for tok in toks:
        t = (tok or "").strip().lower()
        if t in WEEKLY_ALIAS:
            out.append(WEEKLY_ALIAS[t])
        else:
            out.append(t)
    return ",".join([t for t in out if t])


def expand_monthly_aliases(spec: str) -> str:
    spec = (spec or "").strip().lower()
    if not spec:
        return spec
    toks = _common.split_csv_tokens(spec)
    out = []
    for tok in toks:
        t = (tok or "").strip().lower()
        if t in MONTHLY_ALIAS:
            out.append(MONTHLY_ALIAS[t])
        else:
            out.append(t)
    return ",".join([t for t in out if t])


def normalize_weekday(s: str) -> str | None:
    s = (s or "").strip().lower()
    if not s:
        return None
    if s in ("rand", "rand*"):
        return s
    if s in WD_ABBR:
        return s
    try:
        n = int(s)
        if 1 <= n <= 7:
            return WD_ABBR[n - 1]
    except Exception:
        pass
    return None
