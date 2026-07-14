from __future__ import annotations

import hashlib
import json
import re
from datetime import date, timedelta

from .business_calendar import (
    DEFAULT_BUSINESS_CALENDAR,
    WEEKDAY_BUSINESS_DAYS,
    BusinessCalendar,
    business_day_offsets_for_iso_week,
    business_days_in_month,
    nth_business_day_of_month,
)

RAND_ALGORITHM_VERSION = "nautical-rand-v2"


def active_mod_keys(mods: dict) -> set:
    """Return only modifiers that are actually 'used' (truthy / non-zero)."""
    act = set()
    for k, v in (mods or {}).items():
        if v in (None, False, 0, 0.0, "", []):
            continue
        act.add(k)
    return act


def week_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def random_identity(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def random_pick_index(
    seq_len: int,
    *,
    seed_base: str | None,
    domain: str,
    identity: str,
    period: str,
    attempt: int = 0,
    namespace: str = "",
) -> int:
    if seq_len <= 0:
        raise ValueError("random choice requires at least one candidate")
    key = "|".join(
        (
            RAND_ALGORITHM_VERSION,
            namespace,
            seed_base or "preview",
            domain,
            identity,
            period,
            str(attempt),
        )
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % seq_len


def random_count_from_spec(spec: str) -> int | None:
    text = str(spec or "").strip().lower()
    if text == "rand":
        return 1
    match = re.fullmatch(r"([1-9]\d{0,2})rand", text)
    return int(match.group(1)) if match else None


def dnf_has_counted_random(dnf) -> bool:
    for term in dnf or []:
        for atom in term or []:
            count = random_count_from_spec(atom.get("spec") or atom.get("value") or "")
            if count is not None and count > 1:
                return True
    return False


def random_pick_indices(
    seq_len: int,
    count: int,
    *,
    seed_base: str | None,
    domain: str,
    identity: str,
    period: str,
    namespace: str = "",
) -> list[int]:
    if count < 1 or count > seq_len:
        return []
    remaining = list(range(seq_len))
    selected: list[int] = []
    for draw in range(count):
        pos = random_pick_index(
            len(remaining),
            seed_base=seed_base,
            domain=domain,
            identity=identity,
            period=period,
            attempt=draw,
            namespace=namespace,
        )
        selected.append(remaining.pop(pos))
    return sorted(selected)


def weekly_rand_pick(
    iso_year: int,
    iso_week: int,
    mods: dict,
    *,
    seed_base: str | None,
    atom_identity: str,
    namespace: str,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> int | None:
    pool = (
        business_day_offsets_for_iso_week(iso_year, iso_week, business_calendar)
        if (mods.get("bd") or mods.get("wd") is True)
        else list(range(7))
    )
    if not pool:
        return None
    idx = random_pick_index(
        len(pool),
        seed_base=seed_base,
        domain="weekly",
        identity=atom_identity,
        period=f"{iso_year:04d}-W{iso_week:02d}",
        namespace=namespace,
    )
    return pool[idx]


def is_bd(
    dt: date,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> bool:
    return business_calendar.is_business_day(dt)


def term_rand_info(term):
    for i, atom in enumerate(term):
        typ = (atom.get("typ") or atom.get("type") or "").lower()
        spec = str(atom.get("spec") or atom.get("value") or "").lower()
        count = random_count_from_spec(spec)
        if typ == "w" and count is not None and count > 1:
            return (
                "w",
                {
                    "mods": atom.get("mods") or {},
                    "ival": int(atom.get("ival") or atom.get("intv") or 1),
                    "atom_idx": i,
                    "count": count,
                },
            )
        if typ == "m" and count is not None:
            return (
                "m",
                {
                    "mods": atom.get("mods") or {},
                    "ival": int(atom.get("ival") or atom.get("intv") or 1),
                    "atom_idx": i,
                    "count": count,
                },
            )
        if typ == "y":
            if count is not None:
                return (
                    "y",
                    {
                        "mods": atom.get("mods") or {},
                        "month": None,
                        "atom_idx": i,
                        "count": count,
                        "ival": int(atom.get("ival") or atom.get("intv") or 1),
                    },
                )
            if spec.startswith("rand-"):
                mm = int(spec.split("-", 1)[1])
                return (
                    "y",
                    {
                        "mods": atom.get("mods") or {},
                        "month": mm,
                        "atom_idx": i,
                        "count": 1,
                        "ival": int(atom.get("ival") or atom.get("intv") or 1),
                    },
                )
    return (None, None)


def filter_by_w(dt_list, term, *, atype, aspec, weekly_spec_to_wset):
    allowed = None
    for atom in term:
        if atype(atom) != "w":
            continue
        spec = (aspec(atom) or "").lower()
        wset = weekly_spec_to_wset(spec, mods=atom.get("mods") or {})
        allowed = wset if allowed is None else (allowed & wset)
    if allowed is None:
        return dt_list
    if not allowed:
        return []
    return [d for d in dt_list if d.weekday() in allowed]


def month_tokens_for_atom_values(
    y: int,
    m: int,
    spec: str,
    *,
    expand_monthly_aliases,
    days_in_month,
    bd_re,
    nth_weekday_re,
    weekday_map,
    re_mod,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> set[int]:
    spec = expand_monthly_aliases(spec)
    ndays = days_in_month(y, m)
    out: set[int] = set()

    m2 = bd_re.match(spec)
    if m2:
        k = int(m2.group(1))
        bds = [item.day for item in business_days_in_month(y, m, business_calendar)]
        if not bds:
            return out
        if k > 0:
            if k <= len(bds):
                out.add(bds[k - 1])
        else:
            k = -k
            if k <= len(bds):
                out.add(bds[-k])
        return out

    m3 = nth_weekday_re.match(spec)
    if m3:
        idx, wd = m3.group(1), weekday_map[m3.group(2)]
        days = [d for d in range(1, ndays + 1) if date(y, m, d).weekday() == wd]
        if not days:
            return out
        if idx == "last":
            out.add(days[-1])
            return out
        k = int(re_mod.sub(r"(st|nd|rd|th)$", "", idx))
        if k > 0 and k <= len(days):
            out.add(days[k - 1])
        elif k < 0 and -k <= len(days):
            out.add(days[k])
        return out

    if ".." in spec:
        a_s, b_s = spec.split("..", 1)
        try:
            a_i = int(a_s)
            b_i = int(b_s)
        except Exception:
            return out

        def norm(n):
            return ndays + n + 1 if n < 0 else n

        lo, hi = norm(a_i), norm(b_i)
        lo = max(1, lo)
        hi = min(ndays, hi)
        if lo <= hi:
            out.update(range(lo, hi + 1))
        return out

    try:
        k = int(spec)
        if k < 0:
            k = days_in_month(y, m) + k + 1
        if 1 <= k <= ndays:
            out.add(k)
    except Exception:
        pass
    return out


def month_tokens_for_atom(atom: dict, y: int, m: int, *, month_tokens_for_atom_cached) -> set[int]:
    spec = str(atom.get("spec")).lower().strip()
    return month_tokens_for_atom_cached(y, m, spec)


def term_candidates_in_month(
    term: list[dict],
    y: int,
    m: int,
    rand_atom_idx: int,
    bd_only: bool,
    *,
    days_in_month,
    is_bd,
    filter_by_w,
    atype,
    aspec,
    month_tokens_for_atom,
    doms_allowed_by_year,
):
    days = list(range(1, days_in_month(y, m) + 1))
    dates = [date(y, m, d) for d in days]

    if bd_only:
        dates = [d for d in dates if is_bd(d)]

    dates = filter_by_w(dates, term)

    msets = []
    for idx, atom in enumerate(term):
        if idx == rand_atom_idx:
            continue
        if atype(atom) != "m":
            continue
        sp = aspec(atom)
        if random_count_from_spec(sp) is not None:
            continue
        msets.append(month_tokens_for_atom(atom, y, m))
    if msets:
        allowed_days = set.intersection(*msets) if msets else set()
        dates = [d for d in dates if d.day in allowed_days]

    y_specs = [
        str(atom.get("spec") or "")
        for idx, atom in enumerate(term)
        if idx != rand_atom_idx and atype(atom) == "y"
    ]
    if y_specs:
        allowed_dom = doms_allowed_by_year(y, m, y_specs)
        if allowed_dom:
            dates = [d for d in dates if d.day in allowed_dom]
        else:
            dates = []

    return dates


def expand_weekly(spec: str, *, weekly_spec_to_wset):
    return sorted(weekly_spec_to_wset(spec, mods=None))


def expand_weekly_mods(spec: str, bd_only: bool, *, expand_weekly_cached):
    days = expand_weekly_cached(spec)
    if bd_only:
        days = [d for d in days if d in WEEKDAY_BUSINESS_DAYS]
    return days


def expand_yearly(
    spec: str,
    y: int,
    *,
    rewrite_month_names_to_ranges,
    split_csv_lower,
    re_mod,
    month_len,
    yearfmt,
) -> list[date]:
    spec = rewrite_month_names_to_ranges(spec)
    if not spec:
        return []

    def _mlen(mm: int) -> int:
        return month_len(y, mm)

    def _strict_date(d: int, m: int) -> date | None:
        if not (1 <= m <= 12):
            return None
        if not (1 <= d <= _mlen(m)):
            return None
        try:
            return date(y, m, d)
        except Exception:
            return None

    def _clamped_date(d: int, m: int) -> date | None:
        if not (1 <= m <= 12):
            return None
        d = max(1, min(d, _mlen(m)))
        try:
            return date(y, m, d)
        except Exception:
            return None

    def _pair(a: int, b: int) -> tuple[int, int]:
        return (b, a) if yearfmt() == "MD" else (a, b)

    def _year_day(ordinal: int) -> date | None:
        if ordinal == 0 or abs(ordinal) > 366:
            return None
        if ordinal > 0:
            candidate = date(y, 1, 1) + timedelta(days=ordinal - 1)
        else:
            candidate = date(y, 12, 31) + timedelta(days=ordinal + 1)
        return candidate if candidate.year == y else None

    days = []
    for tok in split_csv_lower(spec):
        year_day = re_mod.fullmatch(r"d(-?(?:0|[1-9]\d{0,2}))", tok)
        if year_day:
            candidate = _year_day(int(year_day.group(1)))
            if candidate:
                days.append(candidate)
            continue

        year_day_range = re_mod.fullmatch(
            r"d(-?(?:0|[1-9]\d{0,2}))\.\.d(-?(?:0|[1-9]\d{0,2}))",
            tok,
        )
        if year_day_range:
            start = _year_day(int(year_day_range.group(1)))
            end = _year_day(int(year_day_range.group(2)))
            if not start or not end or end < start:
                continue
            current = start
            while current <= end:
                days.append(current)
                current += timedelta(days=1)
            continue

        match = re_mod.fullmatch(r"(\d{2})-(\d{2})(?:\.\.(\d{2})-(\d{2}))?$", tok)
        if not match:
            continue
        a, b = int(match.group(1)), int(match.group(2))
        if match.group(3):
            c, d = int(match.group(3)), int(match.group(4))
            d1, m1 = _pair(a, b)
            d2, m2 = _pair(c, d)
            start = _clamped_date(d1, m1)
            end = _clamped_date(d2, m2)
            if not start or not end or end < start:
                continue
            cur = start
            while cur <= end:
                days.append(cur)
                cur += timedelta(days=1)
        else:
            d1, m1 = _pair(a, b)
            dd = _strict_date(d1, m1)
            if dd:
                days.append(dd)
    return sorted(days)


def expand_monthly(
    spec: str,
    y: int,
    m: int,
    *,
    month_len,
    expand_monthly_aliases,
    split_csv_lower,
    nth_weekday_re,
    bd_re,
    weekday_map,
    re_mod,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> list[int]:
    out = set()
    last = month_len(y, m)
    spec = expand_monthly_aliases(spec)

    def resolve_num(n):
        if n < 0:
            k = last + 1 + n
            return k if 1 <= k <= last else None
        return n if 1 <= n <= last else None

    def nth_weekday(n: int, wd: int):
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

    for tok in split_csv_lower(spec):
        m1 = nth_weekday_re.match(tok)
        if m1:
            n_raw, wd_s = m1.group(1), m1.group(2)
            if n_raw == "last":
                n = -1
            else:
                n_txt = re_mod.sub(r"(st|nd|rd|th)$", "", n_raw)
                n = int(n_txt)
            d0 = nth_weekday(n, weekday_map[wd_s])
            if d0:
                out.add(d0)
            continue
        m2 = bd_re.match(tok)
        if m2:
            n = int(m2.group(1))
            business_date = nth_business_day_of_month(y, m, n, business_calendar)
            d0 = business_date.day if business_date is not None else None
            if d0:
                out.add(d0)
                continue
        if ".." in tok:
            a_raw, b_raw = tok.split("..", 1)
            a = resolve_num(int(a_raw))
            b = resolve_num(int(b_raw))
            if a is None or b is None:
                continue
            step = 1 if a <= b else -1
            for r in range(a, b + step, step):
                out.add(r)
        else:
            try:
                r = resolve_num(int(tok))
                if r:
                    out.add(r)
            except Exception:
                pass
    return sorted(out)
