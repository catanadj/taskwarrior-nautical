from __future__ import annotations

import hashlib
from datetime import date, timedelta


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


def seeded_int(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def weekly_rand_pick(iso_year: int, iso_week: int, mods: dict, *, wrand_salt: str, seeded_int) -> int:
    pool = [0, 1, 2, 3, 4] if (mods.get("bd") or mods.get("wd") is True) else [0, 1, 2, 3, 4, 5, 6]
    key = f"{wrand_salt}|{iso_year}|{iso_week}|{'bd' if len(pool) == 5 else 'all'}"
    n = seeded_int(key)
    return pool[n % len(pool)]


def is_bd(dt: date) -> bool:
    return dt.weekday() < 5


def sha_pick(seq_len: int, seed_key: str) -> int:
    h = hashlib.sha256(seed_key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % seq_len


def term_rand_info(term):
    for i, atom in enumerate(term):
        typ = (atom.get("typ") or atom.get("type") or "").lower()
        spec = str(atom.get("spec") or atom.get("value") or "").lower()
        if typ == "m" and spec == "rand":
            return (
                "m",
                {
                    "mods": atom.get("mods") or {},
                    "ival": int(atom.get("ival") or atom.get("intv") or 1),
                    "atom_idx": i,
                },
            )
        if typ == "y":
            if spec == "rand":
                return ("y", {"mods": atom.get("mods") or {}, "month": None, "atom_idx": i})
            if spec.startswith("rand-"):
                mm = int(spec.split("-", 1)[1])
                return ("y", {"mods": atom.get("mods") or {}, "month": mm, "atom_idx": i})
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
) -> set[int]:
    spec = expand_monthly_aliases(spec)
    ndays = days_in_month(y, m)
    out: set[int] = set()

    m2 = bd_re.match(spec)
    if m2:
        k = int(m2.group(1))
        bds = [d for d in range(1, ndays + 1) if date(y, m, d).weekday() < 5]
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
        if sp == "rand":
            continue
        msets.append(month_tokens_for_atom(atom, y, m))
    if msets:
        allowed_days = set.intersection(*msets) if msets else set()
        dates = [d for d in dates if d.day in allowed_days]

    y_specs = [str(atom.get("spec") or "") for atom in term if atype(atom) == "y"]
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
        days = [d for d in days if d < 5]
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

    days = []
    for tok in split_csv_lower(spec):
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

    def nth_business_day(n: int):
        if n == 0:
            return None
        if n > 0:
            cnt = 0
            d = date(y, m, 1)
            while d.month == m:
                if d.weekday() < 5:
                    cnt += 1
                    if cnt == n:
                        return d.day
                d = d + timedelta(days=1)
            return None
        cnt = 0
        d = date(y, m, last)
        while d.month == m:
            if d.weekday() < 5:
                cnt += 1
                if cnt == abs(n):
                    return d.day
            d = d - timedelta(days=1)
        return None

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
            d0 = nth_business_day(n)
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
