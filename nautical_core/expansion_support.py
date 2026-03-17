from __future__ import annotations

from datetime import date


def days_in_month(y: int, m: int, *, monthrange) -> int:
    return monthrange(y, m)[1]


def wd_idx(s: str, *, wd_abbr: list[str]) -> int | None:
    s = (s or "").strip().lower()
    if s in wd_abbr:
        return wd_abbr.index(s)
    try:
        n = int(s)
        if 1 <= n <= 7:
            return n - 1
    except Exception:
        pass
    return None


def wday_idx_any(s: str, *, weekdays: dict[str, int], wd_idx) -> int | None:
    s = (s or "").strip().lower()
    if not s:
        return None
    if s in weekdays:
        return weekdays[s]
    return wd_idx(s)


def weekly_spec_to_wset(
    spec: str,
    *,
    mods: dict | None = None,
    expand_weekly_aliases,
    split_csv_lower,
    wday_idx_any,
) -> set[int]:
    spec = expand_weekly_aliases(spec)
    if not spec:
        return set()

    toks = split_csv_lower(spec)
    out: set[int] = set()

    if any(t == "rand" for t in toks):
        pool = (
            {0, 1, 2, 3, 4}
            if ((mods or {}).get("bd") or (mods or {}).get("wd") is True)
            else {0, 1, 2, 3, 4, 5, 6}
        )
        out |= pool

    for tok in toks:
        if tok == "rand":
            continue
        if ".." in tok:
            a, b = tok.split("..", 1)
            ia, ib = wday_idx_any(a), wday_idx_any(b)
            if ia is None or ib is None:
                continue
            rng = (
                list(range(ia, ib + 1))
                if ia <= ib
                else (list(range(ia, 7)) + list(range(0, ib + 1)))
            )
            out.update(rng)
        else:
            i = wday_idx_any(tok)
            if i is not None:
                out.add(i)

    return out


def doms_for_weekly_spec(
    spec: str,
    y: int,
    m: int,
    *,
    expand_weekly_aliases,
    split_csv_tokens,
    wd_idx,
    days_in_month,
) -> set[int]:
    spec = expand_weekly_aliases(spec)
    if not spec:
        return set()
    wset: set[int] = set()
    for tok in split_csv_tokens(spec):
        if ".." in tok:
            a, b = tok.split("..", 1)
            ia, ib = wd_idx(a), wd_idx(b)
            if ia is None or ib is None:
                continue
            rng = list(range(ia, ib + 1)) if ia <= ib else (list(range(ia, 7)) + list(range(0, ib + 1)))
            wset.update(rng)
        else:
            i = wd_idx(tok)
            if i is not None:
                wset.add(i)
    if not wset:
        return set()
    dim = days_in_month(y, m)
    allowed: set[int] = set()
    for d in range(1, dim + 1):
        if date(y, m, d).weekday() in wset:
            allowed.add(d)
    return allowed


def y_ranges_from_spec(spec: str, *, split_csv_lower, re_mod, year_pair) -> list[tuple[int, int, int, int]]:
    out = []
    for tok in split_csv_lower(spec):
        m_randm = re_mod.fullmatch(r"rand-(\d{2})", tok)
        if m_randm:
            mm = int(m_randm.group(1))
            if 1 <= mm <= 12:
                out.append((mm, 1, mm, 31))
            continue

        m = re_mod.fullmatch(r"(\d{2})-(\d{2})(?:\.\.(\d{2})-(\d{2}))?$", tok)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        d1, m1 = year_pair(a, b)
        if m.group(3):
            c, d = int(m.group(3)), int(m.group(4))
            d2, m2 = year_pair(c, d)
        else:
            d2, m2 = d1, m1
        out.append((m1, d1, m2, d2))
    return out


def doms_allowed_by_year(
    y: int,
    m: int,
    y_specs: list[str],
    *,
    y_ranges_from_spec,
    days_in_month,
) -> set[int]:
    if not y_specs:
        return set(range(1, days_in_month(y, m) + 1))
    if any((sp or "").strip().lower() == "rand" for sp in y_specs):
        return set(range(1, days_in_month(y, m) + 1))

    ranges = []
    for sp in y_specs:
        ranges.extend(y_ranges_from_spec(sp))
    if not ranges:
        return set()

    dim = days_in_month(y, m)
    allowed: set[int] = set()
    for (m1, d1, m2, d2) in ranges:
        if m1 == m2:
            if m == m1:
                lo, hi = max(1, d1), min(dim, d2)
                allowed.update(range(lo, hi + 1))
        else:
            if m < m1 or m > m2:
                continue
            if m == m1:
                allowed.update(range(max(1, d1), dim + 1))
            elif m == m2:
                allowed.update(range(1, min(dim, d2) + 1))
            elif m1 < m < m2:
                allowed.update(range(1, dim + 1))
    return allowed
