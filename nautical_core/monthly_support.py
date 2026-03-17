from __future__ import annotations

from datetime import date


def doms_for_monthly_token(
    tok: str,
    y: int,
    m: int,
    *,
    monthly_alias,
    days_in_month,
    re_mod,
    nth_re,
    wd_idx,
) -> set[int]:
    tok = (tok or "").strip().lower()
    if tok in monthly_alias:
        tok = monthly_alias[tok]
    dim = days_in_month(y, m)
    if tok == "rand":
        return set(range(1, dim + 1))
    match = re_mod.fullmatch(r"(\-?\d{1,2})\.\.(\-?\d{1,2})", tok)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        if a < 0:
            a = dim + 1 + a
        if b < 0:
            b = dim + 1 + b
        a = max(1, min(dim, a))
        b = max(1, min(dim, b))
        lo, hi = (a, b) if a <= b else (b, a)
        return set(range(lo, hi + 1))
    if re_mod.fullmatch(r"\-?\d{1,2}", tok):
        d = int(tok)
        if d < 0:
            d = dim + 1 + d
        if 1 <= d <= dim:
            return {d}
        return set()
    match = nth_re.fullmatch(tok)
    if match:
        nth_s, wd_s = match.group(1), match.group(2)
        wd = wd_idx(wd_s)
        if wd is None:
            return set()
        days = [d for d in range(1, dim + 1) if date(y, m, d).weekday() == wd]
        if nth_s:
            idx = int(nth_s) - 1
            return {days[idx]} if 0 <= idx < len(days) else set()
        return {days[-1]} if days else set()
    return set()


def month_allowed_doms_for_monthly_atom(
    atom: dict,
    y: int,
    m: int,
    dim: int,
    *,
    split_csv_lower,
    doms_for_monthly_token,
) -> set[int]:
    spec = str(atom.get("spec") or "")
    toks = split_csv_lower(spec)
    if not toks:
        return set(range(1, dim + 1))
    doms: set[int] = set()
    for tok in toks:
        if tok == "rand":
            doms.update(range(1, dim + 1))
        else:
            doms.update(doms_for_monthly_token(tok, y, m))
    return doms


def intersect_monthly_atoms_allowed(
    term: list[dict],
    *,
    y: int,
    m: int,
    dim: int,
    allowed: set[int],
    month_allowed_doms_for_monthly_atom,
) -> set[int]:
    out = set(allowed)
    for atom in term:
        typ = (atom.get("typ") or atom.get("type") or "").lower()
        if typ != "m":
            continue
        out &= month_allowed_doms_for_monthly_atom(atom, y, m, dim)
        if not out:
            return set()
    return out


def month_doms_safe(spec: str, y: int, m: int, *, expand_monthly_cached) -> list[int]:
    try:
        return sorted(expand_monthly_cached(spec, y, m))
    except Exception:
        return []


def month_has_hit(spec: str, y: int, m: int, *, month_doms_safe) -> bool:
    return bool(month_doms_safe(spec, y, m))


def first_hit_after_probe_in_month(spec: str, y: int, m: int, probe: date, *, month_doms_safe) -> date | None:
    for d0 in month_doms_safe(spec, y, m):
        dt = date(y, m, d0)
        if dt > probe:
            return dt
    return None


def next_valid_month_on_or_after(spec: str, y: int, m: int, *, month_has_hit) -> tuple[int, int]:
    yy, mm = y, m
    for _ in range(480):
        if month_has_hit(spec, yy, mm):
            return yy, mm
        mm += 1
        if mm > 12:
            yy += 1
            mm = 1
    return y, m


def advance_k_valid_months(spec: str, start_y: int, start_m: int, k: int, *, next_valid_month_on_or_after) -> tuple[int, int]:
    yy, mm = start_y, start_m
    steps = max(k, 0)
    while steps >= 0:
        mm += 1
        if mm > 12:
            mm = 1
            yy += 1
        yy, mm = next_valid_month_on_or_after(spec, yy, mm)
        steps -= 1
    return yy, mm


def monthly_align_base_for_interval(
    spec: str,
    base: date,
    probe: date,
    seed: date,
    ival: int,
    *,
    month_has_hit,
    next_valid_month_on_or_after,
    first_hit_after_probe_in_month,
    advance_k_valid_months,
    month_doms_safe,
) -> date:
    by, bm = base.year, base.month
    sy, sm = next_valid_month_on_or_after(spec, seed.year, seed.month)

    if not month_has_hit(spec, by, bm):
        by, bm = next_valid_month_on_or_after(spec, by, bm)
        nxt = first_hit_after_probe_in_month(spec, by, bm, probe)
        if nxt is None:
            ny, nm = advance_k_valid_months(spec, by, bm, 0)
            doms = month_doms_safe(spec, ny, nm)
            base = date(ny, nm, doms[0])
        else:
            base = nxt
    elif base <= probe:
        nxt = first_hit_after_probe_in_month(spec, by, bm, probe)
        if nxt is None:
            ny, nm = advance_k_valid_months(spec, by, bm, 0)
            doms = month_doms_safe(spec, ny, nm)
            base = date(ny, nm, doms[0])
        else:
            base = nxt

    cnt = 0
    ty, tm = sy, sm
    while (ty, tm) != (base.year, base.month) and cnt < 480:
        ty, tm = advance_k_valid_months(spec, ty, tm, 0)
        cnt += 1

    if (cnt % ival) != 0:
        steps = ival - (cnt % ival)
        ny, nm = advance_k_valid_months(spec, base.year, base.month, steps - 1)
        doms = month_doms_safe(spec, ny, nm)
        base = date(ny, nm, doms[0])

    return base
