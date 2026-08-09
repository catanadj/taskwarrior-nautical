from __future__ import annotations

from datetime import timedelta
from math import gcd, lcm

from .business_calendar import is_business_day as default_is_business_day
from .scheduler_models import OccurrenceSearchExhausted


def _choose_rand_dom(
    y: int,
    m: int,
    doms: set[int],
    *,
    seed_base,
    identity: str,
    random_pick_index,
) -> int | None:
    if not doms:
        return None
    pool = sorted(doms)
    idx = random_pick_index(
        len(pool),
        seed_base=seed_base,
        domain="monthly-intersection",
        identity=identity,
        period=f"{y:04d}-{m:02d}",
    )
    return pool[idx]


def _term_has_monthly_rand(term: list[dict]) -> bool:
    return any(
        (a.get("typ") or a.get("type")) == "m"
        and "rand" in str(a.get("spec") or "").lower()
        for a in term
    )


def _term_year_specs(term: list[dict]) -> list[str]:
    return [str(a.get("spec") or "") for a in term if (a.get("typ") or a.get("type")) == "y"]


def _term_has_moon(term: list[dict]) -> bool:
    return any(
        (str(a.get("typ") or a.get("type") or "").lower() == "moon")
        or bool((a.get("mods") or {}).get("moon"))
        for a in term
    )


def _first_day_next_month(y: int, m: int, *, date_cls, days_in_month) -> object:
    return date_cls(y, m, 1) + timedelta(days=days_in_month(y, m))


def _intersect_weekly_atoms_allowed(
    term: list[dict],
    *,
    y: int,
    m: int,
    allowed: set[int],
    doms_for_weekly_spec,
) -> set[int]:
    out = set(allowed)
    for atom in term:
        typ = (atom.get("typ") or atom.get("type") or "").lower()
        if typ != "w":
            continue
        spec = str(atom.get("spec") or "")
        wdom = doms_for_weekly_spec(spec, y, m)
        out = out & wdom if wdom else set()
        if not out:
            return set()
    return out


_GREGORIAN_CYCLE_DAYS = 146097  # 400 Gregorian years; also divisible by 7.
_GREGORIAN_CYCLE_MONTHS = 400 * 12
_LAST_SUPPORTED_YEAR = 9999


def _calendar_month_span(after_date) -> int:
    """Return the number of calendar months still representable, inclusive."""
    return max(1, (_LAST_SUPPORTED_YEAR - after_date.year) * 12 + (12 - after_date.month) + 1)


def _calendar_year_span(after_date) -> int:
    """Return the number of calendar years still representable, inclusive."""
    return max(1, _LAST_SUPPORTED_YEAR - after_date.year + 1)


def _rand_search_limit(term: list[dict], after_date, interval: int, *, unit: str) -> int:
    """Derive a bounded random search from cadence and the supported calendar.

    Calendar selectors and weekday filters can repeat only after the Gregorian
    cycle.  The interval extends that cycle; the supported date range remains
    the hard upper bound.  The extra iteration includes the current bucket so
    an interval boundary immediately after ``after_date`` is not skipped.
    """
    interval = max(1, int(interval or 1))
    has_calendar_filter = any(
        (atom.get("typ") or atom.get("type") or "").lower() in {"w", "y"}
        for atom in term
    )
    if unit == "month":
        cycle = _GREGORIAN_CYCLE_MONTHS if has_calendar_filter else 1
        period = lcm(cycle, interval)
        return min(_calendar_month_span(after_date), period + 1)
    cycle = 400 if has_calendar_filter else 1
    period = lcm(cycle, interval)
    return min(_calendar_year_span(after_date), period + 1)


def _term_interval_lcm(term: list[dict], types: set[str]) -> int:
    interval = 1
    for atom in term:
        typ = (atom.get("typ") or atom.get("type") or "").lower()
        if typ not in types:
            continue
        try:
            interval = lcm(interval, max(1, int(atom.get("ival", 1) or 1)))
        except (TypeError, ValueError):
            continue
    return interval


def _periodic_atom_period_days(atom: dict) -> int | None:
    """Return a conservative period estimate for plain calendar atoms.

    This deliberately excludes dynamic modifiers.  Those atoms retain the
    ordinary guard because their provider may not repeat on the Gregorian
    cycle (astronomy, random selection, business calendars, or rolls).
    """
    typ = str(atom.get("typ") or atom.get("type") or "").lower()
    if typ not in {"w", "m", "y"} or atom.get("kind") == "select":
        return None
    spec = str(atom.get("spec") or "").lower()
    if "rand" in spec:
        return None
    mods = atom.get("mods") or {}
    if any(value not in (None, False, 0, "") for value in mods.values()):
        return None
    try:
        interval = max(1, int(atom.get("ival", 1) or 1))
    except (TypeError, ValueError):
        return None
    base_days = {"w": 7, "m": 31, "y": 366}[typ]
    return base_days * interval


def _next_for_and_periodic_cycle(
    term: list[dict],
    ref_d,
    seed,
    seed_base,
    *,
    next_after_atom_with_mods,
    atom_matches_on,
):
    """Search a finite Gregorian cycle using the sparsest plain atom.

    A leapfrog search can spend thousands of iterations on a valid but rare
    intersection.  Plain weekly/monthly/yearly rules repeat on a Gregorian
    cycle; interval phases extend that cycle by a calculable multiplier.
    Driving from the least frequent atom gives a bounded search without
    inventing a date or widening the normal guard.
    """
    scored = [
        (period, atom)
        for atom in term
        if (period := _periodic_atom_period_days(atom)) is not None
    ]
    if not scored:
        return None
    _period, driver = max(scored, key=lambda item: item[0])
    cycle_multiplier = 1
    for atom in term:
        typ = str(atom.get("typ") or atom.get("type") or "").lower()
        try:
            interval = max(1, int(atom.get("ival", 1) or 1))
        except (TypeError, ValueError):
            return None
        base_period = {"w": 20871, "m": 4800, "y": 400}.get(typ)
        if base_period is None:
            continue
        cycle_multiplier = lcm(cycle_multiplier, interval // gcd(base_period, interval))
    cycle_days = _GREGORIAN_CYCLE_DAYS * cycle_multiplier
    try:
        driver_interval = max(1, int(driver.get("ival", 1) or 1))
    except (TypeError, ValueError):
        return None
    # Each interval bucket can contain at most one candidate per calendar day.
    max_candidates = (cycle_days // driver_interval) + 2
    cursor = ref_d
    for _ in range(max_candidates):
        try:
            candidate = next_after_atom_with_mods(driver, cursor, seed, seed_base=seed_base)
        except OverflowError:
            return None
        if candidate is None:
            return None
        if candidate <= cursor:
            return None
        if all(atom_matches_on(atom, candidate, seed, seed_base=seed_base) for atom in term):
            return candidate
        cursor = candidate
    return None


def next_for_and_rand_yearly(
    term: list[dict],
    ref_d,
    y_specs: list[str],
    *,
    default_seed=None,
    seed_base,
    identity: str,
    random_pick_index,
    days_in_month,
    doms_allowed_by_year,
    intersect_monthly_atoms_allowed,
    doms_for_weekly_spec,
    date_cls,
):
    seed_loc = default_seed or ref_d
    monthly_interval = _term_interval_lcm(term, {"m"})
    yearly_interval = _term_interval_lcm(term, {"y"})

    search_limit = _rand_search_limit(
        term,
        ref_d,
        lcm(monthly_interval, yearly_interval),
        unit="month",
    )
    probe = ref_d + timedelta(days=1)
    for _ in range(search_limit):
        y, m = probe.year, probe.month
        month_offset = (y - seed_loc.year) * 12 + (m - seed_loc.month)
        if month_offset % monthly_interval != 0 or (y - seed_loc.year) % yearly_interval != 0:
            probe = _first_day_next_month(y, m, date_cls=date_cls, days_in_month=days_in_month)
            continue
        dim = days_in_month(y, m)
        allowed = set(range(1, dim + 1))
        allowed &= doms_allowed_by_year(y, m, y_specs)
        if not allowed:
            probe = _first_day_next_month(y, m, date_cls=date_cls, days_in_month=days_in_month)
            continue

        allowed = intersect_monthly_atoms_allowed(term, y=y, m=m, dim=dim, allowed=allowed)
        if not allowed:
            probe = _first_day_next_month(y, m, date_cls=date_cls, days_in_month=days_in_month)
            continue

        allowed = _intersect_weekly_atoms_allowed(
            term,
            y=y,
            m=m,
            allowed=allowed,
            doms_for_weekly_spec=doms_for_weekly_spec,
        )
        if not allowed:
            probe = _first_day_next_month(y, m, date_cls=date_cls, days_in_month=days_in_month)
            continue

        pick = _choose_rand_dom(
            y,
            m,
            allowed,
            seed_base=seed_base,
            identity=identity,
            random_pick_index=random_pick_index,
        )
        if pick is None:
            probe = _first_day_next_month(y, m, date_cls=date_cls, days_in_month=days_in_month)
            continue
        cand = date_cls(y, m, pick)
        if cand > ref_d:
            return cand
        probe = _first_day_next_month(y, m, date_cls=date_cls, days_in_month=days_in_month)
    return None


def next_for_and_fast_path(
    term: list[dict],
    ref_d,
    seed,
    seed_base=None,
    *,
    next_after_atom_with_mods,
    atom_matches_on,
    max_anchor_iter: int,
    warn_once_per_day,
    parse_error_cls,
    os_mod,
):
    probe = ref_d
    stalled = 0
    for _ in range(max_anchor_iter):
        cands = [next_after_atom_with_mods(atom, probe, seed, seed_base=seed_base) for atom in term]
        if not cands:
            raise parse_error_cls("Anchor evaluation term is empty; check anchor spec.")
        if any(candidate is None for candidate in cands):
            return None
        target = max(cands)
        if target <= probe:
            stalled += 1
            if stalled < 3:
                try:
                    probe = probe + timedelta(days=1)
                except OverflowError as exc:
                    raise OccurrenceSearchExhausted(
                        "AND-term scheduling",
                        reference=probe,
                        limit=max_anchor_iter,
                        kind=OccurrenceSearchExhausted.DATE_LIMIT,
                    ) from exc
                continue
            if os_mod.environ.get("NAUTICAL_DIAG") == "1":
                warn_once_per_day(
                    "next_for_and_no_progress",
                    "[nautical] _next_for_and made no progress; failing fast. Check anchor spec.",
                )
            raise parse_error_cls("Anchor evaluation made no forward progress; check anchor spec.")
        stalled = 0
        if all(atom_matches_on(atom, target, seed, seed_base=seed_base) for atom in term):
            return target
        probe = target
    if os_mod.environ.get("NAUTICAL_DIAG") == "1":
        warn_once_per_day(
            "next_for_and_fallback",
            f"[nautical] _next_for_and fallback after {max_anchor_iter} iterations.",
        )
    periodic = _next_for_and_periodic_cycle(
        term,
        ref_d,
        seed,
        seed_base,
        next_after_atom_with_mods=next_after_atom_with_mods,
        atom_matches_on=atom_matches_on,
    )
    if periodic is not None:
        return periodic
    raise OccurrenceSearchExhausted(
        "AND-term scheduling",
        reference=ref_d,
        limit=max_anchor_iter,
        kind=OccurrenceSearchExhausted.SEARCH_LIMIT,
    )


def next_for_and(
    term: list[dict],
    ref_d,
    seed,
    seed_base=None,
    *,
    random_identity,
    random_pick_index,
    days_in_month,
    doms_allowed_by_year,
    intersect_monthly_atoms_allowed,
    doms_for_weekly_spec,
    next_after_atom_with_mods,
    atom_matches_on,
    max_anchor_iter: int,
    warn_once_per_day,
    parse_error_cls,
    os_mod,
    date_cls,
):
    """
    Find the next date > ref_d satisfying ALL atoms in term.
    Rand-aware: if the term contains m:rand and any y:, choose the random
    day from the intersection of ALL constraints for each candidate month.
    Otherwise, fall back to the fast alignment loop.
    """
    has_m_rand = _term_has_monthly_rand(term)
    y_specs = _term_year_specs(term)
    if has_m_rand and y_specs and not _term_has_moon(term):
        rand_yearly = next_for_and_rand_yearly(
            term,
            ref_d,
            y_specs,
            default_seed=seed,
            seed_base=seed_base,
            identity=random_identity(term),
            random_pick_index=random_pick_index,
            days_in_month=days_in_month,
            doms_allowed_by_year=doms_allowed_by_year,
            intersect_monthly_atoms_allowed=intersect_monthly_atoms_allowed,
            doms_for_weekly_spec=doms_for_weekly_spec,
            date_cls=date_cls,
        )
        if rand_yearly is not None:
            return rand_yearly
        raise OccurrenceSearchExhausted(
            "random yearly AND-term scheduling",
            reference=ref_d,
            limit=_rand_search_limit(
                term,
                ref_d,
                lcm(_term_interval_lcm(term, {"m"}), _term_interval_lcm(term, {"y"})),
                unit="month",
            ),
            kind=OccurrenceSearchExhausted.SEARCH_LIMIT,
        )
    return next_for_and_fast_path(
        term,
        ref_d,
        seed,
        seed_base=seed_base,
        next_after_atom_with_mods=next_after_atom_with_mods,
        atom_matches_on=atom_matches_on,
        max_anchor_iter=max_anchor_iter,
        warn_once_per_day=warn_once_per_day,
        parse_error_cls=parse_error_cls,
        os_mod=os_mod,
    )


def next_for_or(dnf: list[list[dict]], ref_d, seed, seed_base=None, *, next_for_and):
    best = None
    exhausted: OccurrenceSearchExhausted | None = None
    for term in dnf:
        try:
            cand = next_for_and(term, ref_d, seed, seed_base=seed_base)
        except OccurrenceSearchExhausted as exc:
            exhausted = exc
            continue
        if cand and cand > ref_d and (best is None or cand < best):
            best = cand
    if best is not None:
        return best
    if exhausted is not None:
        raise exhausted
    return None


def next_after_term(
    term,
    ref_d,
    default_seed,
    seed_base=None,
    *,
    next_after_atom_with_mods,
    atom_matches_on,
    intersection_guard_steps: int,
):
    """Find next date after ref_d that matches all atoms in term."""
    if len(term) == 1:
        atom = term[0]
        nxt = next_after_atom_with_mods(atom, ref_d, default_seed, seed_base=seed_base)
        mods = atom.get("mods") or {}
        hhmm = mods.get("t")
        return nxt, hhmm

    cur = ref_d
    for _ in range(min(intersection_guard_steps, 100)):
        cands = [next_after_atom_with_mods(a, cur, default_seed, seed_base=seed_base) for a in term]
        if any(candidate is None for candidate in cands):
            return None, None
        nxt = max(cands)

        if all(atom_matches_on(a, nxt, default_seed, seed_base=seed_base) for a in term):
            hhmm = None
            for atom in term:
                mods = atom.get("mods") or {}
                if mods.get("t"):
                    tval = mods["t"]
                    if isinstance(tval, list):
                        hhmm = tval[0] if tval else None
                    else:
                        hhmm = tval
                    break
            return nxt, hhmm

        cur = nxt

    periodic = _next_for_and_periodic_cycle(
        term,
        ref_d,
        default_seed,
        seed_base,
        next_after_atom_with_mods=next_after_atom_with_mods,
        atom_matches_on=atom_matches_on,
    )
    if periodic is not None:
        return periodic, None
    raise OccurrenceSearchExhausted(
        "AND-term scheduling",
        reference=ref_d,
        limit=intersection_guard_steps,
        kind=OccurrenceSearchExhausted.SEARCH_LIMIT,
    )


def _is_simple_weekly(dnf, *, active_mod_keys) -> bool:
    if len(dnf) != 1 or len(dnf[0]) != 1:
        return False
    atom = dnf[0][0]
    return (
        atom.get("typ") == "w"
        and "rand" not in (atom.get("spec") or "")
        and atom.get("ival", 1) == 1
        and not active_mod_keys(atom.get("mods"))
    )


def _simple_weekly_next(after_date, weekdays: list) -> object:
    for offset in range(1, 8):
        try:
            cand = after_date + timedelta(days=offset)
        except OverflowError as exc:
            raise OccurrenceSearchExhausted(
                "simple weekly scheduling",
                reference=after_date,
                limit=7,
                kind=OccurrenceSearchExhausted.DATE_LIMIT,
            ) from exc
        if cand.weekday() in weekdays:
            return cand
    try:
        return after_date + timedelta(days=7)
    except OverflowError as exc:
        raise OccurrenceSearchExhausted(
            "simple weekly scheduling",
            reference=after_date,
            limit=7,
            kind=OccurrenceSearchExhausted.DATE_LIMIT,
        ) from exc


def _pick_earlier_candidate(best, best_meta, cand, meta):
    if cand and (best is None or cand < best):
        return cand, meta
    return best, best_meta


def _selected_random_candidates(
    candidates,
    count: int,
    *,
    seed_base,
    domain: str,
    identity: str,
    period: str,
    random_pick_indices,
    date_is_excluded,
):
    pool = [cand for cand in candidates if date_is_excluded is None or not date_is_excluded(cand)]
    if len(pool) < count:
        return []
    indices = random_pick_indices(
        len(pool),
        count,
        seed_base=seed_base,
        domain=domain,
        identity=identity,
        period=period,
    )
    return sorted(pool[idx] for idx in indices)


def _next_after_expr_weekly_rand_candidate(
    term: list[dict],
    term_id: int,
    info: dict,
    after_date,
    default_seed,
    seed_base,
    *,
    random_identity,
    random_pick_indices,
    atom_matches_on,
    date_is_excluded,
    is_business_day=default_is_business_day,
):
    count = int(info.get("count") or 1)
    mods = info.get("mods") or {}
    bd_only = bool(mods.get("bd") or mods.get("wd") is True)
    ival = int(info.get("ival") or 1)
    rand_idx = int(info["atom_idx"])
    seed = default_seed or after_date
    monday = after_date - timedelta(days=after_date.weekday())
    seed_monday = seed - timedelta(days=seed.weekday())

    for _ in range(520):
        week_delta = (monday - seed_monday).days // 7
        if ival <= 1 or week_delta % ival == 0:
            candidates = [monday + timedelta(days=offset) for offset in range(7)]
            if bd_only:
                candidates = [cand for cand in candidates if is_business_day(cand)]
            filtered = []
            for cand in candidates:
                if all(
                    idx == rand_idx
                    or atom_matches_on(atom, cand, default_seed, seed_base=seed_base)
                    for idx, atom in enumerate(term)
                ):
                    filtered.append(cand)
            iso = monday.isocalendar()
            period_key = f"{iso.year:04d}-W{iso.week:02d}"
            selected = _selected_random_candidates(
                filtered,
                count,
                seed_base=seed_base if seed_base is not None else "preview",
                domain=f"weekly:{term_id}",
                identity=random_identity(term),
                period=period_key,
                random_pick_indices=random_pick_indices,
                date_is_excluded=date_is_excluded,
            )
            for choice in selected:
                if choice > after_date:
                    return choice, {"basis": "rand", "rand_period": period_key}
        monday += timedelta(days=7)
    return None, None


def _next_after_expr_monthly_rand_candidate(
    term: list[dict],
    term_id: int,
    info: dict,
    after_date,
    default_seed,
    seed_base,
    *,
    atype,
    next_for_and,
    months_since,
    term_candidates_in_month,
    random_identity,
    random_pick_indices,
    date_is_excluded,
):
    count = int(info.get("count") or 1)
    if count == 1 and any(atype(a) == "y" for a in term):
        cand = next_for_and(term, after_date, default_seed, seed_base=seed_base)
        if cand:
            return cand, {"basis": "rand+yearly"}
        return None, None

    seed_key_base = seed_base if seed_base is not None else "preview"
    mods = info.get("mods") or {}
    bd_only = bool(mods.get("bd"))
    monthly_interval = _term_interval_lcm(term, {"m"})
    yearly_interval = _term_interval_lcm(term, {"y"})

    seed_loc = default_seed or after_date
    y, m = after_date.year, after_date.month

    search_limit = _rand_search_limit(
        term,
        after_date,
        lcm(monthly_interval, yearly_interval),
        unit="month",
    )
    for _ in range(search_limit):
        month_offset = months_since(seed_loc, y, m)
        if month_offset % monthly_interval != 0 or (y - seed_loc.year) % yearly_interval != 0:
            m = 1 if m == 12 else m + 1
            if m == 1:
                y += 1
            continue

        cands = term_candidates_in_month(term, y, m, info["atom_idx"], bd_only)
        if cands:
            period_key = f"{y:04d}{m:02d}"
            selected = _selected_random_candidates(
                cands,
                count,
                seed_base=seed_key_base,
                domain=f"monthly:{term_id}",
                identity=random_identity(term),
                period=period_key,
                random_pick_indices=random_pick_indices,
                date_is_excluded=date_is_excluded,
            )
            for choice in selected:
                if choice > after_date:
                    return choice, {"basis": "rand", "rand_period": period_key}
        m = 1 if m == 12 else m + 1
        if m == 1:
            y += 1

    raise OccurrenceSearchExhausted(
        "monthly random scheduling",
        reference=after_date,
        limit=search_limit,
        kind=OccurrenceSearchExhausted.SEARCH_LIMIT,
    )


def _next_after_expr_yearly_rand_candidate(
    term: list[dict],
    term_id: int,
    info: dict,
    after_date,
    default_seed,
    seed_base,
    *,
    term_candidates_in_month,
    random_identity,
    random_pick_indices,
    date_is_excluded,
):
    seed_key_base = seed_base if seed_base is not None else "preview"
    mods = info.get("mods") or {}
    bd_only = bool(mods.get("bd"))
    target_m = info.get("month", None)
    count = int(info.get("count") or 1)
    yearly_interval = _term_interval_lcm(term, {"y"})
    seed_year = (default_seed or after_date).year
    y = after_date.year

    search_limit = _rand_search_limit(term, after_date, yearly_interval, unit="year")
    for _ in range(search_limit):
        if (y - seed_year) % yearly_interval != 0:
            y += 1
            continue
        if target_m is None:
            by_month = {}
            for mm in range(1, 13):
                month_cands = term_candidates_in_month(term, y, mm, info["atom_idx"], bd_only)
                if month_cands:
                    by_month[mm] = month_cands
            cands = [cand for mm in sorted(by_month) for cand in by_month[mm]]
            period_key = f"{y:04d}"
        else:
            cands = term_candidates_in_month(term, y, int(target_m), info["atom_idx"], bd_only)
            period_key = f"{y:04d}-{int(target_m):02d}"

        if cands:
            selected = _selected_random_candidates(
                cands,
                count,
                seed_base=seed_key_base,
                domain=f"yearly:{term_id}",
                identity=random_identity(term),
                period=period_key,
                random_pick_indices=random_pick_indices,
                date_is_excluded=date_is_excluded,
            )
            for choice in selected:
                if choice > after_date:
                    return choice, {"basis": "rand", "rand_period": period_key}
        y += 1

    raise OccurrenceSearchExhausted(
        "yearly random scheduling",
        reference=after_date,
        limit=search_limit,
        kind=OccurrenceSearchExhausted.SEARCH_LIMIT,
    )


def _next_after_expr_term_candidate(term: list[dict], after_date, default_seed, seed_base, *, next_after_term):
    cand, _ = next_after_term(term, after_date, default_seed, seed_base=seed_base)
    if cand is not None and cand <= after_date:
        cand, _ = next_after_term(term, after_date + timedelta(days=1), default_seed, seed_base=seed_base)
    if cand:
        return cand, {"basis": "term"}
    return None, None


def next_after_expr(
    dnf,
    after_date,
    default_seed=None,
    seed_base=None,
    *,
    active_mod_keys,
    expand_weekly_cached,
    term_rand_info,
    atype,
    next_for_and,
    months_since,
    term_candidates_in_month,
    random_identity,
    random_pick_indices,
    atom_matches_on,
    next_after_term,
    date_is_excluded=None,
    is_business_day=default_is_business_day,
):
    """Return the next matching local date strictly > after_date."""
    if _is_simple_weekly(dnf, active_mod_keys=active_mod_keys):
        atom = dnf[0][0]
        days = expand_weekly_cached(atom["spec"])
        return _simple_weekly_next(after_date, days), {"basis": "simple_weekly"}

    best = None
    best_meta = None
    exhausted: OccurrenceSearchExhausted | None = None

    for term_id, term in enumerate(dnf):
        rk, info = term_rand_info(term)

        if rk == "w":
            try:
                cand, meta = _next_after_expr_weekly_rand_candidate(
                    term,
                    term_id,
                    info,
                    after_date,
                    default_seed,
                    seed_base,
                    random_identity=random_identity,
                    random_pick_indices=random_pick_indices,
                    atom_matches_on=atom_matches_on,
                    date_is_excluded=date_is_excluded,
                    is_business_day=is_business_day,
                )
            except OccurrenceSearchExhausted as exc:
                exhausted = exc
                continue
            best, best_meta = _pick_earlier_candidate(best, best_meta, cand, meta)
            continue

        if rk == "m":
            try:
                cand, meta = _next_after_expr_monthly_rand_candidate(
                    term,
                    term_id,
                    info,
                    after_date,
                    default_seed,
                    seed_base,
                    atype=atype,
                    next_for_and=next_for_and,
                    months_since=months_since,
                    term_candidates_in_month=term_candidates_in_month,
                    random_identity=random_identity,
                    random_pick_indices=random_pick_indices,
                    date_is_excluded=date_is_excluded,
                )
            except OccurrenceSearchExhausted as exc:
                exhausted = exc
                continue
            best, best_meta = _pick_earlier_candidate(best, best_meta, cand, meta)
            continue

        if rk == "y":
            try:
                cand, meta = _next_after_expr_yearly_rand_candidate(
                    term,
                    term_id,
                    info,
                    after_date,
                    default_seed,
                    seed_base,
                    term_candidates_in_month=term_candidates_in_month,
                    random_identity=random_identity,
                    random_pick_indices=random_pick_indices,
                    date_is_excluded=date_is_excluded,
                )
            except OccurrenceSearchExhausted as exc:
                exhausted = exc
                continue
            best, best_meta = _pick_earlier_candidate(best, best_meta, cand, meta)
            continue

        cand, meta = _next_after_expr_term_candidate(
            term,
            after_date,
            default_seed,
            seed_base,
            next_after_term=next_after_term,
        )
        best, best_meta = _pick_earlier_candidate(best, best_meta, cand, meta)

    if best is None and exhausted is not None:
        raise exhausted
    return best, best_meta
