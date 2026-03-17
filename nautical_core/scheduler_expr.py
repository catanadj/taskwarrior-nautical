from __future__ import annotations

import hashlib
from datetime import timedelta


def _choose_rand_dom(wrand_salt: str, y: int, m: int, doms: set[int]) -> int | None:
    """Deterministically pick one day from doms using the configured salt."""
    if not doms:
        return None
    pool = sorted(doms)
    digest = hashlib.sha256(f"{wrand_salt}|{y:04d}-{m:02d}".encode("utf-8")).digest()
    idx = int.from_bytes(digest[:8], "big") % len(pool)
    return pool[idx]


def _term_has_monthly_rand(term: list[dict]) -> bool:
    return any(
        (a.get("typ") or a.get("type")) == "m"
        and "rand" in str(a.get("spec") or "").lower()
        for a in term
    )


def _term_year_specs(term: list[dict]) -> list[str]:
    return [str(a.get("spec") or "") for a in term if (a.get("typ") or a.get("type")) == "y"]


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


def next_for_and_rand_yearly(
    term: list[dict],
    ref_d,
    y_specs: list[str],
    *,
    wrand_salt: str,
    days_in_month,
    doms_allowed_by_year,
    intersect_monthly_atoms_allowed,
    doms_for_weekly_spec,
    date_cls,
):
    probe = ref_d + timedelta(days=1)
    for _ in range(60):  # scan up to 5 years (60 months)
        y, m = probe.year, probe.month
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

        pick = _choose_rand_dom(wrand_salt, y, m, allowed)
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
        cands = [next_after_atom_with_mods(atom, probe, seed) for atom in term]
        if not cands:
            raise parse_error_cls("Anchor evaluation term is empty; check anchor spec.")
        target = max(cands)
        if target <= probe:
            stalled += 1
            if stalled < 3:
                probe = probe + timedelta(days=1)
                continue
            if os_mod.environ.get("NAUTICAL_DIAG") == "1":
                warn_once_per_day(
                    "next_for_and_no_progress",
                    "[nautical] _next_for_and made no progress; failing fast. Check anchor spec.",
                )
            raise parse_error_cls("Anchor evaluation made no forward progress; check anchor spec.")
        stalled = 0
        if all(atom_matches_on(atom, target, seed) for atom in term):
            return target
        probe = target
    if os_mod.environ.get("NAUTICAL_DIAG") == "1":
        warn_once_per_day(
            "next_for_and_fallback",
            f"[nautical] _next_for_and fallback after {max_anchor_iter} iterations.",
        )
    return ref_d + timedelta(days=365)


def next_for_and(
    term: list[dict],
    ref_d,
    seed,
    *,
    wrand_salt: str,
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
    if has_m_rand and y_specs:
        rand_yearly = next_for_and_rand_yearly(
            term,
            ref_d,
            y_specs,
            wrand_salt=wrand_salt,
            days_in_month=days_in_month,
            doms_allowed_by_year=doms_allowed_by_year,
            intersect_monthly_atoms_allowed=intersect_monthly_atoms_allowed,
            doms_for_weekly_spec=doms_for_weekly_spec,
            date_cls=date_cls,
        )
        if rand_yearly is not None:
            return rand_yearly
        return ref_d + timedelta(days=365)
    return next_for_and_fast_path(
        term,
        ref_d,
        seed,
        next_after_atom_with_mods=next_after_atom_with_mods,
        atom_matches_on=atom_matches_on,
        max_anchor_iter=max_anchor_iter,
        warn_once_per_day=warn_once_per_day,
        parse_error_cls=parse_error_cls,
        os_mod=os_mod,
    )


def next_for_or(dnf: list[list[dict]], ref_d, seed, *, next_for_and):
    best = None
    for term in dnf:
        cand = next_for_and(term, ref_d, seed)
        if cand and cand > ref_d and (best is None or cand < best):
            best = cand
    return best or (ref_d + timedelta(days=365))


def next_after_term(
    term,
    ref_d,
    default_seed,
    *,
    next_after_atom_with_mods,
    atom_matches_on,
    intersection_guard_steps: int,
):
    """Find next date after ref_d that matches all atoms in term."""
    if len(term) == 1:
        atom = term[0]
        nxt = next_after_atom_with_mods(atom, ref_d, default_seed)
        mods = atom.get("mods") or {}
        hhmm = mods.get("t")
        return nxt, hhmm

    cur = ref_d
    for _ in range(min(intersection_guard_steps, 100)):
        cands = [next_after_atom_with_mods(a, cur, default_seed) for a in term]
        nxt = max(cands)

        if all(atom_matches_on(a, nxt, default_seed) for a in term):
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

    return ref_d + timedelta(days=365), None


def _is_simple_weekly(dnf, *, active_mod_keys) -> bool:
    if len(dnf) != 1 or len(dnf[0]) != 1:
        return False
    atom = dnf[0][0]
    return (
        atom["typ"] == "w"
        and "rand" not in (atom.get("spec") or "")
        and atom.get("ival", 1) == 1
        and not active_mod_keys(atom.get("mods"))
    )


def _simple_weekly_next(after_date, weekdays: list) -> object:
    for offset in range(1, 8):
        cand = after_date + timedelta(days=offset)
        if cand.weekday() in weekdays:
            return cand
    return after_date + timedelta(days=7)


def _pick_earlier_candidate(best, best_meta, cand, meta):
    if cand and (best is None or cand < best):
        return cand, meta
    return best, best_meta


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
    sha_pick,
):
    if any(atype(a) == "y" for a in term):
        cand = next_for_and(term, after_date, default_seed)
        if cand:
            return cand, {"basis": "rand+yearly"}
        return None, None

    seed_key_base = seed_base if seed_base is not None else "preview"
    mods = info.get("mods") or {}
    bd_only = bool(mods.get("bd"))
    ival = int(info.get("ival") or 1)

    seed_loc = default_seed or after_date
    y, m = after_date.year, after_date.month

    for _ in range(24):
        if ival > 1 and ((months_since(seed_loc, y, m) % ival) != 0):
            m = 1 if m == 12 else m + 1
            if m == 1:
                y += 1
            continue

        cands = term_candidates_in_month(term, y, m, info["atom_idx"], bd_only)
        if cands:
            period_key = f"{y:04d}{m:02d}"
            seed_key = f"{seed_key_base}|m|{term_id}|{period_key}"
            idx = sha_pick(len(cands), seed_key)
            choice = cands[idx]
            if choice > after_date:
                return choice, {"basis": "rand", "rand_period": period_key}
        m = 1 if m == 12 else m + 1
        if m == 1:
            y += 1

    return None, None


def _next_after_expr_yearly_rand_candidate(
    term: list[dict],
    term_id: int,
    info: dict,
    after_date,
    seed_base,
    *,
    term_candidates_in_month,
    sha_pick,
):
    seed_key_base = seed_base if seed_base is not None else "preview"
    mods = info.get("mods") or {}
    bd_only = bool(mods.get("bd"))
    target_m = info.get("month", None)
    y = after_date.year

    for _ in range(10):
        if target_m is None:
            cands = []
            for mm in range(1, 13):
                cands.extend(term_candidates_in_month(term, y, mm, info["atom_idx"], bd_only))
            period_key = f"{y:04d}"
        else:
            cands = term_candidates_in_month(term, y, int(target_m), info["atom_idx"], bd_only)
            period_key = f"{y:04d}-{int(target_m):02d}"

        if cands:
            seed_key = f"{seed_key_base}|y|{term_id}|{period_key}"
            idx = sha_pick(len(cands), seed_key)
            choice = cands[idx]
            if choice > after_date:
                return choice, {"basis": "rand", "rand_period": period_key}
        y += 1

    return None, None


def _next_after_expr_term_candidate(term: list[dict], after_date, default_seed, *, next_after_term):
    cand, _ = next_after_term(term, after_date, default_seed)
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
    sha_pick,
    next_after_term,
):
    """Return the next matching local date strictly > after_date."""
    if _is_simple_weekly(dnf, active_mod_keys=active_mod_keys):
        atom = dnf[0][0]
        days = expand_weekly_cached(atom["spec"])
        return _simple_weekly_next(after_date, days), {"basis": "simple_weekly"}

    best = None
    best_meta = None

    for term_id, term in enumerate(dnf):
        rk, info = term_rand_info(term)

        if rk == "m":
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
                sha_pick=sha_pick,
            )
            best, best_meta = _pick_earlier_candidate(best, best_meta, cand, meta)
            continue

        if rk == "y":
            cand, meta = _next_after_expr_yearly_rand_candidate(
                term,
                term_id,
                info,
                after_date,
                seed_base,
                term_candidates_in_month=term_candidates_in_month,
                sha_pick=sha_pick,
            )
            best, best_meta = _pick_earlier_candidate(best, best_meta, cand, meta)
            continue

        cand, meta = _next_after_expr_term_candidate(
            term,
            after_date,
            default_seed,
            next_after_term=next_after_term,
        )
        best, best_meta = _pick_earlier_candidate(best, best_meta, cand, meta)

    return best, best_meta
