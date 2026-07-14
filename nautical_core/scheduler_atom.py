from __future__ import annotations

import json
import re
from datetime import timedelta

from .business_calendar import is_business_day as default_is_business_day


def base_next_after_atom(
    atom,
    ref_d,
    seed_base=None,
    *,
    expand_weekly_cached_mods,
    split_csv_tokens,
    expand_monthly_cached,
    expand_yearly_cached,
    weekly_rand_pick,
    week_monday,
    date_cls,
) -> object:
    typ = (atom.get("typ") or "").lower()
    spec = (atom.get("spec") or "").lower()
    mods = atom.get("mods") or {}

    if typ == "w" and "rand" in spec:
        atom_identity = json.dumps(
            {
                "typ": typ,
                "spec": spec,
                "ival": atom.get("ival", 1),
                "mods": mods,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        p = ref_d + timedelta(days=1)
        for _ in range(366):
            iso = p.isocalendar()
            dow = weekly_rand_pick(
                iso.year,
                iso.week,
                mods,
                seed_base=seed_base,
                atom_identity=atom_identity,
            )
            mon = week_monday(p)
            if dow is None:
                p = mon + timedelta(days=7)
                continue
            dt = mon + timedelta(days=dow)
            if dt > ref_d:
                return dt
            p = mon + timedelta(days=7, seconds=1)
        return ref_d + timedelta(days=7)

    if typ == "w":
        days = expand_weekly_cached_mods(spec, False)
        if not days:
            return ref_d + timedelta(days=365)
        for i in range(1, 15):
            cand = ref_d + timedelta(days=i)
            if cand.weekday() in days:
                return cand
        return ref_d + timedelta(days=7)

    if typ == "m":
        y, m = ref_d.year, ref_d.month
        tokens = split_csv_tokens(spec)
        for _ in range(24):
            doms_union = set()
            for tok in tokens:
                try:
                    for d0 in expand_monthly_cached(tok, y, m):
                        doms_union.add(d0)
                except Exception:
                    pass
            for d0 in sorted(doms_union):
                cand = date_cls(y, m, d0)
                if cand > ref_d:
                    return cand
            m = 1 if m == 12 else (m + 1)
            if m == 1:
                y += 1
        return ref_d + timedelta(days=365)

    if typ == "y":
        y = ref_d.year
        for _ in range(12):
            days = expand_yearly_cached(spec, y)
            for cand in days:
                if cand > ref_d:
                    return cand
            y += 1
        return ref_d + timedelta(days=366)

    return ref_d + timedelta(days=365)


def _is_pure_iso_week_spec(spec: str) -> bool:
    tokens = [token.strip() for token in str(spec or "").lower().split(",") if token.strip()]
    return bool(tokens) and all(
        re.fullmatch(r"w-?[1-9]\d?(?:\.\.w-?[1-9]\d?)?", token)
        for token in tokens
    )


def interval_allowed_for_atom(
    typ: str,
    ival: int,
    seed,
    cand,
    *,
    weeks_between,
    year_index,
    spec: str = "",
) -> bool:
    if ival <= 1:
        return True
    if typ == "w":
        weeks_diff = weeks_between(seed, cand)
        return weeks_diff % ival == 0
    if typ == "y":
        if _is_pure_iso_week_spec(spec):
            return (cand.isocalendar().year - seed.isocalendar().year) % ival == 0
        return (year_index(cand) - year_index(seed)) % ival == 0
    return True


def advance_probe_for_interval_bucket(
    typ: str,
    ival: int,
    seed,
    cand,
    *,
    weeks_between,
    year_index,
    date_cls,
    spec: str = "",
):
    if ival <= 1:
        return cand
    if typ == "w":
        cur_monday = cand - timedelta(days=cand.weekday())
        weeks_from_seed = weeks_between(seed, cur_monday)
        diff = weeks_from_seed % ival
        add_weeks = (ival - diff) if diff != 0 else 0
        next_allowed_monday = cur_monday + timedelta(weeks=add_weeks or ival)
        return next_allowed_monday - timedelta(days=1)
    if typ == "y":
        if _is_pure_iso_week_spec(spec):
            seed_iso_year = seed.isocalendar().year
            candidate_iso_year = cand.isocalendar().year
            diff = (candidate_iso_year - seed_iso_year) % ival
            next_iso_year = candidate_iso_year + (ival - diff)
            return date_cls.fromisocalendar(next_iso_year, 1, 1) - timedelta(days=1)
        diff = (year_index(cand) - year_index(seed)) % ival
        add_y = (ival - diff) if diff != 0 else 0
        next_jan1 = date_cls(cand.year + (add_y or ival), 1, 1)
        return next_jan1 - timedelta(days=1)
    return cand


def accept_roll_candidate(ref_d, base, cand, roll_kind: str | None) -> bool:
    if roll_kind in ("pbd", "nbd", "nw"):
        return base > ref_d and cand >= ref_d
    return cand > ref_d


def _atom_match_lookback_days(atom) -> int:
    mods = atom.get("mods") or {}
    roll_kind = mods.get("roll")
    day_offset = int(mods.get("day_offset", 0) or 0)
    business_day_offset = int(mods.get("business_day_offset", 0) or 0)
    lookback = 5 + max(0, day_offset) + (2 * max(0, business_day_offset))
    if roll_kind == "next-wd":
        lookback += 7
    elif roll_kind in ("nbd", "nw"):
        lookback += 2
    return lookback


def next_after_atom_with_mods(
    atom,
    ref_d,
    default_seed,
    seed_base=None,
    *,
    active_mod_keys,
    base_next_after_atom,
    interval_allowed_for_atom,
    advance_probe_for_interval_bucket,
    monthly_align_base_for_interval,
    roll_apply,
    apply_day_offset,
    accept_roll_candidate,
    is_business_day=default_is_business_day,
    max_anchor_iter: int,
    warn_once_per_day,
    os_mod,
) -> object:
    ival = int(atom.get("ival", 1) or 1)
    if ival > 100:
        ival = 100

    seed = default_seed or ref_d
    typ = atom["typ"]
    mods = atom.get("mods") or {}
    spec = atom.get("spec") or ""
    roll_kind = mods.get("roll")
    probe = ref_d - timedelta(days=1) if roll_kind == "next-wd" else ref_d

    if ival == 1 and not active_mod_keys(mods):
        candidate = base_next_after_atom(atom, ref_d, seed_base=seed_base)
        if candidate > ref_d:
            return candidate

    for _ in range(max_anchor_iter):
        base = base_next_after_atom(atom, probe, seed_base=seed_base)
        if (mods.get("bd") or mods.get("wd") is True) and not is_business_day(base):
            probe = base + timedelta(days=1)
            continue
        if typ in ("w", "y") and not interval_allowed_for_atom(
            typ,
            ival,
            seed,
            base,
            spec=spec,
        ):
            probe = advance_probe_for_interval_bucket(typ, ival, seed, base, spec=spec)
            continue
        if typ == "m" and ival > 1:
            base = monthly_align_base_for_interval(spec, base, probe, seed, ival)

        rolled = roll_apply(base, mods)
        cand = apply_day_offset(rolled, mods)
        if accept_roll_candidate(ref_d, base, cand, roll_kind):
            return cand
        probe = base + timedelta(days=1)

    if os_mod.environ.get("NAUTICAL_DIAG") == "1":
        warn_once_per_day(
            "next_after_atom_fallback",
            f"[nautical] next_after_atom_with_mods fallback after {max_anchor_iter} iterations.",
        )
    return ref_d + timedelta(days=365)


def atom_matches_on(atom, d, default_seed, seed_base=None, *, next_after_atom_with_mods) -> bool:
    for k in range(1, _atom_match_lookback_days(atom) + 1):
        if next_after_atom_with_mods(atom, d - timedelta(days=k), default_seed, seed_base=seed_base) == d:
            return True
    return False
