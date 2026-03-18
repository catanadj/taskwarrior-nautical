from __future__ import annotations

import time
from datetime import datetime, timedelta


def _has_rand_atoms(dnf: list[list[dict]]) -> bool:
    return any(
        "rand" in str((atom.get("spec") or "")).lower()
        for term in (dnf or [])
        for atom in (term or [])
    )


def precompute_hints(
    dnf: list[list[dict]],
    *,
    start_dt,
    rand_seed: str | None,
    k_next: int,
    sample_days_for_year: int,
    now_local,
    next_after_expr,
    next_for_or,
):
    # Operate in local dates; let hooks add times if they prefer.
    today = now_local().date()
    start_d = (start_dt.date() if isinstance(start_dt, datetime) else start_dt) or today

    has_rand = _has_rand_atoms(dnf)

    out_next: list[str] = []
    ref = start_d

    # Keep /N gating stable relative to preview start.
    default_seed = ref
    seed_base = rand_seed or "preview"

    safety_limit = 366 * 5
    steps = 0
    while len(out_next) < k_next and steps < safety_limit:
        if has_rand:
            nxt, _ = next_after_expr(dnf, ref, default_seed=default_seed, seed_base=seed_base)
        else:
            nxt = next_for_or(dnf, ref, default_seed)

        if not nxt or nxt <= ref:
            break

        out_next.append(nxt.isoformat() + "T00:00")
        ref = nxt + timedelta(days=1)
        steps += 1

    year_hits = 0
    first_hit = last_hit = ""
    ref = today
    steps = 0
    seen = set()

    while steps < sample_days_for_year:
        if has_rand:
            nxt, _ = next_after_expr(dnf, ref, default_seed=default_seed, seed_base=seed_base)
        else:
            nxt = next_for_or(dnf, ref, default_seed)

        if not nxt or nxt <= ref:
            break

        iso_s = nxt.isoformat() + "T00:00"
        if not first_hit:
            first_hit = iso_s
        last_hit = iso_s

        if nxt not in seen:
            seen.add(nxt)
            year_hits += 1

        ref = nxt + timedelta(days=1)
        steps += 1

    return {
        "next_dates": out_next,
        "per_year": {"est": year_hits, "first": first_hit, "last": last_hit},
        "limits": {"stop": "none", "max_left": 0, "until": ""},
        "rand_preview": out_next[:10],
    }


def build_and_cache_hints(
    anchor_expr: str,
    *,
    anchor_mode: str,
    default_due_dt,
    cache_key_for_task,
    cache_load,
    validate_anchor_expr_strict,
    describe_anchor_expr_from_dnf,
    precompute_hints,
    cache_save,
    anchor_year_fmt: str,
    wrand_salt: str,
    local_tz_name: str,
    holiday_region: str,
):
    key = cache_key_for_task(anchor_expr, anchor_mode)
    cached = cache_load(key)
    if cached:
        return cached

    dnf = validate_anchor_expr_strict(anchor_expr)
    natural = describe_anchor_expr_from_dnf(dnf, default_due_dt=default_due_dt)
    hints = precompute_hints(dnf, start_dt=default_due_dt, anchor_mode=anchor_mode)

    payload = {
        "meta": {
            "created": int(time.time()),
            "cfg": {
                "fmt": anchor_year_fmt,
                "salt": wrand_salt,
                "tz": local_tz_name,
                "hol": holiday_region,
            },
        },
        "dnf": dnf,
        "natural": natural,
        **hints,
    }
    cache_save(key, payload)
    return payload


def anchors_between_large_range(
    dnf,
    start_excl,
    end_excl,
    default_seed,
    *,
    seed_base=None,
    until_count_cap: int,
    next_after_expr,
):
    acc: list = []
    cur = start_excl
    batch_size = min(100, until_count_cap)

    while len(acc) < until_count_cap and cur < end_excl:
        nxt, _ = next_after_expr(dnf, cur, default_seed, seed_base=seed_base)
        if nxt is None or nxt >= end_excl:
            break
        acc.append(nxt)
        cur = nxt + timedelta(days=1)

        if len(acc) >= batch_size:
            break

    return acc


def anchors_between_expr(
    dnf,
    start_excl,
    end_excl,
    default_seed,
    *,
    seed_base=None,
    until_count_cap: int,
    next_after_expr,
    anchors_between_large_range,
    warn_once_per_day,
    os_mod,
):
    """Find all matching dates between start_excl and end_excl."""
    if start_excl >= end_excl:
        return []

    if (end_excl - start_excl).days > 365 * 2:
        return anchors_between_large_range(
            dnf,
            start_excl,
            end_excl,
            default_seed,
            seed_base=seed_base,
        )

    acc: list = []
    cur = start_excl
    while len(acc) < until_count_cap:
        nxt, _ = next_after_expr(dnf, cur, default_seed, seed_base=seed_base)
        if nxt is None or nxt >= end_excl:
            break
        if nxt <= cur:
            if os_mod.environ.get("NAUTICAL_DIAG") == "1":
                warn_once_per_day(
                    "anchors_between_no_progress",
                    "[nautical] anchors_between_expr made no progress; stopping early.",
                )
            break
        if acc and nxt <= acc[-1]:
            cur = acc[-1] + timedelta(days=1)
            continue
        acc.append(nxt)
        cur = nxt + timedelta(days=1)
    return acc
