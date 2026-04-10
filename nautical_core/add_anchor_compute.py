from __future__ import annotations

from datetime import timedelta, timezone
from typing import Any, Callable


def anchor_step_once(dnf, prev_local_date, interval_seed, seed_base, *, core: Any):
    return anchor_step_once_with_omit(
        dnf,
        prev_local_date,
        interval_seed,
        seed_base,
        omit_dnf=None,
        core=core,
    )


def anchor_step_once_with_omit(dnf, prev_local_date, interval_seed, seed_base, *, omit_dnf, core: Any):
    try:
        anchor_omit = core._import_sibling("anchor_omit")
        nxt_date, _ = anchor_omit.next_after_expr_with_omit(
            dnf,
            prev_local_date,
            default_seed=interval_seed,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            core=core,
            max_skip_iterations=max(getattr(core, "MAX_ANCHOR_ITER", 128), 128),
        )
        if nxt_date is None or nxt_date <= prev_local_date:
            return None
        return nxt_date
    except Exception:
        return None


def anchor_term_fires_on_date(term, d, interval_seed, seed_base, *, core: Any):
    try:
        nxt, _ = core.next_after_expr(
            [term],
            d - timedelta(days=1),
            default_seed=interval_seed,
            seed_base=seed_base,
        )
        return nxt == d
    except Exception:
        return False


def anchor_expr_fires_on_date(dnf, d, interval_seed, seed_base, *, core: Any):
    return anchor_expr_fires_on_date_with_omit(
        dnf,
        d,
        interval_seed,
        seed_base,
        omit_dnf=None,
        core=core,
    )


def anchor_expr_fires_on_date_with_omit(dnf, d, interval_seed, seed_base, *, omit_dnf, core: Any):
    try:
        anchor_omit = core._import_sibling("anchor_omit")
        if anchor_omit.omit_expr_fires_on_date(
            omit_dnf,
            d,
            interval_seed,
            seed_base,
            core=core,
        ):
            return False
        nxt, _ = core.next_after_expr(
            dnf,
            d - timedelta(days=1),
            default_seed=interval_seed,
            seed_base=seed_base,
        )
        return nxt == d
    except Exception:
        return False


def anchor_times_for_date(
    dnf,
    d,
    interval_seed,
    seed_base,
    *,
    core: Any,
    norm_t_mod: Callable[[Any], list[tuple[int, int]]],
):
    times = set()
    for term in dnf:
        if anchor_term_fires_on_date(term, d, interval_seed, seed_base, core=core):
            for atom in term:
                mods = atom.get("mods") or {}
                for hhmm in norm_t_mod(mods.get("t")):
                    times.add(hhmm)
    return sorted(times)


def anchor_pick_occurrence_local(
    dnf,
    ref_dt_local,
    inclusive: bool,
    fallback_hhmm,
    interval_seed,
    seed_base,
    omit_dnf=None,
    *,
    core: Any,
    norm_t_mod: Callable[[Any], list[tuple[int, int]]],
):
    d0 = ref_dt_local.date()
    if anchor_expr_fires_on_date_with_omit(dnf, d0, interval_seed, seed_base, omit_dnf=omit_dnf, core=core):
        tlist = anchor_times_for_date(
            dnf,
            d0,
            interval_seed,
            seed_base,
            core=core,
            norm_t_mod=norm_t_mod,
        ) or [fallback_hhmm]
        for hhmm in tlist:
            cand_utc = core.build_local_datetime(d0, hhmm)
            cand_local = core.to_local(cand_utc)
            if (cand_local >= ref_dt_local) if inclusive else (cand_local > ref_dt_local):
                return cand_local
    try:
        anchor_omit = core._import_sibling("anchor_omit")
        nxt_d, _ = anchor_omit.next_after_expr_with_omit(
            dnf,
            d0,
            default_seed=interval_seed,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            core=core,
            max_skip_iterations=max(getattr(core, "MAX_ANCHOR_ITER", 128), 128),
        )
    except Exception:
        return None
    tlist = anchor_times_for_date(
        dnf,
        nxt_d,
        interval_seed,
        seed_base,
        core=core,
        norm_t_mod=norm_t_mod,
    ) or [fallback_hhmm]
    return core.to_local(core.build_local_datetime(nxt_d, tlist[0]))


def anchor_next_occurrence_after_local_dt(
    dnf,
    after_dt_local,
    fallback_hhmm,
    interval_seed,
    seed_base,
    omit_dnf=None,
    *,
    core: Any,
    norm_t_mod: Callable[[Any], list[tuple[int, int]]],
):
    d0 = after_dt_local.date()
    try:
        anchor_omit = core._import_sibling("anchor_omit")
        nxt_date, _ = anchor_omit.next_after_expr_with_omit(
            dnf,
            d0 - timedelta(days=1),
            default_seed=interval_seed,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            core=core,
            max_skip_iterations=max(getattr(core, "MAX_ANCHOR_ITER", 128), 128),
        )
        if nxt_date == d0:
            tlist = anchor_times_for_date(
                dnf,
                d0,
                interval_seed,
                seed_base,
                core=core,
                norm_t_mod=norm_t_mod,
            ) or [fallback_hhmm]
            for hhmm in tlist:
                cand_utc = core.build_local_datetime(d0, hhmm)
                cand_local = core.to_local(cand_utc)
                if cand_local > after_dt_local:
                    return cand_local
    except Exception:
        pass

    nxt_d = anchor_step_once_with_omit(dnf, d0, interval_seed, seed_base, omit_dnf=omit_dnf, core=core)
    if not nxt_d:
        return None
    tlist = anchor_times_for_date(
        dnf,
        nxt_d,
        interval_seed,
        seed_base,
        core=core,
        norm_t_mod=norm_t_mod,
    ) or [fallback_hhmm]
    return core.to_local(core.build_local_datetime(nxt_d, tlist[0]))


def anchor_until_summary(
    dnf,
    until_dt,
    first_date_local,
    first_hhmm,
    interval_seed,
    seed_base,
    omit_dnf=None,
    *,
    core: Any,
    to_local_cached: Callable[[Any], Any],
    max_preview_iterations: int,
    max_iterations: int,
):
    if not until_dt:
        return None, None
    end_day = to_local_cached(until_dt).date()
    count = 0
    prev = first_date_local - timedelta(days=1)
    last = None
    iterations = 0
    for _ in range(max_preview_iterations):
        if iterations >= max_iterations:
            break
        iterations += 1
        nxt = anchor_step_once_with_omit(dnf, prev, interval_seed, seed_base, omit_dnf=omit_dnf, core=core)
        if not nxt or nxt > end_day:
            break
        count += 1
        last = nxt
        prev = nxt
    exact_until_count = max(0, count - 1)
    if not last:
        return exact_until_count, None
    final_hhmm = core.pick_hhmm_from_dnf_for_date(dnf, last, first_date_local) or first_hhmm
    final_until_dt = core.build_local_datetime(last, final_hhmm).astimezone(timezone.utc)
    return exact_until_count, final_until_dt


def anchor_build_preview(
    dnf,
    first_due_local_dt,
    preview_limit: int,
    until_dt,
    fallback_hhmm,
    interval_seed,
    seed_base,
    omit_dnf=None,
    *,
    core: Any,
    norm_t_mod: Callable[[Any], list[tuple[int, int]]],
):
    preview = []
    colors = ["bright_cyan", "cyan", "bright_blue", "blue", "bright_black"]
    cur_dt = first_due_local_dt
    for i in range(preview_limit):
        nxt_dt = anchor_next_occurrence_after_local_dt(
            dnf,
            cur_dt,
            fallback_hhmm,
            interval_seed,
            seed_base,
            omit_dnf=omit_dnf,
            core=core,
            norm_t_mod=norm_t_mod,
        )
        if not nxt_dt:
            break
        dt_utc = nxt_dt.astimezone(timezone.utc)
        if until_dt and dt_utc > until_dt:
            break
        color = colors[min(i, len(colors) - 1)]
        preview.append(f"[{color}]{core.fmt_dt_local(dt_utc)}[/{color}]")
        cur_dt = nxt_dt
    return preview
