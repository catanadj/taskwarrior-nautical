"""Public scheduler entry points bound to one core facade instance."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def _apply_day_offset_impl(module: Any, day, mods, business_calendar=None):
    business_calendar = module._business_calendar.effective_business_calendar(business_calendar)
    return module._schedule_utils.apply_day_offset(
        day,
        mods,
        business_calendar=business_calendar,
    )


def _base_next_after_atom_impl(module: Any, atom, ref_d, seed_base=None, business_calendar=None):
    return module._scheduler_atom.base_next_after_atom(
        atom,
        ref_d,
        seed_base=seed_base,
        expand_weekly_cached_mods=module.expand_weekly_cached_mods,
        split_csv_tokens=module._split_csv_tokens,
        expand_monthly_cached=module._with_business_calendar(
            module.expand_monthly_cached,
            business_calendar,
        ),
        expand_yearly_cached=module.expand_yearly_cached,
        weekly_rand_pick=module._with_business_calendar(
            module._weekly_rand_pick,
            business_calendar,
        ),
        week_monday=module._week_monday,
        date_cls=module.date,
        resolve_moon_phase_date=module._resolve_moon_phase_date,
    )


def _interval_allowed_for_atom(module: Any, typ, ival, seed, cand, spec=""):
    return module._scheduler_atom.interval_allowed_for_atom(
        typ,
        ival,
        seed,
        cand,
        weeks_between=module._weeks_between,
        year_index=module._year_index,
        spec=spec,
    )


def _advance_probe_for_interval_bucket(module: Any, typ, ival, seed, cand, spec=""):
    return module._scheduler_atom.advance_probe_for_interval_bucket(
        typ,
        ival,
        seed,
        cand,
        weeks_between=module._weeks_between,
        year_index=module._year_index,
        date_cls=module.date,
        spec=spec,
    )


def _accept_roll_candidate(module: Any, ref_d, base, cand, roll_kind):
    return module._scheduler_atom.accept_roll_candidate(ref_d, base, cand, roll_kind)


def _next_after_atom_with_mods_impl(module: Any, atom, ref_d, default_seed, seed_base=None, business_calendar=None):
    business_calendar = module._business_calendar.effective_business_calendar(business_calendar)
    base_next = module._with_business_calendar(module.base_next_after_atom, business_calendar)
    monthly_align = module._with_business_calendar(module._monthly_align_base_for_interval, business_calendar)
    roll = module._with_business_calendar(module.roll_apply, business_calendar)
    day_offset = module._with_business_calendar(module.apply_day_offset, business_calendar)
    return module._scheduler_atom.next_after_atom_with_mods(
        atom,
        ref_d,
        default_seed,
        seed_base=seed_base,
        active_mod_keys=module._active_mod_keys,
        base_next_after_atom=base_next,
        interval_allowed_for_atom=lambda *args, **kwargs: _interval_allowed_for_atom(module, *args, **kwargs),
        advance_probe_for_interval_bucket=lambda *args, **kwargs: _advance_probe_for_interval_bucket(module, *args, **kwargs),
        monthly_align_base_for_interval=monthly_align,
        roll_apply=roll,
        apply_day_offset=day_offset,
        accept_roll_candidate=lambda *args, **kwargs: _accept_roll_candidate(module, *args, **kwargs),
        is_business_day=business_calendar.is_business_day,
        max_anchor_iter=module.MAX_ANCHOR_ITER,
        warn_once_per_day=module._warn_once_per_day,
        os_mod=module.os,
        resolve_moon_phase_date=module._resolve_moon_phase_date,
        moon_phase_matches_date=module._moon_phase_matches_date,
    )


def _atom_matches_on_impl(module: Any, atom, day, default_seed, seed_base=None, business_calendar=None):
    next_atom = module._with_business_calendar(
        module.next_after_atom_with_mods,
        business_calendar,
    )
    return module._scheduler_atom.atom_matches_on(
        atom,
        day,
        default_seed,
        seed_base=seed_base,
        next_after_atom_with_mods=next_atom,
        moon_phase_matches_date=module._moon_phase_matches_date,
    )


def _next_after_factor_impl(module: Any, factor, ref_d, default_seed, seed_base=None, business_calendar=None):
    if not module._position_selection.is_selection_node(factor):
        next_atom = module._with_business_calendar(
            module.next_after_atom_with_mods,
            business_calendar,
        )
        return next_atom(factor, ref_d, default_seed or ref_d, seed_base=seed_base)
    business_calendar = module._business_calendar.effective_business_calendar(business_calendar)
    return module._position_selection.next_selected_date_with_modifiers(
        factor,
        ref_d,
        matches_on=module._selection_inner_matcher(business_calendar),
        apply_modifiers=module.partial(module._apply_selection_date_modifiers, business_calendar=business_calendar),
        default_seed=default_seed or ref_d,
        seed_base=seed_base,
        calendar_fingerprint=module.business_calendar_fingerprint(business_calendar),
    )


def _factor_matches_on_impl(module: Any, factor, day, default_seed, seed_base=None, business_calendar=None):
    if not module._position_selection.is_selection_node(factor):
        matches = module._with_business_calendar(
            module.atom_matches_on,
            business_calendar,
        )
        return matches(factor, day, default_seed or day, seed_base=seed_base)
    business_calendar = module._business_calendar.effective_business_calendar(business_calendar)
    try:
        previous = day - module.timedelta(days=1)
    except (OverflowError, ValueError):
        return False
    selected = module._position_selection.next_selected_date_with_modifiers(
        factor,
        previous,
        matches_on=module._selection_inner_matcher(business_calendar),
        apply_modifiers=module.partial(module._apply_selection_date_modifiers, business_calendar=business_calendar),
        default_seed=default_seed or day,
        seed_base=seed_base,
        calendar_fingerprint=module.business_calendar_fingerprint(business_calendar),
    )
    return selected == day


def _next_after_term_impl(
    module: Any,
    term,
    ref_d,
    default_seed,
    seed_base=None,
    business_calendar=None,
):
    next_atom = module._with_business_calendar(module.next_after_factor, business_calendar)
    matches = module._with_business_calendar(module.factor_matches_on, business_calendar)
    return module._scheduler_expr.next_after_term(
        term,
        ref_d,
        default_seed,
        seed_base=seed_base,
        next_after_atom_with_mods=next_atom,
        atom_matches_on=matches,
        intersection_guard_steps=module.INTERSECTION_GUARD_STEPS,
    )


def _next_after_expr_impl(
    module: Any,
    dnf,
    after_date,
    default_seed=None,
    seed_base=None,
    date_is_excluded=None,
    business_calendar=None,
):
    business_calendar = module._business_calendar.effective_business_calendar(business_calendar)
    next_for_and_fn = module._with_business_calendar(module._next_for_and, business_calendar)
    term_candidates = module._with_business_calendar(
        module._term_candidates_in_month,
        business_calendar,
    )
    matches = module._with_business_calendar(module.factor_matches_on, business_calendar)
    next_term = module._with_business_calendar(
        lambda term, ref_d, default, seed_base=None, business_calendar=None: _next_after_term_impl(
            module,
            term,
            ref_d,
            default,
            seed_base=seed_base,
            business_calendar=business_calendar,
        ),
        business_calendar,
    )
    return module._scheduler_expr.next_after_expr(
        dnf,
        after_date,
        default_seed=default_seed,
        seed_base=seed_base,
        active_mod_keys=module._active_mod_keys,
        expand_weekly_cached=module.expand_weekly_cached,
        term_rand_info=module._term_rand_info,
        atype=module._atype,
        next_for_and=next_for_and_fn,
        months_since=module._months_since,
        term_candidates_in_month=term_candidates,
        random_identity=module._random_identity,
        random_pick_indices=module._random_pick_indices,
        atom_matches_on=matches,
        next_after_term=next_term,
        date_is_excluded=date_is_excluded,
        is_business_day=business_calendar.is_business_day,
    )


def for_core(module: Any):
    """Create scheduler APIs without sharing state between core loaders."""
    def next_after_atom_with_mods(atom, ref_d, default_seed, seed_base=None, business_calendar=None):
        return _next_after_atom_with_mods_impl(
            module,
            atom,
            ref_d,
            default_seed,
            seed_base=seed_base,
            business_calendar=business_calendar,
        )

    def base_next_after_atom(atom, ref_d, seed_base=None, business_calendar=None):
        return _base_next_after_atom_impl(
            module,
            atom,
            ref_d,
            seed_base=seed_base,
            business_calendar=business_calendar,
        )

    def apply_day_offset(day, mods, business_calendar=None):
        return _apply_day_offset_impl(module, day, mods, business_calendar=business_calendar)

    def interval_allowed_for_atom(typ, ival, seed, cand, spec=""):
        return _interval_allowed_for_atom(module, typ, ival, seed, cand, spec=spec)

    def advance_probe_for_interval_bucket(typ, ival, seed, cand, spec=""):
        return _advance_probe_for_interval_bucket(module, typ, ival, seed, cand, spec=spec)

    def accept_roll_candidate(ref_d, base, cand, roll_kind):
        return _accept_roll_candidate(module, ref_d, base, cand, roll_kind)

    def atom_matches_on(atom, d, default_seed, seed_base=None, business_calendar=None):
        return _atom_matches_on_impl(
            module,
            atom,
            d,
            default_seed,
            seed_base=seed_base,
            business_calendar=business_calendar,
        )

    def next_after_factor(factor, ref_d, default_seed, seed_base=None, business_calendar=None):
        return _next_after_factor_impl(
            module,
            factor,
            ref_d,
            default_seed,
            seed_base=seed_base,
            business_calendar=business_calendar,
        )

    def factor_matches_on(factor, d, default_seed, seed_base=None, business_calendar=None):
        return _factor_matches_on_impl(
            module,
            factor,
            d,
            default_seed,
            seed_base=seed_base,
            business_calendar=business_calendar,
        )

    def next_after_term(term, ref_d, default_seed, seed_base=None, business_calendar=None):
        return _next_after_term_impl(
            module,
            term,
            ref_d,
            default_seed,
            seed_base=seed_base,
            business_calendar=business_calendar,
        )

    def next_after_expr(
        dnf,
        after_date,
        default_seed=None,
        seed_base=None,
        date_is_excluded=None,
        business_calendar=None,
    ):
        return _next_after_expr_impl(
            module,
            dnf,
            after_date,
            default_seed=default_seed,
            seed_base=seed_base,
            date_is_excluded=date_is_excluded,
            business_calendar=business_calendar,
        )

    return SimpleNamespace(
        expand_weekly_cached=module._expand_weekly_cached_impl,
        expand_weekly_cached_mods=module._expand_weekly_cached_mods_impl,
        expand_yearly_cached=module._expand_yearly_cached_impl,
        expand_monthly_cached=module._expand_monthly_cached_impl,
        expand_monthly_for_month=module._expand_monthly_for_month_impl,
        expand_weekly=module._expand_weekly_impl,
        expand_yearly_for_year_strict=module._expand_yearly_for_year_strict_impl,
        roll_apply=module._roll_apply_impl,
        apply_day_offset=apply_day_offset,
        base_next_after_atom=base_next_after_atom,
        interval_allowed_for_atom=interval_allowed_for_atom,
        advance_probe_for_interval_bucket=advance_probe_for_interval_bucket,
        accept_roll_candidate=accept_roll_candidate,
        next_after_atom_with_mods=next_after_atom_with_mods,
        atom_matches_on=atom_matches_on,
        next_after_factor=next_after_factor,
        factor_matches_on=factor_matches_on,
        next_after_term=next_after_term,
        next_after_expr=next_after_expr,
    )


__all__ = ("for_core",)
