"""Public scheduler entry points bound to one core facade instance."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


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
        apply_day_offset=module._apply_day_offset_impl,
        next_after_atom_with_mods=module._next_after_atom_with_mods_impl,
        atom_matches_on=module._atom_matches_on_impl,
        next_after_factor=module._next_after_factor_impl,
        factor_matches_on=module._factor_matches_on_impl,
        next_after_term=next_after_term,
        next_after_expr=next_after_expr,
    )


__all__ = ("for_core",)
