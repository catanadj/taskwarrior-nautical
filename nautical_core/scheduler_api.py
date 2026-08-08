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


def _weeks_between(module: Any, d1, d2) -> int:
    return module._schedule_utils.weeks_between(d1, d2)


def _resolve_moon_phase_date(module: Any, phase: str, reference_day):
    return module._astronomy.resolve_phase_date(
        phase,
        reference_day,
        config=module.ASTRONOMY_CONFIG,
    )


def _moon_phase_matches_date(module: Any, phase: str, day) -> bool:
    return module._astronomy.phase_matches_date(
        phase,
        day,
        config=module.ASTRONOMY_CONFIG,
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


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    """Create scheduler APIs without sharing state between core loaders."""
    core = namespace if namespace is not None else vars(module)
    ttl_lru_cache = core["_ttl_lru_cache"]

    @ttl_lru_cache(maxsize=128)
    def expand_weekly_cached_impl(spec: str):
        return core["_cached_expansion"].expand_weekly(
            spec,
            weekly_spec_to_wset=core["_weekly_spec_to_wset"],
        )

    @ttl_lru_cache(maxsize=128)
    def expand_weekly_cached_mods_impl(spec: str, bd_only: bool):
        return core["_cached_expansion"].expand_weekly_mods(
            spec,
            bd_only,
            expand_weekly_cached=expand_weekly_cached_impl,
        )

    @ttl_lru_cache(maxsize=128)
    def expand_yearly_cached_impl(spec: str, year: int):
        return core["_cached_expansion"].expand_yearly(
            spec,
            year,
            rewrite_month_names_to_ranges=core["_rewrite_month_names_to_ranges"],
            split_csv_lower=core["_split_csv_lower"],
            re_mod=core["re"],
            month_len=core["month_len"],
            yearfmt=core["_yearfmt"],
        )

    @ttl_lru_cache(maxsize=128)
    def expand_monthly_cached_impl(spec: str, year: int, month: int, business_calendar=None):
        business_calendar = core["_business_calendar"].effective_business_calendar(business_calendar)
        return core["_cached_expansion"].expand_monthly(
            spec,
            year,
            month,
            month_len=core["month_len"],
            expand_monthly_aliases=core["_expand_monthly_aliases"],
            split_csv_lower=core["_split_csv_lower"],
            nth_weekday_re=core["_nth_weekday_re"],
            bd_re=core["_bd_re"],
            weekday_map=core["_WEEKDAYS"],
            re_mod=core["re"],
            business_calendar=business_calendar,
        )

    def expand_monthly_for_month_impl(spec: str, year: int, month: int):
        return expand_monthly_cached_impl(spec, year, month)

    def expand_weekly_impl(spec: str):
        return expand_weekly_cached_impl(spec)

    def expand_yearly_for_year_strict_impl(spec: str, year: int):
        return expand_yearly_cached_impl(spec, year)

    def roll_apply_impl(dt, mods, business_calendar=None):
        business_calendar = core["_business_calendar"].effective_business_calendar(business_calendar)
        return core["_schedule_utils"].roll_apply(
            dt,
            mods,
            parse_error_cls=core["ParseError"],
            business_calendar=business_calendar,
        )

    def month_doms_safe(spec, year, month, business_calendar=None):
        return core["_monthly_support"].month_doms_safe(
            spec,
            year,
            month,
            expand_monthly_cached=core["_with_business_calendar"](
                expand_monthly_cached_impl,
                business_calendar,
            ),
        )

    def month_has_hit(spec, year, month, business_calendar=None):
        return core["_monthly_support"].month_has_hit(
            spec,
            year,
            month,
            month_doms_safe=core["_with_business_calendar"](month_doms_safe, business_calendar),
        )

    def first_hit_after_probe_in_month(spec, year, month, probe, business_calendar=None):
        return core["_monthly_support"].first_hit_after_probe_in_month(
            spec,
            year,
            month,
            probe,
            month_doms_safe=core["_with_business_calendar"](month_doms_safe, business_calendar),
        )

    def next_valid_month_on_or_after(spec, year, month, business_calendar=None):
        return core["_monthly_support"].next_valid_month_on_or_after(
            spec,
            year,
            month,
            month_has_hit=core["_with_business_calendar"](month_has_hit, business_calendar),
        )

    def advance_k_valid_months(spec, start_y, start_m, k, business_calendar=None):
        return core["_monthly_support"].advance_k_valid_months(
            spec,
            start_y,
            start_m,
            k,
            next_valid_month_on_or_after=core["_with_business_calendar"](
                next_valid_month_on_or_after,
                business_calendar,
            ),
        )

    def monthly_align_base_for_interval(spec, base, probe, seed, ival, business_calendar=None):
        return core["_monthly_support"].monthly_align_base_for_interval(
            spec,
            base,
            probe,
            seed,
            ival,
            month_has_hit=core["_with_business_calendar"](month_has_hit, business_calendar),
            next_valid_month_on_or_after=core["_with_business_calendar"](
                next_valid_month_on_or_after,
                business_calendar,
            ),
            first_hit_after_probe_in_month=core["_with_business_calendar"](
                first_hit_after_probe_in_month,
                business_calendar,
            ),
            advance_k_valid_months=core["_with_business_calendar"](
                advance_k_valid_months,
                business_calendar,
            ),
            month_doms_safe=core["_with_business_calendar"](month_doms_safe, business_calendar),
        )

    @core["lru_cache"](maxsize=32)
    def selection_inner_matcher(business_calendar):
        return core["partial"](core["atom_matches_on"], business_calendar=business_calendar)

    def apply_selection_date_modifiers(base, mods, business_calendar=None):
        business_calendar = core["_business_calendar"].effective_business_calendar(business_calendar)
        rolled = core["roll_apply"](base, mods, business_calendar=business_calendar)
        return core["apply_day_offset"](rolled, mods, business_calendar=business_calendar)

    # Random candidate and boolean-expression scheduling stay bound to this
    # core instance.  The callbacks are looked up through ``core`` at call
    # time so facade monkeypatches continue to affect scheduling.
    def week_monday(day):
        return core["_cached_expansion"].week_monday(day)

    def weekly_rand_pick(
        iso_year,
        iso_week,
        mods,
        *,
        seed_base,
        atom_identity,
        business_calendar=None,
    ):
        business_calendar = core["_business_calendar"].effective_business_calendar(business_calendar)
        return core["_cached_expansion"].weekly_rand_pick(
            iso_year,
            iso_week,
            mods,
            seed_base=seed_base,
            atom_identity=atom_identity,
            namespace=core["WRAND_SALT"],
            business_calendar=business_calendar,
        )

    def is_bd(day, business_calendar=None):
        business_calendar = core["_business_calendar"].effective_business_calendar(business_calendar)
        return core["_cached_expansion"].is_bd(day, business_calendar)

    def random_identity(value):
        return core["_cached_expansion"].random_identity(value)

    def random_pick_index(seq_len, **kwargs):
        return core["_cached_expansion"].random_pick_index(
            seq_len,
            namespace=core["WRAND_SALT"],
            **kwargs,
        )

    def random_pick_indices(seq_len, count, **kwargs):
        return core["_cached_expansion"].random_pick_indices(
            seq_len,
            count,
            namespace=core["WRAND_SALT"],
            **kwargs,
        )

    def term_rand_info(term):
        return core["_cached_expansion"].term_rand_info(term)

    def dnf_has_counted_random(dnf):
        return core["_cached_expansion"].dnf_has_counted_random(dnf)

    def filter_by_w(dt_list, term):
        return core["_cached_expansion"].filter_by_w(
            dt_list,
            term,
            atype=core["_atype"],
            aspec=core["_aspec"],
            weekly_spec_to_wset=core["_weekly_spec_to_wset"],
        )

    @ttl_lru_cache(maxsize=128)
    def month_tokens_for_atom_cached(year, month, spec, business_calendar=None):
        business_calendar = core["_business_calendar"].effective_business_calendar(business_calendar)
        return core["_cached_expansion"].month_tokens_for_atom_values(
            year,
            month,
            spec,
            expand_monthly_aliases=core["_expand_monthly_aliases"],
            days_in_month=core["_days_in_month"],
            bd_re=core["_bd_re"],
            nth_weekday_re=core["_nth_weekday_re"],
            weekday_map=core["_WD"],
            re_mod=core["re"],
            business_calendar=business_calendar,
        )

    def month_tokens_for_atom(atom, year, month, business_calendar=None):
        return core["_cached_expansion"].month_tokens_for_atom(
            atom,
            year,
            month,
            month_tokens_for_atom_cached=core["_with_business_calendar"](
                month_tokens_for_atom_cached,
                business_calendar,
            ),
        )

    def term_candidates_in_month(
        term,
        year,
        month,
        rand_atom_idx,
        bd_only,
        business_calendar=None,
    ):
        return core["_cached_expansion"].term_candidates_in_month(
            term,
            year,
            month,
            rand_atom_idx,
            bd_only,
            days_in_month=core["_days_in_month"],
            is_bd=core["_with_business_calendar"](is_bd, business_calendar),
            filter_by_w=filter_by_w,
            atype=core["_atype"],
            aspec=core["_aspec"],
            month_tokens_for_atom=core["_with_business_calendar"](
                month_tokens_for_atom,
                business_calendar,
            ),
            doms_allowed_by_year=core["_doms_allowed_by_year"],
        )

    def next_for_and_rand_yearly(term, ref_d, y_specs, seed_base=None):
        return core["_scheduler_expr"].next_for_and_rand_yearly(
            term,
            ref_d,
            y_specs,
            seed_base=seed_base,
            identity=random_identity(term),
            random_pick_index=random_pick_index,
            days_in_month=core["_days_in_month"],
            doms_allowed_by_year=core["_doms_allowed_by_year"],
            intersect_monthly_atoms_allowed=core["_intersect_monthly_atoms_allowed"],
            doms_for_weekly_spec=core["_doms_for_weekly_spec"],
            date_cls=core["date"],
        )

    def next_for_and_fast_path(term, ref_d, seed, seed_base=None, business_calendar=None):
        next_atom = core["_with_business_calendar"](core["next_after_factor"], business_calendar)
        matches = core["_with_business_calendar"](core["factor_matches_on"], business_calendar)
        return core["_scheduler_expr"].next_for_and_fast_path(
            term,
            ref_d,
            seed,
            seed_base=seed_base,
            next_after_atom_with_mods=next_atom,
            atom_matches_on=matches,
            max_anchor_iter=core["MAX_ANCHOR_ITER"],
            warn_once_per_day=core["_warn_once_per_day"],
            parse_error_cls=core["ParseError"],
            os_mod=core["os"],
        )

    def next_for_and(term, ref_d, seed, seed_base=None, business_calendar=None):
        next_atom = core["_with_business_calendar"](core["next_after_factor"], business_calendar)
        matches = core["_with_business_calendar"](core["factor_matches_on"], business_calendar)
        return core["_scheduler_expr"].next_for_and(
            term,
            ref_d,
            seed,
            seed_base=seed_base,
            random_identity=random_identity,
            random_pick_index=random_pick_index,
            days_in_month=core["_days_in_month"],
            doms_allowed_by_year=core["_doms_allowed_by_year"],
            intersect_monthly_atoms_allowed=core["_intersect_monthly_atoms_allowed"],
            doms_for_weekly_spec=core["_doms_for_weekly_spec"],
            next_after_atom_with_mods=next_atom,
            atom_matches_on=matches,
            max_anchor_iter=core["MAX_ANCHOR_ITER"],
            warn_once_per_day=core["_warn_once_per_day"],
            parse_error_cls=core["ParseError"],
            os_mod=core["os"],
            date_cls=core["date"],
        )

    def next_for_or(dnf, ref_d, seed, seed_base=None, business_calendar=None):
        next_for_and_fn = core["_with_business_calendar"](next_for_and, business_calendar)
        return core["_scheduler_expr"].next_for_or(
            dnf,
            ref_d,
            seed,
            seed_base=seed_base,
            next_for_and=next_for_and_fn,
        )

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

    def weeks_between(d1, d2) -> int:
        return _weeks_between(module, d1, d2)

    def resolve_moon_phase_date(phase: str, reference_day):
        return _resolve_moon_phase_date(module, phase, reference_day)

    def moon_phase_matches_date(phase: str, day) -> bool:
        return _moon_phase_matches_date(module, phase, day)

    return SimpleNamespace(
        _expand_weekly_cached_impl=expand_weekly_cached_impl,
        _expand_weekly_cached_mods_impl=expand_weekly_cached_mods_impl,
        _expand_yearly_cached_impl=expand_yearly_cached_impl,
        _expand_monthly_cached_impl=expand_monthly_cached_impl,
        _expand_monthly_for_month_impl=expand_monthly_for_month_impl,
        _expand_weekly_impl=expand_weekly_impl,
        _expand_yearly_for_year_strict_impl=expand_yearly_for_year_strict_impl,
        _roll_apply_impl=roll_apply_impl,
        _month_doms_safe=month_doms_safe,
        _month_has_hit=month_has_hit,
        _first_hit_after_probe_in_month=first_hit_after_probe_in_month,
        _next_valid_month_on_or_after=next_valid_month_on_or_after,
        _advance_k_valid_months=advance_k_valid_months,
        _monthly_align_base_for_interval=monthly_align_base_for_interval,
        _selection_inner_matcher=selection_inner_matcher,
        _apply_selection_date_modifiers=apply_selection_date_modifiers,
        _week_monday=week_monday,
        _weekly_rand_pick=weekly_rand_pick,
        _is_bd=is_bd,
        _random_identity=random_identity,
        _random_pick_index=random_pick_index,
        _random_pick_indices=random_pick_indices,
        _term_rand_info=term_rand_info,
        dnf_has_counted_random=dnf_has_counted_random,
        _filter_by_w=filter_by_w,
        _month_tokens_for_atom_cached=month_tokens_for_atom_cached,
        _month_tokens_for_atom=month_tokens_for_atom,
        _term_candidates_in_month=term_candidates_in_month,
        _next_for_and_rand_yearly=next_for_and_rand_yearly,
        _next_for_and_fast_path=next_for_and_fast_path,
        _next_for_and=next_for_and,
        _next_for_or=next_for_or,
        expand_weekly_cached=expand_weekly_cached_impl,
        expand_weekly_cached_mods=expand_weekly_cached_mods_impl,
        expand_yearly_cached=expand_yearly_cached_impl,
        expand_monthly_cached=expand_monthly_cached_impl,
        expand_monthly_for_month=expand_monthly_for_month_impl,
        expand_weekly=expand_weekly_impl,
        expand_yearly_for_year_strict=expand_yearly_for_year_strict_impl,
        roll_apply=roll_apply_impl,
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
        _weeks_between=weeks_between,
        _resolve_moon_phase_date=resolve_moon_phase_date,
        _moon_phase_matches_date=moon_phase_matches_date,
    )


__all__ = ("for_core",)
