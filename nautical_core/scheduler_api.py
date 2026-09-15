"""Public scheduler entry points bound to one deps facade instance."""

from __future__ import annotations

from datetime import date, timedelta
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable
from .api_bindings import ApiBinding, core_namespace

from .core_context import CoreContext, SchedulerDependencies


@dataclass(frozen=True, slots=True)
class SchedulerAtomDependencies:
    """Explicit collaborators required by the scheduler-atom owner."""

    expand_weekly: Callable[..., Any]
    split_csv: Callable[..., Any]
    expand_monthly: Callable[..., Any]
    expand_yearly: Callable[..., Any]
    weekly_random: Callable[..., Any]
    week_monday: Callable[..., Any]
    resolve_moon: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class SchedulerIntervalDependencies:
    """Explicit collaborators required by interval admission logic."""

    weeks_between: Callable[..., Any]
    year_index: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class SchedulerModifierDependencies:
    """Calendar-bound collaborators for scheduler modifier evaluation."""

    active_mod_keys: Callable[..., Any]
    base_next: Callable[..., Any]
    interval_allowed: Callable[..., Any]
    advance_probe: Callable[..., Any]
    monthly_align: Callable[..., Any]
    roll_apply: Callable[..., Any]
    day_offset: Callable[..., Any]
    accept_roll: Callable[..., Any]
    is_business_day: Callable[..., Any]
    max_anchor_iter: int
    warn_once: Callable[..., Any]
    os_mod: Any
    resolve_moon: Callable[..., Any]
    moon_matches: Callable[..., Any]


def _apply_day_offset_impl(
    module: Any,
    day,
    mods,
    business_calendar=None,
    *,
    calendar_api: Any | None = None,
    schedule_utils: Any | None = None,
):
    calendar_api = calendar_api or module._business_calendar
    schedule_utils = schedule_utils or module._schedule_utils
    business_calendar = calendar_api.effective_business_calendar(business_calendar)
    return schedule_utils.apply_day_offset(
        day,
        mods,
        business_calendar=business_calendar,
    )


def _weeks_between(module: Any, d1, d2) -> int:
    # This helper has no per-deps state; route directly to its owning module
    # instead of traversing the mutable compatibility facade.
    from .schedule_utils import weeks_between

    return weeks_between(d1, d2)


def _resolve_moon_phase_date(
    module: Any,
    phase: str,
    reference_day,
    *,
    astronomy: Any | None = None,
    astronomy_config: Any | None = None,
):
    astronomy = astronomy or module._astronomy
    if astronomy_config is None:
        astronomy_config = module.ASTRONOMY_CONFIG
    return astronomy.resolve_phase_date(
        phase,
        reference_day,
        config=astronomy_config,
    )


def _moon_phase_matches_date(
    module: Any,
    phase: str,
    day,
    *,
    astronomy: Any | None = None,
    astronomy_config: Any | None = None,
) -> bool:
    astronomy = astronomy or module._astronomy
    if astronomy_config is None:
        astronomy_config = module.ASTRONOMY_CONFIG
    return astronomy.phase_matches_date(
        phase,
        day,
        config=astronomy_config,
    )


def _base_next_after_atom_impl(
    module: Any,
    atom,
    ref_d,
    seed_base=None,
    business_calendar=None,
    deps: SchedulerAtomDependencies | None = None,
    *,
    owner_deps: SchedulerDependencies | None = None,
):
    if owner_deps is not None:
        scheduler_atom = owner_deps["_scheduler_atom"]
        resolve_moon = lambda phase, reference_day: _resolve_moon_phase_date(
            module,
            phase,
            reference_day,
            astronomy=owner_deps["_astronomy"],
            astronomy_config=owner_deps["ASTRONOMY_CONFIG"],
        )
        deps = deps or SchedulerAtomDependencies(
            expand_weekly=owner_deps["expand_weekly_cached_mods"],
            split_csv=owner_deps["_split_csv_tokens"],
            expand_monthly=owner_deps["_with_business_calendar"](owner_deps["expand_monthly_cached"], business_calendar),
            expand_yearly=owner_deps["expand_yearly_cached"],
            weekly_random=owner_deps["_with_business_calendar"](owner_deps["_weekly_rand_pick"], business_calendar),
            week_monday=owner_deps["_week_monday"],
            resolve_moon=resolve_moon,
        )
    else:
        scheduler_atom = module.import_sibling("scheduler_atom") if isinstance(module, CoreContext) else module._scheduler_atom
        deps = deps or SchedulerAtomDependencies(
            expand_weekly=module.expand_weekly_cached_mods,
            split_csv=module._split_csv_tokens,
            expand_monthly=module._with_business_calendar(module.expand_monthly_cached, business_calendar),
            expand_yearly=module.expand_yearly_cached,
            weekly_random=module._with_business_calendar(module._weekly_rand_pick, business_calendar),
            week_monday=module._week_monday,
            resolve_moon=module._resolve_moon_phase_date,
        )
    return scheduler_atom.base_next_after_atom(
        atom,
        ref_d,
        seed_base=seed_base,
        expand_weekly_cached_mods=deps.expand_weekly,
        split_csv_tokens=deps.split_csv,
        expand_monthly_cached=deps.expand_monthly,
        expand_yearly_cached=deps.expand_yearly,
        weekly_rand_pick=deps.weekly_random,
        week_monday=deps.week_monday,
        date_cls=date,
        resolve_moon_phase_date=deps.resolve_moon,
    )


def _interval_allowed_for_atom(
    module: Any,
    typ,
    ival,
    seed,
    cand,
    spec="",
    deps: SchedulerIntervalDependencies | None = None,
    *,
    owner_deps: SchedulerDependencies | None = None,
):
    if owner_deps is not None:
        deps = deps or SchedulerIntervalDependencies(
            weeks_between=lambda d1, d2: _weeks_between(module, d1, d2),
            year_index=owner_deps["_year_index"],
        )
        scheduler_atom = owner_deps["_scheduler_atom"]
    else:
        deps = deps or SchedulerIntervalDependencies(
            weeks_between=lambda d1, d2: _weeks_between(module, d1, d2),
            year_index=module._year_index,
        )
        scheduler_atom = module._scheduler_atom
    return scheduler_atom.interval_allowed_for_atom(
        typ,
        ival,
        seed,
        cand,
        weeks_between=deps.weeks_between,
        year_index=deps.year_index,
        spec=spec,
    )


def _advance_probe_for_interval_bucket(
    module: Any,
    typ,
    ival,
    seed,
    cand,
    spec="",
    *,
    owner_deps: SchedulerDependencies | None = None,
):
    scheduler_atom = owner_deps["_scheduler_atom"] if owner_deps is not None else module._scheduler_atom
    if owner_deps is not None:
        weeks_between = lambda d1, d2: _weeks_between(module, d1, d2)
    else:
        weeks_between = module._weeks_between
    year_index = owner_deps["_year_index"] if owner_deps is not None else module._year_index
    return scheduler_atom.advance_probe_for_interval_bucket(
        typ,
        ival,
        seed,
        cand,
        weeks_between=weeks_between,
        year_index=year_index,
        date_cls=date,
        spec=spec,
    )


def _accept_roll_candidate(
    module: Any,
    ref_d,
    base,
    cand,
    roll_kind,
    *,
    scheduler_atom: Any | None = None,
):
    scheduler_atom = scheduler_atom or module._scheduler_atom
    return scheduler_atom.accept_roll_candidate(ref_d, base, cand, roll_kind)


def _next_after_atom_with_mods_impl(
    module: Any,
    atom,
    ref_d,
    default_seed,
    seed_base=None,
    business_calendar=None,
    deps: SchedulerModifierDependencies | None = None,
    *,
    scheduler_atom: Any | None = None,
    business_calendar_api: Any | None = None,
    with_business_calendar: Callable[..., Any] | None = None,
    base_next_after_atom: Callable[..., Any] | None = None,
    monthly_align_base_for_interval: Callable[..., Any] | None = None,
    roll_apply: Callable[..., Any] | None = None,
    apply_day_offset: Callable[..., Any] | None = None,
    active_mod_keys: Callable[..., Any] | None = None,
    max_anchor_iter: int | None = None,
    warn_once_per_day: Callable[..., Any] | None = None,
    os_mod: Any | None = None,
    resolve_moon_phase_date: Callable[..., Any] | None = None,
    moon_phase_matches_date: Callable[..., Any] | None = None,
):
    scheduler_atom = scheduler_atom or module._scheduler_atom
    business_calendar_api = business_calendar_api or module._business_calendar
    with_business_calendar = with_business_calendar or module._with_business_calendar
    base_next_after_atom = base_next_after_atom or module.base_next_after_atom
    monthly_align_base_for_interval = monthly_align_base_for_interval or module._monthly_align_base_for_interval
    roll_apply = roll_apply or module.roll_apply
    apply_day_offset = apply_day_offset or module.apply_day_offset
    active_mod_keys = active_mod_keys or module._active_mod_keys
    max_anchor_iter = max_anchor_iter if max_anchor_iter is not None else module.MAX_ANCHOR_ITER
    warn_once_per_day = warn_once_per_day or module._warn_once_per_day
    os_mod = os_mod or module.os
    resolve_moon_phase_date = resolve_moon_phase_date or module._resolve_moon_phase_date
    moon_phase_matches_date = moon_phase_matches_date or module._moon_phase_matches_date
    business_calendar = business_calendar_api.effective_business_calendar(business_calendar)
    base_next = with_business_calendar(base_next_after_atom, business_calendar)
    monthly_align = with_business_calendar(monthly_align_base_for_interval, business_calendar)
    roll = with_business_calendar(roll_apply, business_calendar)
    day_offset = with_business_calendar(apply_day_offset, business_calendar)
    deps = deps or SchedulerModifierDependencies(
        active_mod_keys=active_mod_keys,
        base_next=base_next,
        interval_allowed=lambda *args, **kwargs: _interval_allowed_for_atom(module, *args, **kwargs),
        advance_probe=lambda *args, **kwargs: _advance_probe_for_interval_bucket(module, *args, **kwargs),
        monthly_align=monthly_align,
        roll_apply=roll,
        day_offset=day_offset,
        accept_roll=lambda *args, **kwargs: _accept_roll_candidate(
            module, *args, scheduler_atom=scheduler_atom, **kwargs
        ),
        is_business_day=business_calendar.is_business_day,
        max_anchor_iter=max_anchor_iter,
        warn_once=warn_once_per_day,
        os_mod=os_mod,
        resolve_moon=resolve_moon_phase_date,
        moon_matches=moon_phase_matches_date,
    )
    return scheduler_atom.next_after_atom_with_mods(
        atom,
        ref_d,
        default_seed,
        seed_base=seed_base,
        active_mod_keys=deps.active_mod_keys,
        base_next_after_atom=deps.base_next,
        interval_allowed_for_atom=deps.interval_allowed,
        advance_probe_for_interval_bucket=deps.advance_probe,
        monthly_align_base_for_interval=deps.monthly_align,
        roll_apply=deps.roll_apply,
        apply_day_offset=deps.day_offset,
        accept_roll_candidate=deps.accept_roll,
        is_business_day=deps.is_business_day,
        max_anchor_iter=deps.max_anchor_iter,
        warn_once_per_day=deps.warn_once,
        os_mod=deps.os_mod,
        resolve_moon_phase_date=deps.resolve_moon,
        moon_phase_matches_date=deps.moon_matches,
    )


def _atom_matches_on_impl(
    module: Any,
    atom,
    day,
    default_seed,
    seed_base=None,
    business_calendar=None,
    *,
    scheduler_atom: Any | None = None,
    with_business_calendar: Callable[..., Any] | None = None,
    next_after_atom_with_mods: Callable[..., Any] | None = None,
    moon_phase_matches_date: Callable[..., Any] | None = None,
):
    scheduler_atom = scheduler_atom or module._scheduler_atom
    with_business_calendar = with_business_calendar or module._with_business_calendar
    next_after_atom_with_mods = next_after_atom_with_mods or module.next_after_atom_with_mods
    moon_phase_matches_date = moon_phase_matches_date or module._moon_phase_matches_date
    next_atom = with_business_calendar(next_after_atom_with_mods, business_calendar)
    return scheduler_atom.atom_matches_on(
        atom,
        day,
        default_seed,
        seed_base=seed_base,
        next_after_atom_with_mods=next_atom,
        moon_phase_matches_date=moon_phase_matches_date,
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
        previous = day - timedelta(days=1)
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


def for_core(module: Any = None, *, namespace: dict[str, Any] | None = None, context: CoreContext | None = None) -> ApiBinding:
    """Create scheduler APIs without sharing state between deps loaders."""
    if context is not None:
        deps = SchedulerDependencies.from_mapping(context.namespace)
        module = context
    else:
        deps = SchedulerDependencies.from_mapping(
            core_namespace(module, namespace, context, "scheduler_api")
        )
    scheduler_expr = context.import_sibling("scheduler_expr") if context is not None else deps["_scheduler_expr"]
    cached_expansion = context.import_sibling("cached_expansion") if context is not None else deps["_cached_expansion"]
    ttl_lru_cache = deps["_ttl_lru_cache"]

    @ttl_lru_cache(maxsize=128)
    def expand_weekly_cached_impl(spec: str):
        return cached_expansion.expand_weekly(
            spec,
            weekly_spec_to_wset=deps["_weekly_spec_to_wset"],
        )

    @ttl_lru_cache(maxsize=128)
    def expand_weekly_cached_mods_impl(spec: str, bd_only: bool):
        return cached_expansion.expand_weekly_mods(
            spec,
            bd_only,
            expand_weekly_cached=expand_weekly_cached_impl,
        )

    @ttl_lru_cache(maxsize=128)
    def expand_yearly_cached_impl(spec: str, year: int):
        return cached_expansion.expand_yearly(
            spec,
            year,
            rewrite_month_names_to_ranges=deps["_rewrite_month_names_to_ranges"],
            split_csv_lower=deps["_split_csv_lower"],
            re_mod=deps["re"],
            month_len=deps["month_len"],
            yearfmt=deps["_yearfmt"],
        )

    @ttl_lru_cache(maxsize=128)
    def expand_monthly_cached_impl(spec: str, year: int, month: int, business_calendar=None):
        business_calendar = deps["_business_calendar"].effective_business_calendar(business_calendar)
        return cached_expansion.expand_monthly(
            spec,
            year,
            month,
            month_len=deps["month_len"],
            expand_monthly_aliases=deps["_expand_monthly_aliases"],
            split_csv_lower=deps["_split_csv_lower"],
            nth_weekday_re=deps["_nth_weekday_re"],
            bd_re=deps["_bd_re"],
            weekday_map=deps["_WEEKDAYS"],
            re_mod=deps["re"],
            business_calendar=business_calendar,
        )

    def expand_monthly_for_month_impl(spec: str, year: int, month: int):
        return expand_monthly_cached_impl(spec, year, month)

    def expand_weekly_impl(spec: str):
        return expand_weekly_cached_impl(spec)

    def expand_yearly_for_year_strict_impl(spec: str, year: int):
        return expand_yearly_cached_impl(spec, year)

    def roll_apply_impl(dt, mods, business_calendar=None):
        business_calendar = deps["_business_calendar"].effective_business_calendar(business_calendar)
        return deps["_schedule_utils"].roll_apply(
            dt,
            mods,
            parse_error_cls=deps["ParseError"],
            business_calendar=business_calendar,
        )

    def month_doms_safe(spec, year, month, business_calendar=None):
        return deps["_monthly_support"].month_doms_safe(
            spec,
            year,
            month,
            expand_monthly_cached=deps["_with_business_calendar"](
                expand_monthly_cached_impl,
                business_calendar,
            ),
        )

    def month_has_hit(spec, year, month, business_calendar=None):
        return deps["_monthly_support"].month_has_hit(
            spec,
            year,
            month,
            month_doms_safe=deps["_with_business_calendar"](month_doms_safe, business_calendar),
        )

    def first_hit_after_probe_in_month(spec, year, month, probe, business_calendar=None):
        return deps["_monthly_support"].first_hit_after_probe_in_month(
            spec,
            year,
            month,
            probe,
            month_doms_safe=deps["_with_business_calendar"](month_doms_safe, business_calendar),
        )

    def next_valid_month_on_or_after(spec, year, month, business_calendar=None):
        return deps["_monthly_support"].next_valid_month_on_or_after(
            spec,
            year,
            month,
            month_has_hit=deps["_with_business_calendar"](month_has_hit, business_calendar),
        )

    def advance_k_valid_months(spec, start_y, start_m, k, business_calendar=None):
        return deps["_monthly_support"].advance_k_valid_months(
            spec,
            start_y,
            start_m,
            k,
            next_valid_month_on_or_after=deps["_with_business_calendar"](
                next_valid_month_on_or_after,
                business_calendar,
            ),
        )

    def monthly_align_base_for_interval(spec, base, probe, seed, ival, business_calendar=None):
        return deps["_monthly_support"].monthly_align_base_for_interval(
            spec,
            base,
            probe,
            seed,
            ival,
            month_has_hit=deps["_with_business_calendar"](month_has_hit, business_calendar),
            next_valid_month_on_or_after=deps["_with_business_calendar"](
                next_valid_month_on_or_after,
                business_calendar,
            ),
            first_hit_after_probe_in_month=deps["_with_business_calendar"](
                first_hit_after_probe_in_month,
                business_calendar,
            ),
            advance_k_valid_months=deps["_with_business_calendar"](
                advance_k_valid_months,
                business_calendar,
            ),
            month_doms_safe=deps["_with_business_calendar"](month_doms_safe, business_calendar),
        )

    @lru_cache(maxsize=32)
    def selection_inner_matcher(business_calendar):
        return deps["partial"](deps["atom_matches_on"], business_calendar=business_calendar)

    def apply_selection_date_modifiers(base, mods, business_calendar=None):
        business_calendar = deps["_business_calendar"].effective_business_calendar(business_calendar)
        rolled = deps["roll_apply"](base, mods, business_calendar=business_calendar)
        return deps["apply_day_offset"](rolled, mods, business_calendar=business_calendar)

    # Random candidate and boolean-expression scheduling stay bound to this
    # deps instance.  The callbacks are looked up through ``deps`` at call
    # time so facade monkeypatches continue to affect scheduling.
    def week_monday(day):
        return cached_expansion.week_monday(day)

    def weekly_rand_pick(
        iso_year,
        iso_week,
        mods,
        *,
        seed_base,
        atom_identity,
        business_calendar=None,
    ):
        business_calendar = deps["_business_calendar"].effective_business_calendar(business_calendar)
        return cached_expansion.weekly_rand_pick(
            iso_year,
            iso_week,
            mods,
            seed_base=seed_base,
            atom_identity=atom_identity,
            namespace=deps["WRAND_SALT"],
            business_calendar=business_calendar,
        )

    def is_bd(day, business_calendar=None):
        business_calendar = deps["_business_calendar"].effective_business_calendar(business_calendar)
        return cached_expansion.is_bd(day, business_calendar)

    def random_identity(value):
        return cached_expansion.random_identity(value)

    def random_pick_index(seq_len, **kwargs):
        return cached_expansion.random_pick_index(
            seq_len,
            namespace=deps["WRAND_SALT"],
            **kwargs,
        )

    def random_pick_indices(seq_len, count, **kwargs):
        return cached_expansion.random_pick_indices(
            seq_len,
            count,
            namespace=deps["WRAND_SALT"],
            **kwargs,
        )

    def term_rand_info(term):
        return cached_expansion.term_rand_info(term)

    def dnf_has_counted_random(dnf):
        return cached_expansion.dnf_has_counted_random(dnf)

    def filter_by_w(dt_list, term):
        return cached_expansion.filter_by_w(
            dt_list,
            term,
            atype=deps["_atype"],
            aspec=deps["_aspec"],
            weekly_spec_to_wset=deps["_weekly_spec_to_wset"],
        )

    @ttl_lru_cache(maxsize=128)
    def month_tokens_for_atom_cached(year, month, spec, business_calendar=None):
        business_calendar = deps["_business_calendar"].effective_business_calendar(business_calendar)
        return cached_expansion.month_tokens_for_atom_values(
            year,
            month,
            spec,
            expand_monthly_aliases=deps["_expand_monthly_aliases"],
            days_in_month=deps["_days_in_month"],
            bd_re=deps["_bd_re"],
            nth_weekday_re=deps["_nth_weekday_re"],
            weekday_map=deps["_WD"],
            re_mod=deps["re"],
            business_calendar=business_calendar,
        )

    def month_tokens_for_atom(atom, year, month, business_calendar=None):
        return cached_expansion.month_tokens_for_atom(
            atom,
            year,
            month,
            month_tokens_for_atom_cached=deps["_with_business_calendar"](
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
        return cached_expansion.term_candidates_in_month(
            term,
            year,
            month,
            rand_atom_idx,
            bd_only,
            days_in_month=deps["_days_in_month"],
            is_bd=deps["_with_business_calendar"](is_bd, business_calendar),
            filter_by_w=filter_by_w,
            atype=deps["_atype"],
            aspec=deps["_aspec"],
            month_tokens_for_atom=deps["_with_business_calendar"](
                month_tokens_for_atom,
                business_calendar,
            ),
            doms_allowed_by_year=deps["_doms_allowed_by_year"],
        )

    def next_for_and_rand_yearly(term, ref_d, y_specs, seed_base=None):
        return scheduler_expr.next_for_and_rand_yearly(
            term,
            ref_d,
            y_specs,
            seed_base=seed_base,
            identity=random_identity(term),
            random_pick_index=random_pick_index,
            days_in_month=deps["_days_in_month"],
            doms_allowed_by_year=deps["_doms_allowed_by_year"],
            intersect_monthly_atoms_allowed=deps["_intersect_monthly_atoms_allowed"],
            doms_for_weekly_spec=deps["_doms_for_weekly_spec"],
            date_cls=date,
        )

    def next_for_and_fast_path(term, ref_d, seed, seed_base=None, business_calendar=None):
        next_atom = deps["_with_business_calendar"](deps["next_after_factor"], business_calendar)
        matches = deps["_with_business_calendar"](deps["factor_matches_on"], business_calendar)
        return scheduler_expr.next_for_and_fast_path(
            term,
            ref_d,
            seed,
            seed_base=seed_base,
            next_after_atom_with_mods=next_atom,
            atom_matches_on=matches,
            max_anchor_iter=deps["MAX_ANCHOR_ITER"],
            warn_once_per_day=deps["_warn_once_per_day"],
            parse_error_cls=deps["ParseError"],
            os_mod=deps["os"],
        )

    def next_for_and(term, ref_d, seed, seed_base=None, business_calendar=None):
        next_atom = deps["_with_business_calendar"](deps["next_after_factor"], business_calendar)
        matches = deps["_with_business_calendar"](deps["factor_matches_on"], business_calendar)
        return scheduler_expr.next_for_and(
            term,
            ref_d,
            seed,
            seed_base=seed_base,
            random_identity=random_identity,
            random_pick_index=random_pick_index,
            days_in_month=deps["_days_in_month"],
            doms_allowed_by_year=deps["_doms_allowed_by_year"],
            intersect_monthly_atoms_allowed=deps["_intersect_monthly_atoms_allowed"],
            doms_for_weekly_spec=deps["_doms_for_weekly_spec"],
            next_after_atom_with_mods=next_atom,
            atom_matches_on=matches,
            max_anchor_iter=deps["MAX_ANCHOR_ITER"],
            warn_once_per_day=deps["_warn_once_per_day"],
            parse_error_cls=deps["ParseError"],
            os_mod=deps["os"],
            date_cls=date,
        )

    def next_for_or(dnf, ref_d, seed, seed_base=None, business_calendar=None):
        next_for_and_fn = deps["_with_business_calendar"](next_for_and, business_calendar)
        return scheduler_expr.next_for_or(
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
            scheduler_atom=deps["_scheduler_atom"],
            business_calendar_api=deps["_business_calendar"],
            with_business_calendar=deps["_with_business_calendar"],
            base_next_after_atom=base_next_after_atom,
            monthly_align_base_for_interval=monthly_align_base_for_interval,
            roll_apply=roll_apply_impl,
            apply_day_offset=apply_day_offset,
            active_mod_keys=deps["_active_mod_keys"],
            max_anchor_iter=deps["MAX_ANCHOR_ITER"],
            warn_once_per_day=deps["_warn_once_per_day"],
            os_mod=deps["os"],
            resolve_moon_phase_date=resolve_moon_phase_date,
            moon_phase_matches_date=moon_phase_matches_date,
        )

    def base_next_after_atom(atom, ref_d, seed_base=None, business_calendar=None):
        return _base_next_after_atom_impl(
            module,
            atom,
            ref_d,
            seed_base=seed_base,
            owner_deps=deps,
            business_calendar=business_calendar,
        )

    def apply_day_offset(day, mods, business_calendar=None):
        return _apply_day_offset_impl(
            module,
            day,
            mods,
            business_calendar=business_calendar,
            calendar_api=deps["_business_calendar"],
            schedule_utils=deps["_schedule_utils"],
        )

    def interval_allowed_for_atom(typ, ival, seed, cand, spec=""):
        return _interval_allowed_for_atom(module, typ, ival, seed, cand, spec=spec, owner_deps=deps)

    def advance_probe_for_interval_bucket(typ, ival, seed, cand, spec=""):
        return _advance_probe_for_interval_bucket(module, typ, ival, seed, cand, spec=spec, owner_deps=deps)

    def accept_roll_candidate(ref_d, base, cand, roll_kind):
        return _accept_roll_candidate(
            module, ref_d, base, cand, roll_kind, scheduler_atom=deps["_scheduler_atom"]
        )

    def atom_matches_on(atom, d, default_seed, seed_base=None, business_calendar=None):
        return _atom_matches_on_impl(
            module,
            atom,
            d,
            default_seed,
            seed_base=seed_base,
            business_calendar=business_calendar,
            scheduler_atom=deps["_scheduler_atom"],
            with_business_calendar=deps["_with_business_calendar"],
            next_after_atom_with_mods=next_after_atom_with_mods,
            moon_phase_matches_date=moon_phase_matches_date,
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
        return _resolve_moon_phase_date(
            module,
            phase,
            reference_day,
            astronomy=deps["_astronomy"],
            astronomy_config=deps["ASTRONOMY_CONFIG"],
        )

    def moon_phase_matches_date(phase: str, day) -> bool:
        return _moon_phase_matches_date(
            module,
            phase,
            day,
            astronomy=deps["_astronomy"],
            astronomy_config=deps["ASTRONOMY_CONFIG"],
        )

    return ApiBinding.from_kwargs(
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
