"""Public scheduler entry points bound to one deps facade instance."""

from __future__ import annotations

from datetime import date, timedelta
from dataclasses import dataclass
from functools import lru_cache, partial as _partial
from typing import Any, Callable, Protocol
from .api_bindings import ApiBinding, core_namespace

from .core_context import CoreContext


class SchedulerCallback(Protocol):
    """Callable service port shared by scheduler dependency bundles."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class SchedulerAtomBindingDependencies:
    """Named collaborators needed to bind base atom scheduling."""

    scheduler_atom: Any
    astronomy: Any
    astronomy_config: Any
    expand_weekly_cached_mods: SchedulerCallback
    split_csv_tokens: SchedulerCallback
    with_business_calendar: Callable[..., Any]
    expand_monthly_cached: SchedulerCallback
    expand_yearly_cached: SchedulerCallback
    weekly_rand_pick: SchedulerCallback
    week_monday: SchedulerCallback


@dataclass(frozen=True, slots=True)
class SchedulerAtomDependencies:
    """Explicit collaborators required by the scheduler-atom owner."""

    expand_weekly: SchedulerCallback
    split_csv: SchedulerCallback
    expand_monthly: SchedulerCallback
    expand_yearly: SchedulerCallback
    weekly_random: SchedulerCallback
    week_monday: SchedulerCallback
    resolve_moon: SchedulerCallback


@dataclass(frozen=True, slots=True)
class SchedulerIntervalDependencies:
    """Explicit collaborators required by interval admission logic."""

    weeks_between: SchedulerCallback
    year_index: SchedulerCallback


@dataclass(frozen=True, slots=True)
class SchedulerModifierDependencies:
    """Immutable collaborators for modified-atom scheduling."""

    scheduler_atom: Any
    business_calendar_api: Any
    with_business_calendar: Callable[..., Any]
    base_next_after_atom: SchedulerCallback
    active_mod_keys: SchedulerCallback
    interval_allowed: SchedulerCallback
    advance_probe: SchedulerCallback
    monthly_align: SchedulerCallback
    roll_apply: SchedulerCallback
    day_offset: SchedulerCallback
    accept_roll: SchedulerCallback
    max_anchor_iter: int
    warn_once: SchedulerCallback
    os_mod: Any
    resolve_moon: SchedulerCallback
    moon_matches: SchedulerCallback


@dataclass(frozen=True, slots=True)
class SchedulerFactorDependencies:
    """Explicit collaborators for factor and positional-selection scheduling."""

    position_selection: Any
    business_calendar_api: Any
    with_business_calendar: Callable[..., Any]
    selection_inner_matcher: SchedulerCallback
    apply_selection_date_modifiers: SchedulerCallback
    business_calendar_fingerprint: SchedulerCallback
    next_after_atom_with_mods: SchedulerCallback
    atom_matches_on: SchedulerCallback


@dataclass(frozen=True, slots=True)
class SchedulerRuntimeDependencies:
    """Bound callbacks shared by factor, term, and expression scheduling."""

    scheduler_expr: Any
    business_calendar_api: Any
    with_business_calendar: Callable[..., Any]
    next_after_factor: SchedulerCallback
    factor_matches_on: SchedulerCallback
    next_for_and: SchedulerCallback
    term_candidates_in_month: SchedulerCallback
    next_after_term: SchedulerCallback
    active_mod_keys: SchedulerCallback
    expand_weekly_cached: SchedulerCallback
    term_rand_info: SchedulerCallback
    atype: SchedulerCallback
    months_since: SchedulerCallback
    random_identity: SchedulerCallback
    random_pick_indices: SchedulerCallback
    intersection_guard_steps: int


@dataclass(frozen=True, slots=True)
class SchedulerMonthlyBinding:
    """Bind monthly expansion and interval alignment operations."""

    monthly_support: Any
    with_business_calendar: Callable[..., Any]
    expand_monthly_cached: SchedulerCallback

    def month_doms_safe(self, spec: Any, year: Any, month: Any, business_calendar: Any = None) -> Any:
        return self.monthly_support.month_doms_safe(
            spec,
            year,
            month,
            expand_monthly_cached=self.with_business_calendar(
                self.expand_monthly_cached, business_calendar
            ),
        )

    def month_has_hit(self, spec: Any, year: Any, month: Any, business_calendar: Any = None) -> Any:
        return self.monthly_support.month_has_hit(
            spec,
            year,
            month,
            month_doms_safe=self.with_business_calendar(self.month_doms_safe, business_calendar),
        )

    def first_hit_after_probe_in_month(
        self, spec: Any, year: Any, month: Any, probe: Any, business_calendar: Any = None
    ) -> Any:
        return self.monthly_support.first_hit_after_probe_in_month(
            spec,
            year,
            month,
            probe,
            month_doms_safe=self.with_business_calendar(self.month_doms_safe, business_calendar),
        )

    def next_valid_month_on_or_after(
        self, spec: Any, year: Any, month: Any, business_calendar: Any = None
    ) -> Any:
        return self.monthly_support.next_valid_month_on_or_after(
            spec,
            year,
            month,
            month_has_hit=self.with_business_calendar(self.month_has_hit, business_calendar),
        )

    def advance_k_valid_months(
        self, spec: Any, start_y: Any, start_m: Any, k: Any, business_calendar: Any = None
    ) -> Any:
        return self.monthly_support.advance_k_valid_months(
            spec,
            start_y,
            start_m,
            k,
            next_valid_month_on_or_after=self.with_business_calendar(
                self.next_valid_month_on_or_after, business_calendar
            ),
        )

    def monthly_align_base_for_interval(
        self,
        spec: Any,
        base: Any,
        probe: Any,
        seed: Any,
        ival: Any,
        business_calendar: Any = None,
    ) -> Any:
        return self.monthly_support.monthly_align_base_for_interval(
            spec,
            base,
            probe,
            seed,
            ival,
            month_has_hit=self.with_business_calendar(self.month_has_hit, business_calendar),
            next_valid_month_on_or_after=self.with_business_calendar(
                self.next_valid_month_on_or_after, business_calendar
            ),
            first_hit_after_probe_in_month=self.with_business_calendar(
                self.first_hit_after_probe_in_month, business_calendar
            ),
            advance_k_valid_months=self.with_business_calendar(
                self.advance_k_valid_months, business_calendar
            ),
            month_doms_safe=self.with_business_calendar(self.month_doms_safe, business_calendar),
        )


def _apply_day_offset_impl(
    day: Any,
    mods: Any,
    *,
    business_calendar: Any = None,
    calendar_api: Any,
    schedule_utils: Any,
) -> Any:
    business_calendar = calendar_api.effective_business_calendar(business_calendar)
    return schedule_utils.apply_day_offset(
        day,
        mods,
        business_calendar=business_calendar,
    )


def _resolve_moon_phase_date(
    phase: str,
    reference_day: Any,
    *,
    astronomy: Any,
    astronomy_config: Any,
) -> Any:
    return astronomy.resolve_phase_date(
        phase,
        reference_day,
        config=astronomy_config,
    )


def _moon_phase_matches_date(
    phase: str,
    day: Any,
    *,
    astronomy: Any,
    astronomy_config: Any,
) -> bool:
    return astronomy.phase_matches_date(
        phase,
        day,
        config=astronomy_config,
    )


def _build_expansion_binding(
    *,
    deps: dict[str, Any],
    module: Any,
    namespace: dict[str, Any] | None,
    context: CoreContext | None,
    state: list[Any | None],
) -> Any:
    """Lazily bind expansion operations once for one scheduler composition."""
    binding = state[0]
    if binding is None:
        expansion_api = (
            context.import_sibling("expansion_api")
            if context is not None else deps["_import_sibling"]("expansion_api")
        )
        binding = expansion_api.for_core(
            module=module,
            namespace=namespace,
            context=context,
        )
        state[0] = binding
    return binding


def _base_next_after_atom_impl(
    atom: Any,
    ref_d: Any,
    seed_base: Any = None,
    business_calendar: Any = None,
    *,
    bindings: SchedulerAtomBindingDependencies,
) -> Any:
    scheduler_atom = bindings.scheduler_atom
    resolve_moon = lambda phase, reference_day: _resolve_moon_phase_date(
        phase,
        reference_day,
        astronomy=bindings.astronomy,
        astronomy_config=bindings.astronomy_config,
    )
    atom_deps = SchedulerAtomDependencies(
        expand_weekly=bindings.expand_weekly_cached_mods,
        split_csv=bindings.split_csv_tokens,
        expand_monthly=bindings.with_business_calendar(
            bindings.expand_monthly_cached, business_calendar
        ),
        expand_yearly=bindings.expand_yearly_cached,
        weekly_random=bindings.with_business_calendar(
            bindings.weekly_rand_pick, business_calendar
        ),
        week_monday=bindings.week_monday,
        resolve_moon=resolve_moon,
    )
    return scheduler_atom.base_next_after_atom(
        atom,
        ref_d,
        seed_base=seed_base,
        expand_weekly_cached_mods=atom_deps.expand_weekly,
        split_csv_tokens=atom_deps.split_csv,
        expand_monthly_cached=atom_deps.expand_monthly,
        expand_yearly_cached=atom_deps.expand_yearly,
        weekly_rand_pick=atom_deps.weekly_random,
        week_monday=atom_deps.week_monday,
        date_cls=date,
        resolve_moon_phase_date=atom_deps.resolve_moon,
    )


def _interval_allowed_for_atom(
    typ: Any,
    ival: Any,
    seed: Any,
    cand: Any,
    spec: str = "",
    *,
    deps: SchedulerIntervalDependencies,
    scheduler_atom: Any,
) -> Any:
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
    typ: Any,
    ival: Any,
    seed: Any,
    cand: Any,
    spec: str = "",
    *,
    deps: SchedulerIntervalDependencies,
    scheduler_atom: Any,
) -> Any:
    return scheduler_atom.advance_probe_for_interval_bucket(
        typ,
        ival,
        seed,
        cand,
        weeks_between=deps.weeks_between,
        year_index=deps.year_index,
        date_cls=date,
        spec=spec,
    )


def _next_after_atom_with_mods_impl(
    atom: Any,
    ref_d: Any,
    default_seed: Any,
    seed_base: Any = None,
    business_calendar: Any = None,
    *,
    deps: SchedulerModifierDependencies,
) -> Any:
    business_calendar = deps.business_calendar_api.effective_business_calendar(business_calendar)
    with_business_calendar = deps.with_business_calendar
    return deps.scheduler_atom.next_after_atom_with_mods(
        atom,
        ref_d,
        default_seed,
        seed_base=seed_base,
        active_mod_keys=deps.active_mod_keys,
        base_next_after_atom=with_business_calendar(deps.base_next_after_atom, business_calendar),
        interval_allowed_for_atom=deps.interval_allowed,
        advance_probe_for_interval_bucket=deps.advance_probe,
        monthly_align_base_for_interval=with_business_calendar(deps.monthly_align, business_calendar),
        roll_apply=with_business_calendar(deps.roll_apply, business_calendar),
        apply_day_offset=with_business_calendar(deps.day_offset, business_calendar),
        accept_roll_candidate=deps.accept_roll,
        is_business_day=business_calendar.is_business_day,
        max_anchor_iter=deps.max_anchor_iter,
        warn_once_per_day=deps.warn_once,
        os_mod=deps.os_mod,
        resolve_moon_phase_date=deps.resolve_moon,
        moon_phase_matches_date=deps.moon_matches,
    )


def _atom_matches_on_impl(
    atom: Any,
    day: Any,
    default_seed: Any,
    seed_base: Any = None,
    business_calendar: Any = None,
    *,
    scheduler_atom: Any,
    with_business_calendar: Callable[..., Any],
    next_after_atom_with_mods: Callable[..., Any],
    moon_phase_matches_date: Callable[..., Any],
) -> Any:
    next_atom = with_business_calendar(next_after_atom_with_mods, business_calendar)
    return scheduler_atom.atom_matches_on(
        atom,
        day,
        default_seed,
        seed_base=seed_base,
        next_after_atom_with_mods=next_atom,
        moon_phase_matches_date=moon_phase_matches_date,
    )


def _next_after_factor_impl(
    factor: Any,
    ref_d: Any,
    default_seed: Any,
    seed_base: Any = None,
    business_calendar: Any = None,
    *,
    deps: SchedulerFactorDependencies,
) -> Any:
    if not deps.position_selection.is_selection_node(factor):
        next_atom = deps.with_business_calendar(
            deps.next_after_atom_with_mods,
            business_calendar,
        )
        return next_atom(factor, ref_d, default_seed or ref_d, seed_base=seed_base)
    business_calendar = deps.business_calendar_api.effective_business_calendar(business_calendar)
    return deps.position_selection.next_selected_date_with_modifiers(
        factor,
        ref_d,
        matches_on=deps.selection_inner_matcher(business_calendar),
        apply_modifiers=_partial(
            deps.apply_selection_date_modifiers, business_calendar=business_calendar
        ),
        default_seed=default_seed or ref_d,
        seed_base=seed_base,
        calendar_fingerprint=deps.business_calendar_fingerprint(business_calendar),
    )


def _factor_matches_on_impl(
    factor: Any,
    day: Any,
    default_seed: Any,
    seed_base: Any = None,
    business_calendar: Any = None,
    *,
    deps: SchedulerFactorDependencies,
) -> Any:
    if not deps.position_selection.is_selection_node(factor):
        matches = deps.with_business_calendar(
            deps.atom_matches_on,
            business_calendar,
        )
        return matches(factor, day, default_seed or day, seed_base=seed_base)
    business_calendar = deps.business_calendar_api.effective_business_calendar(business_calendar)
    try:
        previous = day - timedelta(days=1)
    except (OverflowError, ValueError):
        return False
    selected = deps.position_selection.next_selected_date_with_modifiers(
        factor,
        previous,
        matches_on=deps.selection_inner_matcher(business_calendar),
        apply_modifiers=_partial(
            deps.apply_selection_date_modifiers, business_calendar=business_calendar
        ),
        default_seed=default_seed or day,
        seed_base=seed_base,
        calendar_fingerprint=deps.business_calendar_fingerprint(business_calendar),
    )
    return selected == day


def _next_after_term_impl(
    term: Any,
    ref_d: Any,
    default_seed: Any,
    seed_base: Any = None,
    business_calendar: Any = None,
    *,
    runtime_deps: SchedulerRuntimeDependencies,
) -> Any:
    with_business_calendar = runtime_deps.with_business_calendar
    next_atom = with_business_calendar(runtime_deps.next_after_factor, business_calendar)
    matches = with_business_calendar(runtime_deps.factor_matches_on, business_calendar)
    return runtime_deps.scheduler_expr.next_after_term(
        term,
        ref_d,
        default_seed,
        seed_base=seed_base,
        next_after_atom_with_mods=next_atom,
        atom_matches_on=matches,
        intersection_guard_steps=runtime_deps.intersection_guard_steps,
    )


def _next_after_expr_impl(
    dnf: Any,
    after_date: Any,
    default_seed: Any = None,
    seed_base: Any = None,
    date_is_excluded: Any = None,
    business_calendar: Any = None,
    *,
    runtime_deps: SchedulerRuntimeDependencies,
) -> Any:
    scheduler_expr = runtime_deps.scheduler_expr
    business_calendar_api = runtime_deps.business_calendar_api
    with_business_calendar = runtime_deps.with_business_calendar
    next_for_and = runtime_deps.next_for_and
    term_candidates_in_month = runtime_deps.term_candidates_in_month
    factor_matches_on = runtime_deps.factor_matches_on
    next_after_term = runtime_deps.next_after_term
    active_mod_keys = runtime_deps.active_mod_keys
    expand_weekly_cached = runtime_deps.expand_weekly_cached
    term_rand_info = runtime_deps.term_rand_info
    atype = runtime_deps.atype
    months_since = runtime_deps.months_since
    random_identity = runtime_deps.random_identity
    random_pick_indices = runtime_deps.random_pick_indices
    business_calendar = business_calendar_api.effective_business_calendar(business_calendar)
    next_for_and_fn = with_business_calendar(next_for_and, business_calendar)
    term_candidates = with_business_calendar(term_candidates_in_month, business_calendar)
    matches = with_business_calendar(factor_matches_on, business_calendar)
    next_term = with_business_calendar(next_after_term, business_calendar)
    return scheduler_expr.next_after_expr(
        dnf,
        after_date,
        default_seed=default_seed,
        seed_base=seed_base,
        active_mod_keys=active_mod_keys,
        expand_weekly_cached=expand_weekly_cached,
        term_rand_info=term_rand_info,
        atype=atype,
        next_for_and=next_for_and_fn,
        months_since=months_since,
        term_candidates_in_month=term_candidates,
        random_identity=random_identity,
        random_pick_indices=random_pick_indices,
        atom_matches_on=matches,
        next_after_term=next_term,
        date_is_excluded=date_is_excluded,
        is_business_day=business_calendar.is_business_day,
    )


def for_core(module: Any = None, *, namespace: dict[str, Any] | None = None, context: CoreContext | None = None) -> ApiBinding:
    """Create scheduler APIs without sharing state between deps loaders."""
    if context is not None:
        deps = context.namespace
        module = context
    else:
        deps = core_namespace(module, namespace, context, "scheduler_api")
    base_atom_deps = SchedulerAtomBindingDependencies(
        scheduler_atom=deps["_scheduler_atom"],
        astronomy=deps["_astronomy"],
        astronomy_config=deps["ASTRONOMY_CONFIG"],
        expand_weekly_cached_mods=deps["expand_weekly_cached_mods"],
        split_csv_tokens=deps["_split_csv_tokens"],
        with_business_calendar=deps["_with_business_calendar"],
        expand_monthly_cached=deps["expand_monthly_cached"],
        expand_yearly_cached=deps["expand_yearly_cached"],
        weekly_rand_pick=deps["_weekly_rand_pick"],
        week_monday=deps["_week_monday"],
    )
    core_config = (
        context.import_sibling("core_config")
        if context is not None
        else deps["_core_config"]
    )
    warn_once_per_day = core_config.warn_once_per_day
    scheduler_expr = context.import_sibling("scheduler_expr") if context is not None else deps["_scheduler_expr"]
    cached_expansion = context.import_sibling("cached_expansion") if context is not None else deps["_cached_expansion"]
    ttl_lru_cache = deps["_ttl_lru_cache"]
    expansion_binding_state: list[Any | None] = [None]

    def expansion_binding() -> Any:
        return _build_expansion_binding(
            deps=deps,
            module=module,
            namespace=namespace,
            context=context,
            state=expansion_binding_state,
        )

    def weekly_spec_to_wset(spec: str, mods: dict | None = None) -> set[int]:
        return expansion_binding()._weekly_spec_to_wset(spec, mods)

    def doms_allowed_by_year(year: int, month: int, y_specs: list[str]) -> set[int]:
        return expansion_binding()._doms_allowed_by_year(year, month, y_specs)

    def doms_for_weekly_spec(spec: str, year: int, month: int) -> set[int]:
        return expansion_binding()._doms_for_weekly_spec(spec, year, month)

    interval_deps = SchedulerIntervalDependencies(
        weeks_between=deps["_schedule_utils"].weeks_between,
        year_index=deps["_year_index"],
    )

    @ttl_lru_cache(maxsize=128)
    def expand_weekly_cached_impl(spec: str) -> Any:
        return cached_expansion.expand_weekly(
            spec,
            weekly_spec_to_wset=weekly_spec_to_wset,
        )

    @ttl_lru_cache(maxsize=128)
    def expand_weekly_cached_mods_impl(spec: str, bd_only: bool) -> Any:
        return cached_expansion.expand_weekly_mods(
            spec,
            bd_only,
            expand_weekly_cached=expand_weekly_cached_impl,
        )

    @ttl_lru_cache(maxsize=128)
    def expand_yearly_cached_impl(spec: str, year: int) -> Any:
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
    def expand_monthly_cached_impl(spec: str, year: int, month: int, business_calendar: Any = None) -> Any:
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

    monthly_binding = SchedulerMonthlyBinding(
        monthly_support=deps["_monthly_support"],
        with_business_calendar=deps["_with_business_calendar"],
        expand_monthly_cached=expand_monthly_cached_impl,
    )

    def roll_apply_impl(dt: Any, mods: Any, business_calendar: Any = None) -> Any:
        business_calendar = deps["_business_calendar"].effective_business_calendar(business_calendar)
        return deps["_schedule_utils"].roll_apply(
            dt,
            mods,
            parse_error_cls=deps["ParseError"],
            business_calendar=business_calendar,
        )

    month_doms_safe = monthly_binding.month_doms_safe
    month_has_hit = monthly_binding.month_has_hit
    first_hit_after_probe_in_month = monthly_binding.first_hit_after_probe_in_month
    next_valid_month_on_or_after = monthly_binding.next_valid_month_on_or_after
    advance_k_valid_months = monthly_binding.advance_k_valid_months
    monthly_align_base_for_interval = monthly_binding.monthly_align_base_for_interval

    @lru_cache(maxsize=32)
    def selection_inner_matcher(business_calendar: Any) -> Any:
        return _partial(deps["atom_matches_on"], business_calendar=business_calendar)

    def apply_selection_date_modifiers(base: Any, mods: Any, business_calendar: Any = None) -> Any:
        business_calendar = deps["_business_calendar"].effective_business_calendar(business_calendar)
        rolled = deps["roll_apply"](base, mods, business_calendar=business_calendar)
        return deps["apply_day_offset"](rolled, mods, business_calendar=business_calendar)

    # Random candidate and boolean-expression scheduling stay bound to this
    # deps instance.  The callbacks are looked up through ``deps`` at call
    # time so facade monkeypatches continue to affect scheduling.
    def week_monday(day: Any) -> Any:
        return cached_expansion.week_monday(day)

    def weekly_rand_pick(
        iso_year: Any,
        iso_week: Any,
        mods: Any,
        *,
        seed_base: Any,
        atom_identity: Any,
        business_calendar: Any = None,
    ) -> Any:
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

    def is_bd(day: Any, business_calendar: Any = None) -> Any:
        business_calendar = deps["_business_calendar"].effective_business_calendar(business_calendar)
        return cached_expansion.is_bd(day, business_calendar)

    def random_identity(value: Any) -> Any:
        return cached_expansion.random_identity(value)

    def random_pick_index(seq_len: Any, **kwargs: Any) -> Any:
        return cached_expansion.random_pick_index(
            seq_len,
            namespace=deps["WRAND_SALT"],
            **kwargs,
        )

    def random_pick_indices(seq_len: Any, count: Any, **kwargs: Any) -> Any:
        return cached_expansion.random_pick_indices(
            seq_len,
            count,
            namespace=deps["WRAND_SALT"],
            **kwargs,
        )

    def term_rand_info(term: Any) -> Any:
        return cached_expansion.term_rand_info(term)

    def dnf_has_counted_random(dnf: Any) -> Any:
        return cached_expansion.dnf_has_counted_random(dnf)

    def filter_by_w(dt_list: Any, term: Any) -> Any:
        return cached_expansion.filter_by_w(
            dt_list,
            term,
            atype=deps["_atype"],
            aspec=deps["_aspec"],
            weekly_spec_to_wset=weekly_spec_to_wset,
        )

    @ttl_lru_cache(maxsize=128)
    def month_tokens_for_atom_cached(year: Any, month: Any, spec: Any, business_calendar: Any = None) -> Any:
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

    def month_tokens_for_atom(atom: Any, year: Any, month: Any, business_calendar: Any = None) -> Any:
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
        term: Any,
        year: Any,
        month: Any,
        rand_atom_idx: Any,
        bd_only: Any,
        business_calendar: Any = None,
    ) -> Any:
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
            doms_allowed_by_year=doms_allowed_by_year,
        )

    def next_for_and_rand_yearly(term: Any, ref_d: Any, y_specs: Any, seed_base: Any = None) -> Any:
        return scheduler_expr.next_for_and_rand_yearly(
            term,
            ref_d,
            y_specs,
            seed_base=seed_base,
            identity=random_identity(term),
            random_pick_index=random_pick_index,
            days_in_month=deps["_days_in_month"],
            doms_allowed_by_year=doms_allowed_by_year,
            intersect_monthly_atoms_allowed=deps["_intersect_monthly_atoms_allowed"],
            doms_for_weekly_spec=doms_for_weekly_spec,
            date_cls=date,
        )

    def next_for_and_fast_path(term: Any, ref_d: Any, seed: Any, seed_base: Any = None, business_calendar: Any = None) -> Any:
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
            warn_once_per_day=warn_once_per_day,
            parse_error_cls=deps["ParseError"],
            os_mod=deps["os"],
        )

    def next_for_and(term: Any, ref_d: Any, seed: Any, seed_base: Any = None, business_calendar: Any = None) -> Any:
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
            doms_allowed_by_year=doms_allowed_by_year,
            intersect_monthly_atoms_allowed=deps["_intersect_monthly_atoms_allowed"],
            doms_for_weekly_spec=doms_for_weekly_spec,
            next_after_atom_with_mods=next_atom,
            atom_matches_on=matches,
            max_anchor_iter=deps["MAX_ANCHOR_ITER"],
            warn_once_per_day=warn_once_per_day,
            parse_error_cls=deps["ParseError"],
            os_mod=deps["os"],
            date_cls=date,
        )

    def next_for_or(dnf: Any, ref_d: Any, seed: Any, seed_base: Any = None, business_calendar: Any = None) -> Any:
        next_for_and_fn = deps["_with_business_calendar"](next_for_and, business_calendar)
        return scheduler_expr.next_for_or(
            dnf,
            ref_d,
            seed,
            seed_base=seed_base,
            next_for_and=next_for_and_fn,
        )

    def next_after_atom_with_mods(atom: Any, ref_d: Any, default_seed: Any, seed_base: Any = None, business_calendar: Any = None) -> Any:
        return _next_after_atom_with_mods_impl(
            atom,
            ref_d,
            default_seed,
            seed_base=seed_base,
            business_calendar=business_calendar,
            deps=modifier_deps,
        )

    def base_next_after_atom(atom: Any, ref_d: Any, seed_base: Any = None, business_calendar: Any = None) -> Any:
        return _base_next_after_atom_impl(
            atom,
            ref_d,
            seed_base=seed_base,
            bindings=base_atom_deps,
            business_calendar=business_calendar,
        )

    def apply_day_offset(day: Any, mods: Any, business_calendar: Any = None) -> Any:
        return _apply_day_offset_impl(
            day,
            mods,
            business_calendar=business_calendar,
            calendar_api=deps["_business_calendar"],
            schedule_utils=deps["_schedule_utils"],
        )

    def interval_allowed_for_atom(typ: Any, ival: Any, seed: Any, cand: Any, spec: str = "") -> Any:
        return _interval_allowed_for_atom(
            typ, ival, seed, cand, spec=spec, deps=interval_deps, scheduler_atom=deps["_scheduler_atom"]
        )

    def advance_probe_for_interval_bucket(typ: Any, ival: Any, seed: Any, cand: Any, spec: str = "") -> Any:
        return _advance_probe_for_interval_bucket(
            typ, ival, seed, cand, spec=spec, deps=interval_deps, scheduler_atom=deps["_scheduler_atom"]
        )

    def atom_matches_on(atom: Any, d: Any, default_seed: Any, seed_base: Any = None, business_calendar: Any = None) -> Any:
        return _atom_matches_on_impl(
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

    factor_deps = SchedulerFactorDependencies(
        position_selection=deps["_position_selection"],
        business_calendar_api=deps["_business_calendar"],
        with_business_calendar=deps["_with_business_calendar"],
        selection_inner_matcher=selection_inner_matcher,
        apply_selection_date_modifiers=apply_selection_date_modifiers,
        business_calendar_fingerprint=deps["business_calendar_fingerprint"],
        next_after_atom_with_mods=next_after_atom_with_mods,
        atom_matches_on=atom_matches_on,
    )

    def next_after_factor(factor: Any, ref_d: Any, default_seed: Any, seed_base: Any = None, business_calendar: Any = None) -> Any:
        return _next_after_factor_impl(
            factor,
            ref_d,
            default_seed,
            seed_base=seed_base,
            business_calendar=business_calendar,
            deps=factor_deps,
        )

    def factor_matches_on(factor: Any, d: Any, default_seed: Any, seed_base: Any = None, business_calendar: Any = None) -> Any:
        return _factor_matches_on_impl(
            factor,
            d,
            default_seed,
            seed_base=seed_base,
            business_calendar=business_calendar,
            deps=factor_deps,
        )

    def next_after_term(term: Any, ref_d: Any, default_seed: Any, seed_base: Any = None, business_calendar: Any = None) -> Any:
        return _next_after_term_impl(
            term,
            ref_d,
            default_seed,
            seed_base=seed_base,
            business_calendar=business_calendar,
            runtime_deps=runtime_deps,
        )

    def next_after_expr(
        dnf: Any,
        after_date: Any,
        default_seed: Any = None,
        seed_base: Any = None,
        date_is_excluded: Any = None,
        business_calendar: Any = None,
    ) -> Any:
        return _next_after_expr_impl(
            dnf,
            after_date,
            default_seed=default_seed,
            seed_base=seed_base,
            date_is_excluded=date_is_excluded,
            business_calendar=business_calendar,
            runtime_deps=runtime_deps,
        )

    def resolve_moon_phase_date(phase: str, reference_day: Any) -> Any:
        return _resolve_moon_phase_date(
            phase,
            reference_day,
            astronomy=deps["_astronomy"],
            astronomy_config=deps["ASTRONOMY_CONFIG"],
        )

    def moon_phase_matches_date(phase: str, day: Any) -> bool:
        return _moon_phase_matches_date(
            phase,
            day,
            astronomy=deps["_astronomy"],
            astronomy_config=deps["ASTRONOMY_CONFIG"],
        )

    modifier_deps = SchedulerModifierDependencies(
        scheduler_atom=deps["_scheduler_atom"],
        business_calendar_api=deps["_business_calendar"],
        with_business_calendar=deps["_with_business_calendar"],
        base_next_after_atom=base_next_after_atom,
        active_mod_keys=deps["_active_mod_keys"],
        interval_allowed=interval_allowed_for_atom,
        advance_probe=advance_probe_for_interval_bucket,
        monthly_align=monthly_align_base_for_interval,
        roll_apply=roll_apply_impl,
        day_offset=apply_day_offset,
        accept_roll=deps["_scheduler_atom"].accept_roll_candidate,
        max_anchor_iter=deps["MAX_ANCHOR_ITER"],
        warn_once=warn_once_per_day,
        os_mod=deps["os"],
        resolve_moon=resolve_moon_phase_date,
        moon_matches=moon_phase_matches_date,
    )

    runtime_deps = SchedulerRuntimeDependencies(
        scheduler_expr=scheduler_expr,
        business_calendar_api=deps["_business_calendar"],
        with_business_calendar=deps["_with_business_calendar"],
        next_after_factor=next_after_factor,
        factor_matches_on=factor_matches_on,
        next_for_and=next_for_and,
        term_candidates_in_month=term_candidates_in_month,
        next_after_term=next_after_term,
        active_mod_keys=deps["_active_mod_keys"],
        expand_weekly_cached=expand_weekly_cached_impl,
        term_rand_info=term_rand_info,
        atype=deps["_atype"],
        months_since=deps["_months_since"],
        random_identity=random_identity,
        random_pick_indices=random_pick_indices,
        intersection_guard_steps=deps["INTERSECTION_GUARD_STEPS"],
    )

    return ApiBinding.from_kwargs(
        _expand_weekly_cached_impl=expand_weekly_cached_impl,
        _expand_weekly_cached_mods_impl=expand_weekly_cached_mods_impl,
        _expand_yearly_cached_impl=expand_yearly_cached_impl,
        _expand_monthly_cached_impl=expand_monthly_cached_impl,
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
        roll_apply=roll_apply_impl,
        apply_day_offset=apply_day_offset,
        base_next_after_atom=base_next_after_atom,
        interval_allowed_for_atom=interval_allowed_for_atom,
        advance_probe_for_interval_bucket=advance_probe_for_interval_bucket,
        accept_roll_candidate=deps["_scheduler_atom"].accept_roll_candidate,
        next_after_atom_with_mods=next_after_atom_with_mods,
        atom_matches_on=atom_matches_on,
        next_after_factor=next_after_factor,
        factor_matches_on=factor_matches_on,
        next_after_term=next_after_term,
        next_after_expr=next_after_expr,
        _resolve_moon_phase_date=resolve_moon_phase_date,
        _moon_phase_matches_date=moon_phase_matches_date,
    )


__all__ = ("for_core",)
