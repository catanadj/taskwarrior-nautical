"""Public natural-language description entry points for the core facade."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    """Create description APIs bound to one core module instance."""
    core = namespace if namespace is not None else vars(module)
    natural = core["_natural_language"]

    def ordinal(n: int) -> str:
        return natural.ordinal(n)

    def term_collect_mods(term: list) -> dict:
        return natural.term_collect_mods(term)

    def fmt_hhmm_for_term(term: list, default_due_dt):
        return natural.fmt_hhmm_for_term(term, default_due_dt)

    def fmt_weekdays_list(spec: str) -> str:
        return natural.fmt_weekdays_list(
            spec,
            expand_weekly_aliases=core["_expand_weekly_aliases"],
            split_csv_lower=core["_split_csv_lower"],
            wday_idx_any=core["_wday_idx_any"],
        )

    def fmt_monthly_atom(spec: str) -> str:
        return natural.fmt_monthly_atom(
            spec,
            monthly_alias=core["_MONTHLY_ALIAS"],
            safe_match=core["_safe_match"],
            nth_wd_re=core["_nth_wd_re"],
            bd_re=core["_bd_re"],
        )

    def fmt_md(day: int, month: int) -> str:
        fmt = (core.get("ANCHOR_YEAR_FMT") or "DM").upper()
        name = natural._MONTH_ABBR[month - 1]
        return f"{day} {name}" if fmt == "DM" else f"{name} {day}"

    def is_full_month(d1, m1, d2, m2) -> int | None:
        if m1 != m2 or d1 != 1:
            return None
        return m1 if 28 <= d2 <= 31 else None

    def fmt_yearly_atom(token: str) -> str:
        return natural.fmt_yearly_atom(
            token,
            rand_mm_re=core["_rand_mm_re"],
            md_range_re=core["_md_range_re"],
            yearfmt=core["_yearfmt"],
        )

    def describe_monthly_tokens(spec: str):
        return natural.describe_monthly_tokens(spec, split_csv_lower=core["_split_csv_lower"])

    def describe_is_pure_nth_weekday_spec(spec: str):
        return natural.describe_is_pure_nth_weekday_spec(
            spec,
            split_csv_lower=core["_split_csv_lower"],
            safe_match=core["_safe_match"],
            nth_wd_re=core["_nth_wd_re"],
        )

    def describe_is_pure_dom_spec(spec: str):
        return natural.describe_is_pure_dom_spec(spec, split_csv_lower=core["_split_csv_lower"])

    def describe_single_full_month_from_yearly_spec(spec: str):
        return natural.describe_single_full_month_from_yearly_spec(
            spec,
            year_range_colon_re=core["_year_range_colon_re"],
        )

    def describe_term_roll_shift(term) -> str | None:
        return natural.describe_term_roll_shift(term)

    def describe_term_bd_filter(term) -> bool:
        return natural.describe_term_bd_filter(term)

    def describe_roll_suffix(roll: str) -> str:
        return natural.describe_roll_suffix(roll)

    def describe_inject_schedule_suffixes(txt: str, term) -> str:
        return natural.describe_inject_schedule_suffixes(txt, term)

    def describe_anchor_term_collect(term):
        return natural.describe_anchor_term_collect(
            term,
            fmt_weekdays_list=fmt_weekdays_list,
            split_csv_tokens=core["_split_csv_tokens"],
            fmt_monthly_atom=fmt_monthly_atom,
            fmt_yearly_atom=fmt_yearly_atom,
        )

    def describe_anchor_term_fused_month_year(
        term,
        default_due_dt,
        monthly_specs,
        yearly_specs,
        yr_ival: int,
        bd_filter: bool,
        m_parts: list[str],
    ) -> str | None:
        return natural.describe_anchor_term_fused_month_year(
            term,
            default_due_dt,
            monthly_specs,
            yearly_specs,
            yr_ival,
            bd_filter,
            m_parts,
            describe_is_pure_nth_weekday_spec=describe_is_pure_nth_weekday_spec,
            describe_single_full_month_from_yearly_spec=describe_single_full_month_from_yearly_spec,
            fmt_hhmm_for_term=fmt_hhmm_for_term,
        )

    def describe_anchor_term_interval_prefix(wk_ival, mo_ival, yr_ival, monthly_specs):
        return natural.describe_anchor_term_interval_prefix(
            wk_ival,
            mo_ival,
            yr_ival,
            monthly_specs,
            describe_is_pure_nth_weekday_spec=describe_is_pure_nth_weekday_spec,
            describe_is_pure_dom_spec=describe_is_pure_dom_spec,
        )

    def describe_anchor_term_parts(w_phrase, m_parts, y_parts, bd_filter: bool) -> list[str]:
        return natural.describe_anchor_term_parts(w_phrase, m_parts, y_parts, bd_filter)

    def describe_anchor_term(term: list, default_due_dt=None) -> str:
        selections = [factor for factor in term if core["_position_selection"].is_selection_node(factor)]
        if selections:
            selection = selections[0]
            inner = describe_anchor_expr_from_dnf(selection.get("expr") or [], default_due_dt=default_due_dt)
            text = core["_position_selection"].describe_selection(selection, inner)
            hhmm = fmt_hhmm_for_term([selection], default_due_dt)
            if hhmm:
                text += f" at {hhmm}"
            text = describe_inject_schedule_suffixes(text, [selection])
            mods = selection.get("mods") or {}
            roll = mods.get("roll")
            weekday = mods.get("wd")
            if roll in ("next-wd", "prev-wd") and isinstance(weekday, int) and 0 <= weekday < 7:
                direction = "next" if roll == "next-wd" else "previous"
                suffix = f", shifted to the {direction} {natural._WDNAME[weekday]}"
                if " at " in text:
                    head, separator, tail = text.partition(" at ")
                    text = f"{head}{suffix}{separator}{tail}"
                else:
                    text += suffix
            plain_factors = [factor for factor in term if not core["_position_selection"].is_selection_node(factor)]
            if plain_factors:
                constraint = natural.describe_anchor_term(
                    plain_factors,
                    default_due_dt=default_due_dt,
                    fmt_weekdays_list=fmt_weekdays_list,
                    split_csv_tokens=core["_split_csv_tokens"],
                    fmt_monthly_atom=fmt_monthly_atom,
                    fmt_yearly_atom=fmt_yearly_atom,
                    describe_is_pure_nth_weekday_spec=describe_is_pure_nth_weekday_spec,
                    describe_single_full_month_from_yearly_spec=describe_single_full_month_from_yearly_spec,
                    fmt_hhmm_for_term=fmt_hhmm_for_term,
                    describe_is_pure_dom_spec=describe_is_pure_dom_spec,
                )
                if constraint:
                    text += f" that also matches {constraint}"
            return text
        return natural.describe_anchor_term(
            term,
            default_due_dt=default_due_dt,
            fmt_weekdays_list=fmt_weekdays_list,
            split_csv_tokens=core["_split_csv_tokens"],
            fmt_monthly_atom=fmt_monthly_atom,
            fmt_yearly_atom=fmt_yearly_atom,
            describe_is_pure_nth_weekday_spec=describe_is_pure_nth_weekday_spec,
            describe_single_full_month_from_yearly_spec=describe_single_full_month_from_yearly_spec,
            fmt_hhmm_for_term=fmt_hhmm_for_term,
            describe_is_pure_dom_spec=describe_is_pure_dom_spec,
        )

    def describe_anchor_expr_from_dnf(dnf: list, default_due_dt=None) -> str:
        return natural.describe_anchor_expr_from_dnf(
            dnf,
            default_due_dt=default_due_dt,
            describe_anchor_term=describe_anchor_term,
        )

    def describe_anchor_expr_impl(anchor_expr: str, default_due_dt=None) -> str:
        return natural.describe_anchor_expr(
            anchor_expr,
            default_due_dt=default_due_dt,
            parse_anchor_expr_to_dnf_cached=core["parse_anchor_expr_to_dnf_cached"],
            describe_anchor_expr_from_dnf=describe_anchor_expr_from_dnf,
        )

    def term_prevnext_wd(term):
        return natural.term_prevnext_wd(term, wdname=natural._WDNAME)

    def inject_prevnext_phrase(txt: str, term) -> str:
        return natural.inject_prevnext_phrase(txt, term, wdname=natural._WDNAME)

    def join_natural_or_terms(terms: list[str]) -> str:
        return natural.join_natural_or_terms(terms)

    def longest_common_suffix(parts: list[str]) -> str:
        return natural.longest_common_suffix(parts)

    def compress_or_terms_by_clause(terms: list[str], delim: str) -> str | None:
        return natural.compress_or_terms_by_clause(terms, delim)

    def normalize_range_token(token: str) -> str | None:
        return natural.normalize_range_token(
            token,
            safe_match=core["_safe_match"],
            int_range_re=core["_int_range_re"],
        )

    def rand_bucket_time_from_mods(mods: dict) -> str | None:
        return natural.rand_bucket_time_from_mods(mods)

    def rand_bucket_merge_mods(mods: dict, time_str: str | None, bd_flag: bool) -> tuple[str | None, bool]:
        return natural.rand_bucket_merge_mods(mods, time_str, bd_flag)

    def rand_bucket_signature(term: list[dict]) -> tuple | None:
        return natural.rand_bucket_signature(term, normalize_range_token=normalize_range_token)

    def try_bucket_rand_monthly(dnf: list[list[dict]], task: dict) -> str | None:
        return natural.try_bucket_rand_monthly(dnf, task, rand_bucket_signature=rand_bucket_signature)

    return SimpleNamespace(
        _ordinal=ordinal,
        _term_collect_mods=term_collect_mods,
        _fmt_hhmm_for_term=fmt_hhmm_for_term,
        _fmt_weekdays_list=fmt_weekdays_list,
        _fmt_monthly_atom=fmt_monthly_atom,
        _fmt_md=fmt_md,
        _is_full_month=is_full_month,
        _fmt_yearly_atom=fmt_yearly_atom,
        _describe_monthly_tokens=describe_monthly_tokens,
        _describe_is_pure_nth_weekday_spec=describe_is_pure_nth_weekday_spec,
        _describe_is_pure_dom_spec=describe_is_pure_dom_spec,
        _describe_single_full_month_from_yearly_spec=describe_single_full_month_from_yearly_spec,
        _describe_term_roll_shift=describe_term_roll_shift,
        _describe_term_bd_filter=describe_term_bd_filter,
        _describe_roll_suffix=describe_roll_suffix,
        _describe_inject_schedule_suffixes=describe_inject_schedule_suffixes,
        _describe_anchor_term_collect=describe_anchor_term_collect,
        _describe_anchor_term_fused_month_year=describe_anchor_term_fused_month_year,
        _describe_anchor_term_interval_prefix=describe_anchor_term_interval_prefix,
        _describe_anchor_term_parts=describe_anchor_term_parts,
        describe_anchor_term=describe_anchor_term,
        _describe_anchor_expr_from_dnf=describe_anchor_expr_from_dnf,
        _describe_anchor_expr_impl=describe_anchor_expr_impl,
        _term_prevnext_wd=term_prevnext_wd,
        _inject_prevnext_phrase=inject_prevnext_phrase,
        _join_natural_or_terms=join_natural_or_terms,
        _longest_common_suffix=longest_common_suffix,
        _compress_or_terms_by_clause=compress_or_terms_by_clause,
        _describe_anchor_dnf_impl=lambda dnf, task: natural.describe_anchor_dnf(
            dnf,
            task,
            try_bucket_rand_monthly=try_bucket_rand_monthly,
            parse_dt_any=core["parse_dt_any"],
            describe_anchor_term=describe_anchor_term,
        ),
        _normalize_range_token=normalize_range_token,
        _rand_bucket_time_from_mods=rand_bucket_time_from_mods,
        _rand_bucket_merge_mods=rand_bucket_merge_mods,
        _rand_bucket_signature=rand_bucket_signature,
        _try_bucket_rand_monthly=try_bucket_rand_monthly,
        describe_anchor_expr=describe_anchor_expr_impl,
        describe_anchor_dnf=lambda dnf, task: natural.describe_anchor_dnf(
            dnf,
            task,
            try_bucket_rand_monthly=try_bucket_rand_monthly,
            parse_dt_any=core["parse_dt_any"],
            describe_anchor_term=describe_anchor_term,
        ),
    )


__all__ = ("for_core",)
