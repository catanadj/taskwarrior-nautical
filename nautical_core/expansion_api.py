"""Core-bound monthly, weekly, and yearly expansion adapters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    core = namespace if namespace is not None else vars(module)
    ttl_lru_cache = core["_ttl_lru_cache"]

    def days_in_month(year: int, month: int) -> int:
        return core["_expansion_support"].days_in_month(
            year,
            month,
            monthrange=core["monthrange"],
        )

    def wd_idx(value: str) -> int | None:
        return core["_expansion_support"].wd_idx(
            value,
            wd_abbr=core["_WD_ABBR"],
        )

    @ttl_lru_cache(maxsize=128)
    def wday_idx_any(value: str) -> int | None:
        return core["_expansion_support"].wday_idx_any(
            value,
            weekdays=core["_WEEKDAYS"],
            wd_idx=wd_idx,
        )

    def weekly_spec_to_wset(spec: str, mods: dict | None = None) -> set[int]:
        return core["_expansion_support"].weekly_spec_to_wset(
            spec,
            mods=mods,
            expand_weekly_aliases=core["_expand_weekly_aliases"],
            split_csv_lower=core["_split_csv_lower"],
            wday_idx_any=wday_idx_any,
        )

    def doms_for_weekly_spec(spec: str, year: int, month: int) -> set[int]:
        return core["_expansion_support"].doms_for_weekly_spec(
            spec,
            year,
            month,
            expand_weekly_aliases=core["_expand_weekly_aliases"],
            split_csv_tokens=core["_split_csv_tokens"],
            wd_idx=wd_idx,
            days_in_month=days_in_month,
        )

    def doms_for_monthly_token(token: str, year: int, month: int) -> set[int]:
        return core["_monthly_support"].doms_for_monthly_token(
            token,
            year,
            month,
            monthly_alias=core["_MONTHLY_ALIAS"],
            days_in_month=days_in_month,
            re_mod=core["re"],
            nth_re=core["_NTH_RE"],
            wd_idx=wd_idx,
        )

    def y_ranges_from_spec(spec: str) -> list[tuple[int, int, int, int]]:
        return core["_expansion_support"].y_ranges_from_spec(
            spec,
            split_csv_lower=core["_split_csv_lower"],
            re_mod=core["re"],
            year_pair=core["_year_pair"],
        )

    def doms_allowed_by_year(year: int, month: int, y_specs: list[str]) -> set[int]:
        return core["_expansion_support"].doms_allowed_by_year(
            year,
            month,
            y_specs,
            y_ranges_from_spec=y_ranges_from_spec,
            days_in_month=days_in_month,
            expand_yearly=core["expand_yearly_cached"],
        )

    def month_allowed_doms_for_monthly_atom(atom: dict, year: int, month: int, dim: int) -> set[int]:
        return core["_monthly_support"].month_allowed_doms_for_monthly_atom(
            atom,
            year,
            month,
            dim,
            split_csv_lower=core["_split_csv_lower"],
            doms_for_monthly_token=doms_for_monthly_token,
        )

    def intersect_monthly_atoms_allowed(term, *, y, m, dim, allowed):
        return core["_monthly_support"].intersect_monthly_atoms_allowed(
            term,
            y=y,
            m=m,
            dim=dim,
            allowed=allowed,
            month_allowed_doms_for_monthly_atom=month_allowed_doms_for_monthly_atom,
        )

    return SimpleNamespace(
        _days_in_month=days_in_month,
        _wd_idx=wd_idx,
        _wday_idx_any=wday_idx_any,
        _weekly_spec_to_wset=weekly_spec_to_wset,
        _doms_for_weekly_spec=doms_for_weekly_spec,
        _doms_for_monthly_token=doms_for_monthly_token,
        _y_ranges_from_spec=y_ranges_from_spec,
        _doms_allowed_by_year=doms_allowed_by_year,
        _month_allowed_doms_for_monthly_atom=month_allowed_doms_for_monthly_atom,
        _intersect_monthly_atoms_allowed=intersect_monthly_atoms_allowed,
    )


__all__ = ("for_core",)
