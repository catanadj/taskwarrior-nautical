"""Core-bound linting callback assembly."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    core = namespace if namespace is not None else vars(module)
    linting = core["_linting"]

    def iter_y_segments(value: str):
        yield from linting.iter_y_segments(value, re_mod=core["re"])

    def lint_expand_year_month_aliases(value: str) -> str:
        return linting.lint_expand_year_month_aliases(
            value,
            month_from_alias=core["_month_from_alias"],
            year_full_month_range_token=core["_year_full_month_range_token"],
            re_mod=core["re"],
        )

    def lint_check_weekly_delimiter_contract(value: str) -> str | None:
        return linting.lint_check_weekly_delimiter_contract(value, re_mod=core["re"])

    def lint_check_yearly_segments(value: str) -> str | None:
        return linting.lint_check_yearly_segments(
            value,
            yearfmt=core["_yearfmt"],
            iter_y_segments=iter_y_segments,
            split_csv_tokens=core["_split_csv_tokens"],
            re_mod=core["re"],
        )

    def lint_check_global_md_dm_confusion(value: str) -> str | None:
        return linting.lint_check_global_md_dm_confusion(
            value,
            yearfmt=core["_yearfmt"],
            re_mod=core["re"],
        )

    def lint_check_invalid_weekday_names(value: str) -> str | None:
        return linting.lint_check_invalid_weekday_names(
            value,
            wd_abbr=core["_WD_ABBR"],
            re_mod=core["re"],
            difflib_mod=core["difflib"],
        )

    def lint_check_nth_weekday_suffixes(value: str) -> str | None:
        return linting.lint_check_nth_weekday_suffixes(value, re_mod=core["re"])

    def lint_check_unsat_pure_weekly_and(value: str) -> str | None:
        return linting.lint_check_unsat_pure_weekly_and(
            value,
            wd_abbr=core["_WD_ABBR"],
            split_csv_tokens=core["_split_csv_tokens"],
            re_mod=core["re"],
        )

    def lint_check_backward_quarter_ranges(value: str) -> str | None:
        return linting.lint_check_backward_quarter_ranges(value, re_mod=core["re"])

    def lint_collect_warnings(value: str) -> list[str]:
        return linting.lint_collect_warnings(value, re_mod=core["re"])

    def lint_anchor_expr(expr: str) -> tuple[str | None, list[str]]:
        return linting.lint_anchor_expr(
            expr,
            unwrap_quotes=core["_unwrap_quotes"],
            lint_expand_year_month_aliases=lint_expand_year_month_aliases,
            lint_check_weekly_delimiter_contract=lint_check_weekly_delimiter_contract,
            lint_check_yearly_segments=lint_check_yearly_segments,
            lint_check_global_md_dm_confusion=lint_check_global_md_dm_confusion,
            lint_check_invalid_weekday_names=lint_check_invalid_weekday_names,
            lint_check_nth_weekday_suffixes=lint_check_nth_weekday_suffixes,
            lint_check_unsat_pure_weekly_and=lint_check_unsat_pure_weekly_and,
            lint_check_backward_quarter_ranges=lint_check_backward_quarter_ranges,
            lint_collect_warnings=lint_collect_warnings,
            re_mod=core["re"],
        )

    return SimpleNamespace(
        _iter_y_segments=iter_y_segments,
        _lint_expand_year_month_aliases=lint_expand_year_month_aliases,
        _lint_check_weekly_delimiter_contract=lint_check_weekly_delimiter_contract,
        _lint_check_yearly_segments=lint_check_yearly_segments,
        _lint_check_global_md_dm_confusion=lint_check_global_md_dm_confusion,
        _lint_check_invalid_weekday_names=lint_check_invalid_weekday_names,
        _lint_check_nth_weekday_suffixes=lint_check_nth_weekday_suffixes,
        _lint_check_unsat_pure_weekly_and=lint_check_unsat_pure_weekly_and,
        _lint_check_backward_quarter_ranges=lint_check_backward_quarter_ranges,
        _lint_collect_warnings=lint_collect_warnings,
        lint_anchor_expr=lint_anchor_expr,
    )


__all__ = ("for_core",)
