"""Core-bound token formatting and normalization adapters."""

from __future__ import annotations

from typing import Any


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    core = namespace if namespace is not None else vars(module)

    def yearfmt():
        fmt = (core.get("ANCHOR_YEAR_FMT") or "MD").upper()
        return "DM" if fmt == "DM" else "MD"

    def tok(d: int, m: int) -> str:
        return f"{d:02d}-{m:02d}" if yearfmt() == "DM" else f"{m:02d}-{d:02d}"

    def tok_range(d1: int, m1: int, d2: int, m2: int) -> str:
        if yearfmt() == "DM":
            return f"{d1:02d}-{m1:02d}..{d2:02d}-{m2:02d}"
        return f"{m1:02d}-{d1:02d}..{m2:02d}-{d2:02d}"

    def safe_match(pattern, text: str, max_len: int = 256):
        if text is None:
            return None
        if len(text) > max_len:
            raise core["ParseError"]("Expression too complex")
        return pattern.match(text)

    def sanitize_text(v: str, max_len: int = 1024) -> str:
        return core["_common"].sanitize_text(v, max_len=max_len)

    def sanitize_task_strings(task: dict, max_len: int = 1024) -> None:
        core["_common"].sanitize_task_strings(task, max_len=max_len)

    def split_csv_tokens(spec: str) -> list[str]:
        return core["_common"].split_csv_tokens(spec)

    def split_csv_lower(spec: str) -> list[str]:
        return core["_common"].split_csv_lower(spec)

    def iso_week_index(d) -> int:
        iso = d.isocalendar()
        return iso.year * 53 + iso.week

    def month_index(d) -> int:
        return d.year * 12 + d.month

    def year_index(d) -> int:
        return d.year

    def static_month_last_day(mm: int) -> int:
        return core["_tokenutil"].static_month_last_day(mm)

    def month_from_alias(tok_value: str) -> int | None:
        return core["_tokenutil"].month_from_alias(tok_value)

    def year_full_months_span_token(m1: int, m2: int) -> str:
        return core["_year_tokens"].year_full_months_span_token(m1, m2, tok_range=tok_range)

    def rewrite_month_names_to_ranges(spec: str) -> str:
        return core["_year_tokens"].rewrite_month_names_to_ranges(spec, tok_range=tok_range)

    def unwrap_quotes(s: str) -> str:
        return core["_tokenutil"].unwrap_quotes(s)

    def year_full_month_range_token(mm: int) -> str:
        return core["_year_tokens"].year_full_month_range_token(mm, tok_range=tok_range)

    def mon_to_int(tok_value: str) -> int | None:
        return core["_tokenutil"].mon_to_int(tok_value)

    def expand_weekly_aliases(spec: str) -> str:
        return core["_tokenutil"].expand_weekly_aliases(spec)

    def expand_monthly_aliases(spec: str) -> str:
        return core["_tokenutil"].expand_monthly_aliases(spec)

    def normalize_weekday(s: str) -> str | None:
        return core["_tokenutil"].normalize_weekday(s)

    from types import SimpleNamespace

    return SimpleNamespace(
        _yearfmt=yearfmt,
        _tok=tok,
        _tok_range=tok_range,
        _safe_match=safe_match,
        sanitize_text=sanitize_text,
        sanitize_task_strings=sanitize_task_strings,
        _split_csv_tokens=split_csv_tokens,
        _split_csv_lower=split_csv_lower,
        _iso_week_index=iso_week_index,
        _month_index=month_index,
        _year_index=year_index,
        _static_month_last_day=static_month_last_day,
        _month_from_alias=month_from_alias,
        _year_full_months_span_token=year_full_months_span_token,
        _rewrite_month_names_to_ranges=rewrite_month_names_to_ranges,
        _unwrap_quotes=unwrap_quotes,
        _year_full_month_range_token=year_full_month_range_token,
        _mon_to_int=mon_to_int,
        _expand_weekly_aliases=expand_weekly_aliases,
        _expand_monthly_aliases=expand_monthly_aliases,
        _normalize_weekday=normalize_weekday,
    )


__all__ = ("for_core",)
