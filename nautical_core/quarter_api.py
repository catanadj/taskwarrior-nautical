"""Core-bound quarter and yearly-token rewrite adapters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    core = namespace if namespace is not None else vars(module)

    def yearly_tokens(term):
        return core["_quarter_helpers"].yearly_tokens(
            term,
            split_csv_tokens=core["_split_csv_tokens"],
        )

    def monthly_tokens(term):
        return core["_quarter_helpers"].monthly_tokens(
            term,
            split_csv_tokens=core["_split_csv_tokens"],
        )

    def quarters_from_first_month_tokens(tokens):
        return core["_quarter_helpers"].quarters_from_tokens(
            tokens,
            token_rev=core["_Q_FIRST_MONTH_TOKEN_REV"],
        )

    def quarters_from_start_day_tokens(tokens):
        return core["_quarter_helpers"].quarters_from_tokens(
            tokens,
            token_rev=core["_Q_START_DAY_REV"],
        )

    def quarters_from_end_day_tokens(tokens):
        return core["_quarter_helpers"].quarters_from_tokens(
            tokens,
            token_rev=core["_Q_END_DAY_REV"],
        )

    def format_quarter_set(values):
        return core["_quarter_helpers"].format_quarter_set(values)

    def rewrite_quarter_spec_mode(spec: str, mode: str, meta_out=None) -> str:
        return core["_quarter_rewrite"].rewrite_quarter_spec_mode(
            spec,
            mode,
            meta_out=meta_out,
            split_csv_lower=core["_split_csv_lower"],
            tok_range=core["_tok_range"],
            static_month_last_day=core["_static_month_last_day"],
            quarter_pos_month=core["_QUARTER_POS_MONTH"],
            re_mod=core["re"],
        )

    def quarter_atom_spec(atom: dict) -> str:
        return core["_quarter_selector"].quarter_atom_spec(atom)

    def has_quarter_tokens(spec: str) -> bool:
        return core["_quarter_selector"].has_quarter_tokens(
            spec,
            split_csv_lower=core["_split_csv_lower"],
            re_mod=core["re"],
        )

    def has_plain_quarter_tokens(spec: str) -> bool:
        return core["_quarter_selector"].has_plain_quarter_tokens(
            spec,
            split_csv_lower=core["_split_csv_lower"],
            re_mod=core["re"],
        )

    def is_start_month_selector(tok: str) -> bool:
        return core["_quarter_selector"].is_start_month_selector(
            tok,
            parse_error_cls=core["ParseError"],
            safe_match=core["_safe_match"],
            nth_weekday_re=core["_nth_weekday_re"],
        )

    def is_end_month_selector(tok: str) -> bool:
        return core["_quarter_selector"].is_end_month_selector(
            tok,
            parse_error_cls=core["ParseError"],
            safe_match=core["_safe_match"],
            nth_weekday_re=core["_nth_weekday_re"],
            bd_re=core["_bd_re"],
        )

    def quarter_month_selector_mode(m_atoms: list[dict]) -> str:
        return core["_quarter_selector"].quarter_month_selector_mode(
            m_atoms,
            parse_error_cls=core["ParseError"],
            expand_monthly_aliases=core["_expand_monthly_aliases"],
            split_csv_tokens=core["_split_csv_tokens"],
            is_start_month_selector=is_start_month_selector,
            is_end_month_selector=is_end_month_selector,
        )

    def term_quarter_rewrite_mode(y_atoms: list[dict], m_atoms: list[dict]) -> str:
        return core["_quarter_selector"].term_quarter_rewrite_mode(
            y_atoms,
            m_atoms,
            quarter_atom_spec=quarter_atom_spec,
            has_plain_quarter_tokens=has_plain_quarter_tokens,
            quarter_month_selector_mode=quarter_month_selector_mode,
        )

    def rewrite_quarter_year_atoms(y_atoms: list[dict], mode: str) -> None:
        core["_quarter_rewrite"].rewrite_quarter_year_atoms(
            y_atoms,
            mode,
            quarter_atom_spec=quarter_atom_spec,
            has_quarter_tokens=has_quarter_tokens,
            rewrite_quarter_spec_mode=rewrite_quarter_spec_mode,
        )

    def rewrite_quarters_in_context(dnf):
        return core["_quarter_rewrite"].rewrite_quarters_in_context(
            dnf,
            has_quarter_tokens=has_quarter_tokens,
            quarter_atom_spec=quarter_atom_spec,
            term_quarter_rewrite_mode=term_quarter_rewrite_mode,
            rewrite_quarter_year_atoms=rewrite_quarter_year_atoms,
        )

    return SimpleNamespace(
        _yearly_tokens=yearly_tokens,
        _monthly_tokens=monthly_tokens,
        _quarters_from_first_month_tokens=quarters_from_first_month_tokens,
        _quarters_from_start_day_tokens=quarters_from_start_day_tokens,
        _quarters_from_end_day_tokens=quarters_from_end_day_tokens,
        _format_quarter_set=format_quarter_set,
        _rewrite_quarter_spec_mode=rewrite_quarter_spec_mode,
        _quarter_atom_spec=quarter_atom_spec,
        _has_quarter_tokens=has_quarter_tokens,
        _has_plain_quarter_tokens=has_plain_quarter_tokens,
        _is_start_month_selector=is_start_month_selector,
        _is_end_month_selector=is_end_month_selector,
        _quarter_month_selector_mode=quarter_month_selector_mode,
        _term_quarter_rewrite_mode=term_quarter_rewrite_mode,
        _rewrite_quarter_year_atoms=rewrite_quarter_year_atoms,
        _rewrite_quarters_in_context=rewrite_quarters_in_context,
    )


__all__ = ("for_core",)
