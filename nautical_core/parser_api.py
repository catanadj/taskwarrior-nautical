"""Public anchor parser API layered over the core parser implementation."""

from __future__ import annotations

import importlib
import re
import sys
from datetime import date
from types import SimpleNamespace
from typing import Any


def _core_module():
    package = __package__ or "nautical_core"
    return sys.modules.get(package) or importlib.import_module(package)


def _parse_anchor_expr_to_dnf_impl(module: Any, s: str):
    """Run the parser pipeline against one isolated core facade."""
    s = module.resolve_anchor_presets(s)
    return module._parser_dnf.parse_anchor_expr_to_dnf(
        s,
        normalize_anchor_expr_input=module._normalize_anchor_expr_input,
        raise_on_bad_colon_year_tokens=module._raise_on_bad_colon_year_tokens,
        parse_anchor_atom_at=module._parse_anchor_atom_at,
        parse_atom_mods=module._parse_atom_mods,
        skip_ws_pos=module._skip_ws_pos,
        rewrite_quarters_in_context=module._rewrite_quarters_in_context,
        rewrite_year_month_aliases_in_context=module._rewrite_year_month_aliases_in_context,
        validate_year_tokens_in_dnf=module._validate_year_tokens_in_dnf,
        validate_and_terms_satisfiable=module._validate_and_terms_satisfiable,
        max_anchor_dnf_terms=module.MAX_ANCHOR_DNF_TERMS,
        parse_error_cls=module.ParseError,
        today=date.today,
    )


def _validate_anchor_expr_strict_impl(module: Any, expr: Any):
    """Run strict validation against one isolated core facade."""
    return module._strict_validation.validate_anchor_expr_strict(
        expr,
        normalize_anchor_input_to_dnf=lambda value: _normalize_anchor_input_to_dnf(module, value),
        assert_dnf_structure_strict=lambda value: _assert_dnf_structure_strict(module, value),
        validate_anchor_dnf_atoms_strict=lambda value: _validate_anchor_dnf_atoms_strict(module, value),
    )


def _normalize_anchor_input_to_dnf(module: Any, expr):
    return module._strict_validation.normalize_anchor_input_to_dnf(
        expr,
        parse_anchor_expr_to_dnf_cached=module.parse_anchor_expr_to_dnf_cached,
        parse_error_cls=module.ParseError,
    )


def _assert_dnf_structure_strict(module: Any, dnf):
    module._strict_validation.assert_dnf_structure_strict(
        dnf,
        is_atom_like=module._is_atom_like,
        parse_error_cls=module.ParseError,
    )


def _validate_anchor_atom_strict(module: Any, atom: dict) -> None:
    module._strict_validation.validate_anchor_atom_strict(
        atom,
        validate_weekly_spec=module._validate_weekly_spec,
        validate_monthly_spec=module._validate_monthly_spec,
        active_mod_keys=module._active_mod_keys,
        validate_yearly_token_format=module._validate_yearly_token_format,
        parse_error_cls=module.ParseError,
    )


def _validate_anchor_dnf_atoms_strict(module: Any, dnf) -> None:
    module._strict_validation.validate_anchor_dnf_atoms_strict(
        dnf,
        validate_anchor_atom_strict=lambda atom: _validate_anchor_atom_strict(module, atom),
        is_selection_node=module._position_selection.is_selection_node,
        validate_selection_node=module._position_selection.validate_public_selection_node,
        parse_error_cls=module.ParseError,
    )


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    """Create parser entry points bound to one core module instance."""
    core = namespace if namespace is not None else vars(module)
    preset_ref_re = re.compile(r"@([A-Za-z][A-Za-z0-9_-]*)")

    def resolve_preset_refs(
        expr: str,
        *,
        presets: dict,
        table_name: str,
        label: str,
        _seen: tuple[str, ...] | frozenset[str] | None = None,
    ) -> str:
        raw = core["_unwrap_quotes"](expr or "").strip()
        if not raw:
            return raw
        presets = dict(presets or {})
        seen_chain = tuple(sorted(_seen)) if isinstance(_seen, frozenset) else tuple(_seen or ())
        seen = set(seen_chain)

        def repl(match):
            start = match.start()
            if start > 0 and raw[start - 1] not in " \t\r\n(|+,":
                return match.group(0)
            end = match.end()
            if end < len(raw) and raw[end] == "=":
                return match.group(0)
            name = match.group(1).strip().lower()
            if name not in presets:
                available = ", ".join(f"@{item}" for item in sorted(presets))
                hint = f" Available {label} presets: {available}." if presets else f" No {label} presets are configured."
                raise core["ParseError"](
                    f"Unknown {label} preset '@{name}'.{hint} Define it under [{table_name}] in config-nautical.toml."
                )
            if name in seen:
                chain = " -> ".join([*(f"@{x}" for x in seen_chain), f"@{name}"])
                raise core["ParseError"](f"Recursive {label} preset reference detected: {chain}")
            resolved = resolve_preset_refs(
                presets[name],
                presets=presets,
                table_name=table_name,
                label=label,
                _seen=(*seen_chain, name),
            )
            return f"({resolved})"

        return preset_ref_re.sub(repl, raw)

    def resolve_anchor_presets_impl(expr: str, *, _seen=None) -> str:
        return resolve_preset_refs(
            expr,
            presets=core["ANCHOR_PRESETS"],
            table_name="anchor_presets",
            label="anchor",
            _seen=_seen,
        )

    def resolve_omit_presets(expr: str, *, _seen=None) -> str:
        return resolve_preset_refs(
            expr,
            presets=core["OMIT_PRESETS"],
            table_name="omit_presets",
            label="omit",
            _seen=_seen,
        )

    def preset_display_value(name: str, presets: dict, *, table_name: str, label: str) -> str:
        raw = str(presets[name] or "").strip()
        try:
            resolved = resolve_preset_refs(
                raw,
                presets=presets,
                table_name=table_name,
                label=label,
                _seen=(name,),
            ).strip()
        except core["ParseError"]:
            return raw
        return resolved[1:-1].strip() if resolved.startswith("(") and resolved.endswith(")") else resolved

    def anchor_preset_display(expr: str) -> tuple[str, str] | None:
        raw = core["_unwrap_quotes"](expr or "").strip()
        match = re.match(r"^@([A-Za-z][A-Za-z0-9_-]*)$", raw)
        if not match:
            return None
        name = match.group(1).strip().lower()
        presets = dict(core["ANCHOR_PRESETS"] or {})
        if name not in presets:
            return None
        return "Preset", f"@{name} → {preset_display_value(name, presets, table_name='anchor_presets', label='anchor')}"

    def omit_preset_display(expr: str) -> tuple[str, str] | None:
        raw = core["_unwrap_quotes"](expr or "").strip()
        match = re.match(r"^@([A-Za-z][A-Za-z0-9_-]*)$", raw)
        if not match:
            return None
        name = match.group(1).strip().lower()
        presets = dict(core["OMIT_PRESETS"] or {})
        if name not in presets:
            return None
        return "Omit preset", f"@{name} → {preset_display_value(name, presets, table_name='omit_presets', label='omit')}"

    def normalize_anchor_expr_input(value: str) -> str:
        return core["_parser_frontend"].normalize_anchor_expr_input(
            value,
            unwrap_quotes=core["_unwrap_quotes"],
            rewrite_weekly_multi_time_atoms=core["_rewrite_weekly_multi_time_atoms"],
            re_mod=core["re"],
            parse_error_cls=core["ParseError"],
        )

    def normalize_monthly_ordinal_spec(spec: str) -> str:
        return core["_parser_atoms"].normalize_monthly_ordinal_spec(spec, re_mod=core["re"])

    def build_anchor_atom_dnf(head: str, full_tail: str):
        return core["_parser_atoms"].build_anchor_atom_dnf(
            head,
            full_tail,
            parse_atom_head=core["_parse_atom_head"],
            parse_group_with_inline_mods=core["_parse_group_with_inline_mods"],
            normalize_monthly_ordinal_spec=normalize_monthly_ordinal_spec,
            split_csv_lower=core["_split_csv_lower"],
            parse_atom_mods=core["_parse_atom_mods"],
            parse_error_cls=core["ParseError"],
        )

    def parse_anchor_atom_at(value: str, index: int, length: int):
        return core["_parser_atoms"].parse_anchor_atom_at(
            value,
            index,
            length,
            skip_ws_pos=core["_skip_ws_pos"],
            raise_if_comma_joined_anchors=core["_raise_if_comma_joined_anchors"],
            build_anchor_atom_dnf=build_anchor_atom_dnf,
            parse_error_cls=core["ParseError"],
        )

    def yearly_pair_from_fmt(a: int, b: int, fmt: str) -> tuple[int, int]:
        return core["_yearly_validation"].yearly_pair_from_fmt(a, b, fmt)

    def yearly_mmdd_error(mm: int, dd: int) -> str | None:
        return core["_yearly_validation"].yearly_mmdd_error(mm, dd)

    def validate_yearly_token_allowlist(token: str, fmt: str) -> None:
        core["_yearly_validation"].validate_yearly_token_allowlist(
            token,
            fmt,
            year_token_format_error_cls=core["YearTokenFormatError"],
            month_from_alias=core["_month_from_alias"],
        )

    def validate_yearly_token_detailed(token: str, fmt: str) -> tuple[str, str] | None:
        return core["_yearly_validation"].validate_yearly_token_detailed(
            token,
            fmt,
            year_token_format_error_cls=core["YearTokenFormatError"],
        )

    def validate_yearly_token_format(spec: str):
        return core["_yearly_validation"].validate_yearly_token_format(
            spec,
            yearfmt=core["_yearfmt"],
            split_csv_lower=core["_split_csv_lower"],
            year_token_format_error_cls=core["YearTokenFormatError"],
            month_from_alias=core["_month_from_alias"],
        )

    def validate_year_tokens_in_dnf(dnf):
        return core["_yearly_validation"].validate_year_tokens_in_dnf(
            dnf,
            validate_yearly_token_format=validate_yearly_token_format,
        )

    def validate_yearly_token(token: str):
        return core["_yearly_validation"].validate_yearly_token(
            token,
            quarters=core["_QUARTERS"],
            parse_y_token=core["_parse_y_token"],
            parse_error_cls=core["ParseError"],
        )

    def yearly_last_day(month: int) -> int:
        return core["_yearly_validation"].yearly_last_day(month)

    def yearly_check_day_month(day: int, month: int, label: str, token: str) -> None:
        core["_yearly_validation"].yearly_check_day_month(
            day,
            month,
            label,
            token,
            parse_error_cls=core["ParseError"],
            month_full=core["_natural_language"]._MONTH_FULL,
        )

    def validate_yearly_spec_token(token: str) -> None:
        core["_yearly_validation"].validate_yearly_spec_token(
            token,
            parse_error_cls=core["ParseError"],
            month_full=core["_natural_language"]._MONTH_FULL,
        )

    def validate_yearly_spec(spec: str):
        return core["_yearly_validation"].validate_yearly_spec(
            spec,
            split_csv_lower=core["_split_csv_lower"],
            validate_yearly_spec_token=validate_yearly_spec_token,
            parse_error_cls=core["ParseError"],
        )

    leap_year_for_checks = 2028

    def weekday_set_from_weekly_atom(atom) -> set[int]:
        return core["_satisfiability"].weekday_set_from_weekly_atom(
            atom,
            weekly_spec_to_wset=core["_weekly_spec_to_wset"],
        )

    def md_pairs_from_yearly_spec(spec: str) -> set[tuple[int, int]]:
        return core["_satisfiability"].md_pairs_from_yearly_spec(
            spec,
            expand_yearly_cached=core["expand_yearly_cached"],
            leap_year_for_checks=leap_year_for_checks,
        )

    def quick_weekly_and_check(term: list[dict]) -> None:
        core["_satisfiability"].quick_weekly_and_check(
            term,
            weekday_set_from_weekly_atom=weekday_set_from_weekly_atom,
            and_term_unsatisfiable_cls=core["AndTermUnsatisfiable"],
        )

    def quick_yearly_and_check(term: list[dict]) -> None:
        core["_satisfiability"].quick_yearly_and_check(
            term,
            md_pairs_from_yearly_spec=md_pairs_from_yearly_spec,
            and_term_unsatisfiable_cls=core["AndTermUnsatisfiable"],
        )

    def quick_moon_and_check(term: list[dict]) -> None:
        core["_satisfiability"].quick_moon_and_check(
            term,
            and_term_unsatisfiable_cls=core["AndTermUnsatisfiable"],
        )

    def term_has_any_match_within(term: list[dict], start, seed, years: int = 8) -> bool:
        return core["_satisfiability"].term_has_any_match_within(
            term,
            start,
            seed,
            atom_matches_on=core["atom_matches_on"],
            years=years,
        )

    def validate_and_terms_satisfiable(dnf: list[list[dict]], ref_d):
        for term in dnf:
            for factor in term:
                if core["_position_selection"].is_selection_node(factor):
                    validate_and_terms_satisfiable(factor.get("expr") or [], ref_d)
                    if not core["_position_selection"].seasonal_candidate_has_match(
                        factor,
                        matches_on=core["atom_matches_on"],
                        default_seed=ref_d,
                    ):
                        scope = str(factor.get("scope") or "season")
                        boundary = core["_season_support"].fixed_season_boundary_description(scope)
                        raise core["AndTermUnsatisfiable"](
                            f"@in-{scope} candidate expression has no dates within its fixed "
                            f"{boundary} window."
                        )
        plain_dnf = [
            term for term in dnf
            if not any(core["_position_selection"].is_selection_node(factor) for factor in term)
        ]
        if not plain_dnf:
            return
        return core["_satisfiability"].validate_and_terms_satisfiable(
            plain_dnf,
            ref_d,
            quick_weekly_and_check=quick_weekly_and_check,
            quick_yearly_and_check=quick_yearly_and_check,
            quick_moon_and_check=quick_moon_and_check,
            term_has_any_match_within=term_has_any_match_within,
            normalize_spec_for_acf=core["_normalize_spec_for_acf"],
            month_from_alias=core["_month_from_alias"],
            and_term_unsatisfiable_cls=core["AndTermUnsatisfiable"],
        )

    return SimpleNamespace(
        build_acf=lambda expr: module._build_acf_impl(expr),
        _resolve_preset_refs=resolve_preset_refs,
        _resolve_anchor_presets_impl=resolve_anchor_presets_impl,
        _resolve_omit_presets_impl=resolve_omit_presets,
        resolve_omit_presets=resolve_omit_presets,
        anchor_preset_display=anchor_preset_display,
        omit_preset_display=omit_preset_display,
        _normalize_anchor_expr_input=normalize_anchor_expr_input,
        _normalize_monthly_ordinal_spec=normalize_monthly_ordinal_spec,
        _build_anchor_atom_dnf=build_anchor_atom_dnf,
        _parse_anchor_atom_at=parse_anchor_atom_at,
        _yearly_pair_from_fmt=yearly_pair_from_fmt,
        _yearly_mmdd_error=yearly_mmdd_error,
        _validate_yearly_token_allowlist=validate_yearly_token_allowlist,
        _validate_yearly_token_detailed=validate_yearly_token_detailed,
        _validate_yearly_token_format=validate_yearly_token_format,
        _validate_year_tokens_in_dnf=validate_year_tokens_in_dnf,
        _validate_yearly_token=validate_yearly_token,
        _yearly_last_day=yearly_last_day,
        _yearly_check_day_month=yearly_check_day_month,
        _validate_yearly_spec_token=validate_yearly_spec_token,
        _validate_yearly_spec=validate_yearly_spec,
        _weekday_set_from_weekly_atom=weekday_set_from_weekly_atom,
        _md_pairs_from_yearly_spec=md_pairs_from_yearly_spec,
        _quick_weekly_and_check=quick_weekly_and_check,
        _quick_yearly_and_check=quick_yearly_and_check,
        _quick_moon_and_check=quick_moon_and_check,
        _term_has_any_match_within=term_has_any_match_within,
        _validate_and_terms_satisfiable=validate_and_terms_satisfiable,
        resolve_anchor_presets=resolve_anchor_presets_impl,
        parse_anchor_expr_to_dnf=lambda s: _parse_anchor_expr_to_dnf_impl(module, s),
        parse_anchor_expr_to_dnf_cached=lambda s: module._parse_anchor_expr_to_dnf_cached_impl(s),
        validate_anchor_expr_strict=lambda expr: _validate_anchor_expr_strict_impl(module, expr),
        normalize_anchor_input_to_dnf=lambda expr: _normalize_anchor_input_to_dnf(module, expr),
        assert_dnf_structure_strict=lambda dnf: _assert_dnf_structure_strict(module, dnf),
        validate_anchor_atom_strict=lambda atom: _validate_anchor_atom_strict(module, atom),
        validate_anchor_dnf_atoms_strict=lambda dnf: _validate_anchor_dnf_atoms_strict(module, dnf),
    )


def build_acf(expr: str) -> str:
    return _core_module()._build_acf_impl(expr)


def resolve_anchor_presets(expr: str, *, _seen=None) -> str:
    return _core_module()._resolve_anchor_presets_impl(expr, _seen=_seen)


def resolve_omit_presets(expr: str, *, _seen=None) -> str:
    return _core_module()._resolve_omit_presets_impl(expr, _seen=_seen)


def parse_anchor_expr_to_dnf(s: str):
    return _parse_anchor_expr_to_dnf_impl(_core_module(), s)


def parse_anchor_expr_to_dnf_cached(s: str):
    return _core_module()._parse_anchor_expr_to_dnf_cached_impl(s)


def validate_anchor_expr_strict(expr: Any):
    return _validate_anchor_expr_strict_impl(_core_module(), expr)


__all__ = (
    "build_acf",
    "parse_anchor_expr_to_dnf",
    "parse_anchor_expr_to_dnf_cached",
    "resolve_anchor_presets",
    "resolve_omit_presets",
    "validate_anchor_expr_strict",
)
