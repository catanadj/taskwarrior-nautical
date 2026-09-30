"""Public anchor parser API layered over the deps parser implementation."""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable
from .api_bindings import ApiBinding, core_namespace

from .core_context import CoreContext, ParserDependencies


@dataclass(frozen=True, slots=True)
class ParserOwnerDependencies:
    """Explicit collaborators required by the pure DNF parser owner."""

    normalize_input: Callable[..., Any]
    raise_bad_year_colons: Callable[..., Any]
    parse_atom: Callable[..., Any]
    parse_mods: Callable[..., Any]
    skip_ws: Callable[..., Any]
    rewrite_quarters: Callable[..., Any]
    rewrite_year_month: Callable[..., Any]
    validate_year_tokens: Callable[..., Any]
    validate_satisfiable: Callable[..., Any]
    max_terms: int
    parse_error: type[Exception]
    today: Callable[[], date]
    parser_dnf: Any
    resolve_presets: Callable[[str], str]


@dataclass(frozen=True, slots=True)
class ParserValidationDependencies:
    """Explicit collaborators required by strict anchor validation."""

    strict_validation: Any
    parse_cached: Callable[..., Any]
    parse_error: type[Exception]
    is_atom_like: Callable[..., Any]
    validate_weekly_spec: Callable[..., Any]
    validate_monthly_spec: Callable[..., Any]
    active_mod_keys: Callable[..., Any]
    validate_yearly_token_format: Callable[..., Any]
    position_selection: Any


def _core_module() -> Any:
    package = __package__ or "nautical_core"
    return sys.modules.get(package) or importlib.import_module(package)


def _parse_anchor_expr_to_dnf_impl(s: str, deps: ParserOwnerDependencies) -> Any:
    """Run the pure DNF parser using its explicit owner dependencies."""
    s = deps.resolve_presets(s)
    return deps.parser_dnf.parse_anchor_expr_to_dnf(
        s,
        normalize_anchor_expr_input=deps.normalize_input,
        raise_on_bad_colon_year_tokens=deps.raise_bad_year_colons,
        parse_anchor_atom_at=deps.parse_atom,
        parse_atom_mods=deps.parse_mods,
        skip_ws_pos=deps.skip_ws,
        rewrite_quarters_in_context=deps.rewrite_quarters,
        rewrite_year_month_aliases_in_context=deps.rewrite_year_month,
        validate_year_tokens_in_dnf=deps.validate_year_tokens,
        validate_and_terms_satisfiable=deps.validate_satisfiable,
        max_anchor_dnf_terms=deps.max_terms,
        parse_error_cls=deps.parse_error,
        today=deps.today,
    )


def _validate_anchor_expr_strict_impl(deps: ParserValidationDependencies, expr: Any) -> Any:
    """Run strict validation using its explicit dependency snapshot."""
    return deps.strict_validation.validate_anchor_expr_strict(
        expr,
        normalize_anchor_input_to_dnf=lambda value: _normalize_anchor_input_to_dnf(deps, value),
        assert_dnf_structure_strict=lambda value: _assert_dnf_structure_strict(deps, value),
        validate_anchor_dnf_atoms_strict=lambda value: _validate_anchor_dnf_atoms_strict(deps, value),
    )


def _normalize_anchor_input_to_dnf(deps: ParserValidationDependencies, expr: Any) -> Any:
    return deps.strict_validation.normalize_anchor_input_to_dnf(
        expr,
        parse_anchor_expr_to_dnf_cached=deps.parse_cached,
        parse_error_cls=deps.parse_error,
    )


def _assert_dnf_structure_strict(deps: ParserValidationDependencies, dnf: Any) -> Any:
    deps.strict_validation.assert_dnf_structure_strict(
        dnf,
        is_atom_like=deps.is_atom_like,
        parse_error_cls=deps.parse_error,
    )


def _validate_anchor_atom_strict(deps: ParserValidationDependencies, atom: dict) -> None:
    deps.strict_validation.validate_anchor_atom_strict(
        atom,
        validate_weekly_spec=deps.validate_weekly_spec,
        validate_monthly_spec=deps.validate_monthly_spec,
        active_mod_keys=deps.active_mod_keys,
        validate_yearly_token_format=deps.validate_yearly_token_format,
        parse_error_cls=deps.parse_error,
    )


def _validate_anchor_dnf_atoms_strict(deps: ParserValidationDependencies, dnf: Any) -> None:
    deps.strict_validation.validate_anchor_dnf_atoms_strict(
        dnf,
        validate_anchor_atom_strict=lambda atom: _validate_anchor_atom_strict(deps, atom),
        is_selection_node=deps.position_selection.is_selection_node,
        validate_selection_node=deps.position_selection.validate_public_selection_node,
        parse_error_cls=deps.parse_error,
    )


def for_core(module: Any = None, *, namespace: dict[str, Any] | None = None, context: CoreContext | None = None) -> ApiBinding:
    """Create parser entry points bound to one deps module instance."""
    if context is not None:
        deps = ParserDependencies.from_mapping(context.namespace)
        module = context
    else:
        if module is None and namespace is None:
            module = _core_module()
        deps = ParserDependencies.from_mapping(
            core_namespace(module, namespace, context, "parser_api")
        )
    parser_atoms = context.import_sibling("parsing.parser_atoms") if context is not None else deps["_parser_atoms"]
    parser_dnf = context.import_sibling("parsing.parser_dnf") if context is not None else deps["_parser_dnf"]
    parser_frontend = context.import_sibling("parsing.parser_frontend") if context is not None else deps["_parser_frontend"]
    quarter_binding_state: list[Any | None] = [None]
    position_selection = context.import_sibling("position_selection") if context is not None else deps["_position_selection"]
    validation_deps_state: list[ParserValidationDependencies | None] = [None]

    def validation_dependencies() -> ParserValidationDependencies:
        bound = validation_deps_state[0]
        if bound is None:
            parse_error = deps.get("ParseError")
            if parse_error is None:
                parser_models = (
                    context.import_sibling("parsing.parser_models")
                    if context is not None else deps["_import_sibling"]("parsing.parser_models")
                )
                parse_error = parser_models.ParseError
            bound = ParserValidationDependencies(
                strict_validation=(
                    context.import_sibling("strict_validation")
                    if context is not None else deps["_strict_validation"]
                ),
                parse_cached=deps["_parse_anchor_expr_to_dnf_cached_impl"],
                parse_error=parse_error,
                is_atom_like=deps["_is_atom_like"],
                validate_weekly_spec=deps["_validate_weekly_spec"],
                validate_monthly_spec=deps["_validate_monthly_spec"],
                active_mod_keys=deps["_active_mod_keys"],
                validate_yearly_token_format=deps["_validate_yearly_token_format"],
                position_selection=position_selection,
            )
            validation_deps_state[0] = bound
        return bound
    preset_ref_re = re.compile(r"@([A-Za-z][A-Za-z0-9_-]*)")

    def resolve_preset_refs(
        expr: str,
        *,
        presets: dict,
        table_name: str,
        label: str,
        _seen: tuple[str, ...] | frozenset[str] | None = None,
    ) -> str:
        raw = deps["_unwrap_quotes"](expr or "").strip()
        if not raw:
            return raw
        presets = dict(presets or {})
        seen_chain = tuple(sorted(_seen)) if isinstance(_seen, frozenset) else tuple(_seen or ())
        seen = set(seen_chain)

        def repl(match: Any) -> str:
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
                raise deps["ParseError"](
                    f"Unknown {label} preset '@{name}'.{hint} Define it under [{table_name}] in config-nautical.toml."
                )
            if name in seen:
                chain = " -> ".join([*(f"@{x}" for x in seen_chain), f"@{name}"])
                raise deps["ParseError"](f"Recursive {label} preset reference detected: {chain}")
            resolved = resolve_preset_refs(
                presets[name],
                presets=presets,
                table_name=table_name,
                label=label,
                _seen=(*seen_chain, name),
            )
            return f"({resolved})"

        return preset_ref_re.sub(repl, raw)

    def resolve_anchor_presets_impl(expr: str, *, _seen: Any = None) -> str:
        return resolve_preset_refs(
            expr,
            presets=deps["ANCHOR_PRESETS"],
            table_name="anchor_presets",
            label="anchor",
            _seen=_seen,
        )

    def resolve_omit_presets(expr: str, *, _seen: Any = None) -> str:
        return resolve_preset_refs(
            expr,
            presets=deps["OMIT_PRESETS"],
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
        except deps["ParseError"]:
            return raw
        return resolved[1:-1].strip() if resolved.startswith("(") and resolved.endswith(")") else resolved

    def anchor_preset_display(expr: str) -> tuple[str, str] | None:
        raw = deps["_unwrap_quotes"](expr or "").strip()
        match = re.match(r"^@([A-Za-z][A-Za-z0-9_-]*)$", raw)
        if not match:
            return None
        name = match.group(1).strip().lower()
        presets = dict(deps["ANCHOR_PRESETS"] or {})
        if name not in presets:
            return None
        return "Preset", f"@{name} → {preset_display_value(name, presets, table_name='anchor_presets', label='anchor')}"

    def omit_preset_display(expr: str) -> tuple[str, str] | None:
        raw = deps["_unwrap_quotes"](expr or "").strip()
        match = re.match(r"^@([A-Za-z][A-Za-z0-9_-]*)$", raw)
        if not match:
            return None
        name = match.group(1).strip().lower()
        presets = dict(deps["OMIT_PRESETS"] or {})
        if name not in presets:
            return None
        return "Omit preset", f"@{name} → {preset_display_value(name, presets, table_name='omit_presets', label='omit')}"

    def normalize_anchor_expr_input(value: str) -> str:
        return parser_frontend.normalize_anchor_expr_input(
            value,
            unwrap_quotes=deps["_unwrap_quotes"],
            rewrite_weekly_multi_time_atoms=deps["_rewrite_weekly_multi_time_atoms"],
            re_mod=deps["re"],
            parse_error_cls=deps["ParseError"],
        )

    def normalize_monthly_ordinal_spec(spec: str) -> str:
        return parser_atoms.normalize_monthly_ordinal_spec(spec, re_mod=deps["re"])

    def parse_hhmm(value: str) -> Any:
        return parser_atoms.parse_hhmm(value, hhmm_re=deps["_hhmm_re"])

    def parse_atom_head(head: str) -> Any:
        return parser_atoms.parse_atom_head(
            head,
            re_mod=deps["re"],
            parse_error_cls=deps["ParseError"],
        )

    def parse_atom_mods(mods_str: str) -> Any:
        return parser_atoms.parse_atom_mods(
            mods_str,
            split_csv_tokens=deps["_split_csv_tokens"],
            parse_hhmm=parse_hhmm,
            next_prev_wd_re=deps["_next_prev_wd_re"],
            weekdays=deps["_WEEKDAYS"],
            day_offset_re=deps["_day_offset_re"],
            parse_error_cls=deps["ParseError"],
        )

    def skip_ws_pos(value: str, index: int, length: int) -> int:
        return parser_frontend.skip_ws_pos(value, index, length)

    def raise_if_comma_joined_anchors(full_tail: str) -> None:
        parser_frontend.raise_if_comma_joined_anchors(
            full_tail,
            re_mod=deps["re"],
            parse_error_cls=deps["ParseError"],
        )

    def rewrite_quarters_in_context(dnf: Any) -> Any:
        quarter_binding = quarter_binding_state[0]
        if quarter_binding is None:
            quarter_api = (
                context.import_sibling("quarter_api")
                if context is not None else deps["_import_sibling"]("quarter_api")
            )
            quarter_binding = quarter_api.for_core(
                module=module,
                namespace=namespace,
                context=context,
            )
            quarter_binding_state[0] = quarter_binding
        return quarter_binding._rewrite_quarters_in_context(dnf)

    def build_anchor_atom_dnf(head: str, full_tail: str) -> Any:
        return parser_atoms.build_anchor_atom_dnf(
            head,
            full_tail,
            parse_atom_head=parse_atom_head,
            parse_group_with_inline_mods=deps["_parse_group_with_inline_mods"],
            normalize_monthly_ordinal_spec=normalize_monthly_ordinal_spec,
            split_csv_lower=deps["_split_csv_lower"],
            parse_atom_mods=parse_atom_mods,
            parse_error_cls=deps["ParseError"],
        )

    def parse_anchor_atom_at(value: str, index: int, length: int) -> Any:
        return parser_atoms.parse_anchor_atom_at(
            value,
            index,
            length,
            skip_ws_pos=skip_ws_pos,
            raise_if_comma_joined_anchors=raise_if_comma_joined_anchors,
            build_anchor_atom_dnf=build_anchor_atom_dnf,
            parse_error_cls=deps["ParseError"],
        )

    def yearly_pair_from_fmt(a: int, b: int, fmt: str) -> tuple[int, int]:
        return deps["_yearly_validation"].yearly_pair_from_fmt(a, b, fmt)

    def yearly_mmdd_error(mm: int, dd: int) -> str | None:
        return deps["_yearly_validation"].yearly_mmdd_error(mm, dd)

    def validate_yearly_token_allowlist(token: str, fmt: str) -> None:
        deps["_yearly_validation"].validate_yearly_token_allowlist(
            token,
            fmt,
            year_token_format_error_cls=deps["YearTokenFormatError"],
            month_from_alias=deps["_month_from_alias"],
        )

    def validate_yearly_token_detailed(token: str, fmt: str) -> tuple[str, str] | None:
        return deps["_yearly_validation"].validate_yearly_token_detailed(
            token,
            fmt,
            year_token_format_error_cls=deps["YearTokenFormatError"],
        )

    def validate_yearly_token_format(spec: str) -> Any:
        return deps["_yearly_validation"].validate_yearly_token_format(
            spec,
            yearfmt=deps["_yearfmt"],
            split_csv_lower=deps["_split_csv_lower"],
            year_token_format_error_cls=deps["YearTokenFormatError"],
            month_from_alias=deps["_month_from_alias"],
        )

    def validate_year_tokens_in_dnf(dnf: Any) -> Any:
        return deps["_yearly_validation"].validate_year_tokens_in_dnf(
            dnf,
            validate_yearly_token_format=validate_yearly_token_format,
        )

    def validate_yearly_token(token: str) -> Any:
        return deps["_yearly_validation"].validate_yearly_token(
            token,
            quarters=deps["_QUARTERS"],
            parse_y_token=deps["_parse_y_token"],
            parse_error_cls=deps["ParseError"],
        )

    def yearly_last_day(month: int) -> int:
        return deps["_yearly_validation"].yearly_last_day(month)

    def yearly_check_day_month(day: int, month: int, label: str, token: str) -> None:
        deps["_yearly_validation"].yearly_check_day_month(
            day,
            month,
            label,
            token,
            parse_error_cls=deps["ParseError"],
            month_full=deps["_natural_language"]._MONTH_FULL,
        )

    def validate_yearly_spec_token(token: str) -> None:
        deps["_yearly_validation"].validate_yearly_spec_token(
            token,
            parse_error_cls=deps["ParseError"],
            month_full=deps["_natural_language"]._MONTH_FULL,
        )

    def validate_yearly_spec(spec: str) -> Any:
        return deps["_yearly_validation"].validate_yearly_spec(
            spec,
            split_csv_lower=deps["_split_csv_lower"],
            validate_yearly_spec_token=validate_yearly_spec_token,
            parse_error_cls=deps["ParseError"],
        )

    leap_year_for_checks = 2028

    def weekday_set_from_weekly_atom(atom: Any) -> set[int]:
        return deps["_satisfiability"].weekday_set_from_weekly_atom(
            atom,
            weekly_spec_to_wset=deps["_weekly_spec_to_wset"],
        )

    def md_pairs_from_yearly_spec(spec: str) -> set[tuple[int, int]]:
        return deps["_satisfiability"].md_pairs_from_yearly_spec(
            spec,
            expand_yearly_cached=deps["expand_yearly_cached"],
            leap_year_for_checks=leap_year_for_checks,
        )

    def quick_weekly_and_check(term: list[dict]) -> None:
        deps["_satisfiability"].quick_weekly_and_check(
            term,
            weekday_set_from_weekly_atom=weekday_set_from_weekly_atom,
            and_term_unsatisfiable_cls=deps["AndTermUnsatisfiable"],
        )

    def quick_yearly_and_check(term: list[dict]) -> None:
        deps["_satisfiability"].quick_yearly_and_check(
            term,
            md_pairs_from_yearly_spec=md_pairs_from_yearly_spec,
            and_term_unsatisfiable_cls=deps["AndTermUnsatisfiable"],
        )

    def quick_moon_and_check(term: list[dict]) -> None:
        deps["_satisfiability"].quick_moon_and_check(
            term,
            and_term_unsatisfiable_cls=deps["AndTermUnsatisfiable"],
        )

    def term_has_any_match_within(term: list[dict], start: Any, seed: Any, years: int = 8) -> bool:
        return deps["_satisfiability"].term_has_any_match_within(
            term,
            start,
            seed,
            atom_matches_on=deps["atom_matches_on"],
            years=years,
        )

    def fatal_bad_colon_in_year_tail(tail: str) -> str | None:
        return parser_frontend.fatal_bad_colon_in_year_tail(
            tail,
            split_csv_tokens=deps["_split_csv_tokens"],
            re_mod=deps["re"],
            yearfmt=deps["_yearfmt"],
        )

    def raise_bad_year_colons(value: str) -> None:
        parser_frontend.raise_on_bad_colon_year_tokens(
            value,
            re_mod=deps["re"],
            fatal_bad_colon_in_year_tail=fatal_bad_colon_in_year_tail,
            parse_error_cls=deps["ParseError"],
        )

    def validate_and_terms_satisfiable(dnf: list[list[dict]], ref_d: Any) -> Any:
        for term in dnf:
            for factor in term:
                if position_selection.is_selection_node(factor):
                    validate_and_terms_satisfiable(factor.get("expr") or [], ref_d)
                    if not position_selection.seasonal_candidate_has_match(
                        factor,
                        matches_on=deps["atom_matches_on"],
                        default_seed=ref_d,
                    ):
                        scope = str(factor.get("scope") or "season")
                        mode = deps["_season_support"].active_mode()
                        boundary = (
                            f"the four {mode} seasonal windows"
                            if scope == "season"
                            else deps["_season_support"].season_boundary_description(scope)
                        )
                        raise deps["AndTermUnsatisfiable"](
                            f"@in-{scope} candidate expression has no dates within its {mode} "
                            f"{boundary} window."
                        )
        plain_dnf = [
            term for term in dnf
            if not any(position_selection.is_selection_node(factor) for factor in term)
        ]
        if not plain_dnf:
            return
        return deps["_satisfiability"].validate_and_terms_satisfiable(
            plain_dnf,
            ref_d,
            quick_weekly_and_check=quick_weekly_and_check,
            quick_yearly_and_check=quick_yearly_and_check,
            quick_moon_and_check=quick_moon_and_check,
            term_has_any_match_within=term_has_any_match_within,
            normalize_spec_for_acf=deps["_normalize_spec_for_acf"],
            month_from_alias=deps["_month_from_alias"],
            and_term_unsatisfiable_cls=deps["AndTermUnsatisfiable"],
        )

    def parse_anchor_expr_to_dnf_bound(s: str) -> Any:
        owner_deps = ParserOwnerDependencies(
            normalize_input=normalize_anchor_expr_input,
            raise_bad_year_colons=raise_bad_year_colons,
            parse_atom=parse_anchor_atom_at,
            parse_mods=parse_atom_mods,
            skip_ws=skip_ws_pos,
            rewrite_quarters=rewrite_quarters_in_context,
            rewrite_year_month=deps["_rewrite_year_month_aliases_in_context"],
            validate_year_tokens=validate_year_tokens_in_dnf,
            validate_satisfiable=validate_and_terms_satisfiable,
            max_terms=deps["MAX_ANCHOR_DNF_TERMS"],
            parse_error=deps["ParseError"],
            today=date.today,
            parser_dnf=parser_dnf,
            resolve_presets=resolve_anchor_presets,
        )
        return _parse_anchor_expr_to_dnf_impl(s, owner_deps)

    return ApiBinding.from_kwargs(
        build_acf=lambda expr: deps["_build_acf_impl"](expr),
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
        parse_anchor_expr_to_dnf=parse_anchor_expr_to_dnf_bound,
        parse_anchor_expr_to_dnf_cached=lambda s: deps["_parse_anchor_expr_to_dnf_cached_impl"](s),
        validate_anchor_expr_strict=lambda expr: _validate_anchor_expr_strict_impl(validation_dependencies(), expr),
        normalize_anchor_input_to_dnf=lambda expr: _normalize_anchor_input_to_dnf(validation_dependencies(), expr),
        assert_dnf_structure_strict=lambda dnf: _assert_dnf_structure_strict(validation_dependencies(), dnf),
        validate_anchor_atom_strict=lambda atom: _validate_anchor_atom_strict(validation_dependencies(), atom),
        validate_anchor_dnf_atoms_strict=lambda dnf: _validate_anchor_dnf_atoms_strict(validation_dependencies(), dnf),
    )


def build_acf(expr: str) -> str:
    return _compat_build_acf(expr)


def resolve_anchor_presets(expr: str, *, _seen: Any = None) -> str:
    return _core_module()._resolve_anchor_presets_impl(expr, _seen=_seen)


def resolve_omit_presets(expr: str, *, _seen: Any = None) -> str:
    return _core_module()._resolve_omit_presets_impl(expr, _seen=_seen)


def parse_anchor_expr_to_dnf(s: str) -> Any:
    return for_core(module=_core_module()).parse_anchor_expr_to_dnf(s)


def parse_anchor_expr_to_dnf_cached(s: str) -> Any:
    return _compat_parse_anchor_expr_to_dnf_cached(s)


def validate_anchor_expr_strict(expr: Any) -> Any:
    return _validate_anchor_expr_strict_impl(_compat_validation_dependencies(), expr)


def _compat_build_acf(expr: str) -> str:
    """Bridge the legacy public call to the facade's current ACF binding."""
    module = _core_module()
    return module._build_acf_impl(expr)


def _compat_parse_anchor_expr_to_dnf_cached(s: str) -> Any:
    """Bridge the legacy public call to the facade's cached parser binding."""
    module = _core_module()
    return module._parse_anchor_expr_to_dnf_cached_impl(s)


def _compat_validation_dependencies() -> ParserValidationDependencies:
    """Snapshot strict-validation collaborators at the compatibility edge."""
    module = _core_module()
    position_selection = module._position_selection
    return ParserValidationDependencies(
        strict_validation=module._strict_validation,
        parse_cached=module._parse_anchor_expr_to_dnf_cached_impl,
        parse_error=module.ParseError,
        is_atom_like=module._is_atom_like,
        validate_weekly_spec=module._validate_weekly_spec,
        validate_monthly_spec=module._validate_monthly_spec,
        active_mod_keys=module._active_mod_keys,
        validate_yearly_token_format=module._validate_yearly_token_format,
        position_selection=position_selection,
    )


__all__ = (
    "build_acf",
    "parse_anchor_expr_to_dnf",
    "parse_anchor_expr_to_dnf_cached",
    "resolve_anchor_presets",
    "resolve_omit_presets",
    "validate_anchor_expr_strict",
)
