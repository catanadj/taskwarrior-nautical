"""Public anchor parser API layered over the core parser implementation."""

from __future__ import annotations

import importlib
import re
import sys
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
        today=module.date.today,
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

    return SimpleNamespace(
        build_acf=lambda expr: module._build_acf_impl(expr),
        _resolve_preset_refs=resolve_preset_refs,
        _resolve_anchor_presets_impl=resolve_anchor_presets_impl,
        resolve_omit_presets=resolve_omit_presets,
        anchor_preset_display=anchor_preset_display,
        omit_preset_display=omit_preset_display,
        _normalize_anchor_expr_input=normalize_anchor_expr_input,
        _normalize_monthly_ordinal_spec=normalize_monthly_ordinal_spec,
        _build_anchor_atom_dnf=build_anchor_atom_dnf,
        _parse_anchor_atom_at=parse_anchor_atom_at,
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
    "validate_anchor_expr_strict",
)
