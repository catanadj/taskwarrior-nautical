"""Public anchor parser API layered over the core parser implementation."""

from __future__ import annotations

import importlib
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
        normalize_anchor_input_to_dnf=module._normalize_anchor_input_to_dnf,
        assert_dnf_structure_strict=module._assert_dnf_structure_strict,
        validate_anchor_dnf_atoms_strict=module._validate_anchor_dnf_atoms_strict,
    )


def for_core(module: Any):
    """Create parser entry points bound to one core module instance."""
    return SimpleNamespace(
        build_acf=lambda expr: module._build_acf_impl(expr),
        resolve_anchor_presets=lambda expr, *, _seen=None: module._resolve_anchor_presets_impl(
            expr, _seen=_seen
        ),
        parse_anchor_expr_to_dnf=lambda s: _parse_anchor_expr_to_dnf_impl(module, s),
        parse_anchor_expr_to_dnf_cached=lambda s: module._parse_anchor_expr_to_dnf_cached_impl(s),
        validate_anchor_expr_strict=lambda expr: _validate_anchor_expr_strict_impl(module, expr),
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
