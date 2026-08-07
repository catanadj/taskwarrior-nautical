"""Public anchor parser API layered over the core parser implementation."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any


def _core_module():
    package = __package__ or "nautical_core"
    return sys.modules.get(package) or importlib.import_module(package)


def for_core(module: Any):
    """Create parser entry points bound to one core module instance."""
    return SimpleNamespace(
        build_acf=lambda expr: module._build_acf_impl(expr),
        resolve_anchor_presets=lambda expr, *, _seen=None: module._resolve_anchor_presets_impl(
            expr, _seen=_seen
        ),
        parse_anchor_expr_to_dnf=lambda s: module._parse_anchor_expr_to_dnf_impl(s),
        parse_anchor_expr_to_dnf_cached=lambda s: module._parse_anchor_expr_to_dnf_cached_impl(s),
        validate_anchor_expr_strict=lambda expr: module._validate_anchor_expr_strict_impl(expr),
    )


def build_acf(expr: str) -> str:
    return _core_module()._build_acf_impl(expr)


def resolve_anchor_presets(expr: str, *, _seen=None) -> str:
    return _core_module()._resolve_anchor_presets_impl(expr, _seen=_seen)


def parse_anchor_expr_to_dnf(s: str):
    return _core_module()._parse_anchor_expr_to_dnf_impl(s)


def parse_anchor_expr_to_dnf_cached(s: str):
    return _core_module()._parse_anchor_expr_to_dnf_cached_impl(s)


def validate_anchor_expr_strict(expr: Any):
    return _core_module()._validate_anchor_expr_strict_impl(expr)


__all__ = (
    "build_acf",
    "parse_anchor_expr_to_dnf",
    "parse_anchor_expr_to_dnf_cached",
    "resolve_anchor_presets",
    "validate_anchor_expr_strict",
)
