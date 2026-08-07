"""Public scheduler entry points bound to one core facade instance."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def for_core(module: Any):
    """Create scheduler APIs without sharing state between core loaders."""
    return SimpleNamespace(
        expand_weekly_cached=module._expand_weekly_cached_impl,
        expand_weekly_cached_mods=module._expand_weekly_cached_mods_impl,
        expand_yearly_cached=module._expand_yearly_cached_impl,
        expand_monthly_cached=module._expand_monthly_cached_impl,
        expand_monthly_for_month=module._expand_monthly_for_month_impl,
        expand_weekly=module._expand_weekly_impl,
        expand_yearly_for_year_strict=module._expand_yearly_for_year_strict_impl,
        roll_apply=module._roll_apply_impl,
        apply_day_offset=module._apply_day_offset_impl,
        next_after_atom_with_mods=module._next_after_atom_with_mods_impl,
        atom_matches_on=module._atom_matches_on_impl,
        next_after_factor=module._next_after_factor_impl,
        factor_matches_on=module._factor_matches_on_impl,
        next_after_term=module._next_after_term_impl,
        next_after_expr=module._next_after_expr_impl,
    )


__all__ = ("for_core",)
