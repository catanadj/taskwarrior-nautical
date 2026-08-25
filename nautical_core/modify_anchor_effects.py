"""Anchor omission-state assembly for the typed on-modify workflow."""

from __future__ import annotations

from typing import Any


def omit_dnf_from_parent(host: Any, task_mapping: dict[str, Any]):
    expr_str = (task_mapping.get("omit") or "").strip()
    omit_file = (task_mapping.get("omit_file") or "").strip()
    omit_dnf = None
    omit_dates: frozenset[Any] = frozenset()
    omit_descriptions: dict[Any, str] = {}
    if expr_str:
        try:
            omit_dnf = host._validate_omit_expr_cached(expr_str)
        except Exception as exc:
            raise ValueError(f"Invalid omit expression '{expr_str}': {exc}") from exc
    if omit_file:
        try:
            omit_files = host.core._import_sibling("omit_files")
            omit_dates, omit_descriptions = omit_files.load_omit_file_data(
                omit_file, getattr(host.core, "OMIT_FILE_DIR", "")
            )
        except Exception as exc:
            raise ValueError(f"Invalid omit_file '{omit_file}': {exc}") from exc
    if not omit_dnf and not omit_dates and not omit_descriptions:
        return "", None
    return expr_str, host._module("anchor_omit").combine_omit_state(
        omit_dnf=omit_dnf,
        omit_dates=omit_dates,
        omit_descriptions=omit_descriptions,
    )


__all__ = ("omit_dnf_from_parent",)
