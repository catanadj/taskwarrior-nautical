"""Validation effects for ordinary on-modify recurrence edits."""

from __future__ import annotations

import re
from typing import Any


def anchor_error_message(anchor_expr: str, default_msg: str) -> str:
    if re.search(r"(?:^|[^A-Za-z])(w|m|y)(?:/\d+)?:", anchor_expr, re.IGNORECASE):
        return default_msg
    return f"{default_msg} (expected an anchor such as w:mon, m:15, or y:jul)"


def anchor_mode(host: Any, old: Any, new: Any) -> str:
    raw = str(new.get("anchor_mode") or old.get("anchor_mode") or "skip").strip()
    mode = raw.lower()
    aliases = {"all": "all", "skip": "skip", "flex": "flex"}
    normalized = aliases.get(mode)
    if normalized is None:
        host._panel("⚠ Anchor mode", [("Warning", f"Unknown anchor mode {raw!r}; using skip.")], kind="warning")
        normalized = "skip"
        new["anchor_mode"] = normalized
    elif new.get("anchor_mode"):
        new["anchor_mode"] = normalized
    return normalized.upper()


def validate_anchor(host: Any, old: Any, new: Any, anchor_expr: str) -> None:
    try:
        _, warns = host.core.lint_anchor_expr(anchor_expr)
        if warns:
            host._panel("ℹ️  Lint", [("Hint", warning) for warning in warns], kind="note")
        mode = anchor_mode(host, old, new)
        due = host._safe_dt(new.get("due") or old.get("due"))
        if host.core.ENABLE_ANCHOR_CACHE:
            host.core.build_and_cache_hints(anchor_expr, mode, default_due_dt=due, include_per_year=False)
        else:
            host.core.validate_anchor_expr_strict(anchor_expr)
    except TypeError:
        host.core.validate_anchor_expr_strict(anchor_expr)
    except Exception as exc:
        astronomy = host.core._import_sibling("astronomy")
        if astronomy.is_astronomy_error(exc):
            host._fail_and_exit("Invalid anchor", astronomy.scheduling_error_message(exc))
        host._fail_and_exit("Invalid anchor", anchor_error_message(anchor_expr, str(exc)))


def validate_omit(host: Any, anchor_expr: str, anchor_file_expr: str, omit_expr: str, omit_file: str) -> None:
    try:
        host._validate_shared_omit_on_modify(omit_expr)
        findings = host.core._import_sibling("hook_validation_pipeline").validate_recurrence_files(
            anchor_expr,
            anchor_file_expr,
            omit_expr,
            omit_file,
            load_anchor_file=host._load_anchor_file_dates,
            load_omit_file=host._load_omit_file_dates,
        )
    except Exception as exc:
        host._fail_and_exit("Invalid omit", str(exc))
        return
    if findings:
        finding = findings[0]
        host._fail_and_exit(f"Invalid {finding.field}", finding.reason)


def semantic_diff_value(old_text: str, new_text: str) -> str:
    return f"[dim]{old_text}[/] [cyan]→[/] [bold]{new_text}[/]"


__all__ = ("validate_anchor", "validate_omit", "semantic_diff_value")
