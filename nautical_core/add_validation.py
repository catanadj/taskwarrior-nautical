from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Callable


def validate_until_not_past(until_dt: Any, now_utc: datetime, *, core: Any) -> tuple[bool, str | None]:
    if not until_dt:
        return (True, None)
    grace = timedelta(minutes=1)
    if until_dt < (now_utc - grace):
        past_s = core.humanize_delta(until_dt, now_utc, use_months_days=False)
        return (False, f"chainUntil is in the past (was {past_s} ago)")
    return (True, None)


def check_due_in_past(due_dt: Any, now_utc: datetime, *, core: Any) -> tuple[bool, str | None]:
    if not due_dt:
        return (False, None)
    grace = timedelta(minutes=1)
    if due_dt < (now_utc - grace):
        ago_s = core.humanize_delta(due_dt, now_utc, use_months_days=False)
        return (True, f"Due date is in the past ({ago_s} ago).")
    return (False, None)


def validate_chain_duration_reasonable(
    until_dt: Any,
    now_utc: datetime,
    first_due: Any,
    kind: str,
    *,
    max_chain_duration_years: int,
) -> tuple[bool, str | None]:
    _ = first_due
    _ = kind
    if not until_dt:
        return (True, None)
    span = until_dt - now_utc
    years = span.days / 365.25
    if years > max_chain_duration_years:
        return (False, f"Chain extends {years:.1f} years into future.")
    return (True, None)


def validate_kind_not_conflicting(cp_str: Any, anchor_str: Any) -> tuple[bool, str | None]:
    has_cp = bool((cp_str or "").strip())
    has_anchor = bool((anchor_str or "").strip())
    if has_cp and has_anchor:
        return (False, "Cannot set both 'cp' and 'anchor'. Choose one.")
    return (True, None)


def validate_cpmax_positive(cpmax: Any) -> tuple[bool, str | None]:
    if cpmax <= 0:
        return (False, "chainMax must be > 0")
    return (True, None)


def safe_parse_datetime(
    s: Any,
    field_name: str,
    *,
    core: Any,
    diag: Callable[[str], None],
) -> tuple[datetime | None, str | None]:
    if not s:
        return (None, None)
    try:
        dt = core.parse_dt_any(s)
        if dt is None:
            return (None, f"{field_name}: Unrecognized datetime format '{s}'")
        return (dt, None)
    except ValueError as e:
        diag(f"{field_name} parse value error: {e}")
        return (None, f"{field_name}: Invalid datetime value")
    except Exception as e:
        diag(f"{field_name} parse unexpected error: {e}")
        return (None, f"{field_name}: Unexpected parsing error")


def validate_no_legacy_colon_ranges(expr: str) -> tuple[bool, str | None]:
    if not expr:
        return (True, None)
    expr = expr.strip()
    day_names = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
    for term in expr.split():
        clean_term = term.strip("()")
        if ":" in clean_term or "-" in clean_term:
            parts = re.split(r"[:\-]", clean_term)
            if len(parts) >= 2 and all(p.lower() in day_names for p in parts):
                legacy_example = clean_term
                return (
                    False,
                    f"Legacy weekly range '{legacy_example}' is not supported. Use '..' (e.g., 'w:mon..fri').",
                )
    return (True, None)


def safe_parse_duration(
    s: Any,
    field_name: str,
    *,
    core: Any,
    diag: Callable[[str], None],
) -> tuple[timedelta | None, str | None]:
    if not s:
        return (None, None)
    try:
        td = core.parse_cp_duration(s)
        if td is None:
            return (
                None,
                f"{field_name}: Invalid duration format '{s}' (expected: 3d, 2w, 1h, etc.)",
            )
        return (td, None)
    except ValueError as e:
        diag(f"{field_name} duration parse value error: {e}")
        return (None, f"{field_name}: Invalid duration value")
    except Exception as e:
        diag(f"{field_name} duration parse unexpected error: {e}")
        return (None, f"{field_name}: Unexpected parsing error")


def validate_anchor_syntax_strict(
    expr: str | list[list[dict[str, Any]]],
    *,
    validate_anchor_expr_cached: Callable[[str | list[list[dict[str, Any]]]], list[list[dict[str, Any]]]],
    core: Any,
    diag: Callable[[str], None],
) -> tuple[list[list[dict[str, Any]]] | None, str | None]:
    try:
        dnf = validate_anchor_expr_cached(expr)
        return dnf, None
    except Exception as e:
        parse_err_t = getattr(core, "ParseError", None)
        if parse_err_t is not None and isinstance(e, parse_err_t):
            return None, str(e)
        diag(f"anchor validation unexpected error: {e}")
        return None, "anchor syntax error"


def validate_anchor_mode(mode_str: Any) -> tuple[str, str | None]:
    mode = (mode_str or "skip").strip().lower()
    if mode not in ("skip", "all", "flex"):
        return (
            "skip",
            f"anchor_mode must be 'skip', 'all', or 'flex' (got '{mode}'). Defaulting to 'skip'.",
        )
    return (mode, None)
